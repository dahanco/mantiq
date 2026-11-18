"""tools/mtp_from_gguf.py - the MTP draft layer's runtime files (what tools/mtp_rt.py writes) straight from a GGUF
that ships its NextN block, so drafting needs no second download.

    python tools/mtp_from_gguf.py --gguf <model>.gguf --out <pack>/mtp

Two architectures are supported, and they write DIFFERENT expert formats because `src/core/mtp.cpp` reads the
experts differently on each:

  qwen4exp    the 512 experts as Q2_0 blobs (gate/up rows interleaved, then down; codes, then scales), the
              drafter's original format (`cpu::BLOB`).  CYBER-FROST-3.8-PS's blk.48.
  qwen35moe   the 256 experts as the block's OWN GGUF blocks - [gate rows | up rows | down rows], raw Q4_K/Q6_K
              bytes, exactly what `native_expert_layout` describes.  No re-quantization: the bytes the model
              ships are the bytes the grouped native kernel reads.  Qwopus3.6-35B-A3B's blk.40.
              An extra `experts.txt` line (`gu_type up_type d_type n_embd n_ff`) tells the loader which layout it got.

The block's tensors are dequantized (gguf-py) and written as the drafter reads them (src/core/mtp.cpp):
  dense.bin     the projections as Q8_0, the router/norms as BF16/F32
  dense.txt     one line per tensor: name kind rows cols offset bytes

A draft's quality only moves the acceptance rate: every drafted token is checked by the model itself, so the
answer is the same with or without the drafter.

The GGUF holds the trunk's norms with the Gemma "+1" already applied in some conversions; the NextN block's two
input norms (enorm, hnorm) are taken as stored too (--input-norms-plus-one adds it, for a file that kept the raw
gammas).  qwen35moe's norms are plain RMSNorm and the flag is not needed.  eh_proj is [embedding | hidden] along
its input (--eh-order hidden-first for the other order).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from mtp_rt import blob_of, q8_0  # noqa: E402  (the runtime layout, ggml's Q8_0 rounding)

# ---- qwen4exp's fixed shape (the Q2_0 blob layout is built around it)
H4, FF4, NE4 = 2560, 640, 512


def q2_0(w: np.ndarray) -> np.ndarray:
    """tools/mtp_pack.py's Q2_0 (64-value blocks, grid {-1,0,1,2} x d, the scale chosen per block for the least squared
    error) - kept here so this tool needs no BF16 checkpoint."""
    x = w.reshape(-1, 64).astype(np.float32)
    amax = np.abs(x).max(axis=1, keepdims=True)
    best_err = np.full((x.shape[0], 1), np.inf, dtype=np.float32)
    best_d = np.zeros_like(amax)
    for f in np.linspace(0.5, 1.0, 17, dtype=np.float32):
        d = amax * f
        inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        q = np.clip(np.rint(x * inv), -1, 2)
        err = ((q * d - x) ** 2).sum(axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)
    d16 = best_d.astype(np.float16)
    d = d16.astype(np.float32)
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    codes = (np.clip(np.rint(x * inv), -1, 2) + 1).astype(np.uint8)
    c = codes.reshape(-1, 16, 4)
    packed = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    out = np.empty((x.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.reshape(-1)


def bf16(x: np.ndarray) -> bytes:
    """float32 -> BF16, round to nearest even."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="the model's GGUF (the shard holding the NextN block)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eh-order", choices=("emb-first", "hidden-first"), default="emb-first")
    ap.add_argument("--input-norms-plus-one", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    r = gguf.GGUFReader(a.gguf)
    arch = str(bytes(r.fields["general.architecture"].parts[-1]), "utf-8")
    kv = lambda k: int(r.fields[f"{arch}.{k}"].parts[-1][0]) if f"{arch}.{k}" in r.fields else None  # noqa: E731
    nextn, blocks = kv("nextn_predict_layers"), kv("block_count")
    if arch not in ("qwen4exp", "qwen35moe") or not nextn or not blocks:
        print(f"{a.gguf}: not a GGUF with a NextN block (architecture {arch}, nextn_predict_layers {nextn})")
        return 2
    b = blocks - nextn
    tens = {t.name: t for t in r.tensors}

    def f32(name: str) -> np.ndarray:
        t = tens[f"blk.{b}.{name}"]
        return np.asarray(dequantize(np.asarray(t.data), t.tensor_type), dtype=np.float32).reshape(
            [int(x) for x in reversed(t.shape)])

    def raw(name: str) -> np.ndarray:
        return np.ascontiguousarray(tens[f"blk.{b}.{name}"].data)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plus = 1.0 if a.input_norms_plus_one else 0.0

    if arch == "qwen35moe":
        # the expert FF width is `expert_feed_forward_length` in llama.cpp's qwen35moe (512); some conversions
        # only carry the plain `feed_forward_length`
        H, NE = kv("embedding_length"), kv("expert_count")
        FF = kv("expert_feed_forward_length") or kv("feed_forward_length")
        if H is None or FF is None or NE is None:
            print(f"{a.gguf}: qwen35moe metadata is incomplete (embedding_length/expert_feed_forward_length/expert_count)")
            return 2
        gu_t, d_t = tens[f"blk.{b}.ffn_gate_exps.weight"], tens[f"blk.{b}.ffn_down_exps.weight"]
        u_t = tens[f"blk.{b}.ffn_up_exps.weight"]
        # ---- the experts: the block's own blocks, gate/up rows then down rows, one expert after another
        with open(out / "experts.bin.tmp", "wb") as f:
            g_raw, u_raw, dn_raw = raw("ffn_gate_exps.weight"), raw("ffn_up_exps.weight"), raw("ffn_down_exps.weight")
            for e in range(NE):
                f.write(g_raw[e].tobytes())
                f.write(u_raw[e].tobytes())
                f.write(dn_raw[e].tobytes())
        (out / "experts.bin.tmp").replace(out / "experts.bin")
        (out / "experts.txt").write_text(
            f"{int(gu_t.tensor_type)} {int(u_t.tensor_type)} {int(d_t.tensor_type)} {H} {FF}\n", encoding="utf-8")
        eh = f32("nextn.eh_proj.weight")                     # (H, 2H): rows out, columns in
        emb, hid = (eh[:, :H], eh[:, H:]) if a.eh_order == "emb-first" else (eh[:, H:], eh[:, :H])
        dense = [
            ("fc_embedding.weight", "q8_0", emb), ("fc_hidden.weight", "q8_0", hid),
            ("pre_fc_norm_embedding.weight", "f32", f32("nextn.enorm.weight") + plus),
            ("pre_fc_norm_hidden.weight", "f32", f32("nextn.hnorm.weight") + plus),
            ("self_attn.q_proj.weight", "q8_0", f32("attn_q.weight")),
            ("self_attn.k_proj.weight", "q8_0", f32("attn_k.weight")),
            ("self_attn.v_proj.weight", "q8_0", f32("attn_v.weight")),
            ("self_attn.o_proj.weight", "q8_0", f32("attn_output.weight")),
            ("self_attn.q_norm.weight", "f32", f32("attn_q_norm.weight")),
            ("self_attn.k_norm.weight", "f32", f32("attn_k_norm.weight")),
            # the plain-residual architecture's own norms (layer.cpp's lm_head_mix uses output_norm there too)
            ("attn_norm.weight", "f32", f32("attn_norm.weight")),
            ("post_attention_norm.weight", "f32", f32("post_attention_norm.weight")),
            ("output_norm.weight", "f32", f32("nextn.shared_head_norm.weight")),
            ("mlp.gate.weight", "bf16", f32("ffn_gate_inp.weight")),
            ("mlp.shared_expert.gate_proj.weight", "q8_0", f32("ffn_gate_shexp.weight")),
            ("mlp.shared_expert.up_proj.weight", "q8_0", f32("ffn_up_shexp.weight")),
            ("mlp.shared_expert.down_proj.weight", "q8_0", f32("ffn_down_shexp.weight")),
            ("mlp.shared_expert_gate.weight", "bf16", f32("ffn_gate_inp_shexp.weight").reshape(1, -1)),
        ]
    else:
        H, FF, NE = H4, FF4, NE4
        # ---- the experts: per expert, so the whole block never sits in RAM as float32
        gate_t, up_t, down_t = (tens[f"blk.{b}.ffn_{k}_exps.weight"] for k in ("gate", "up", "down"))
        raw3 = [np.asarray(t.data) for t in (gate_t, up_t, down_t)]
        with open(out / "experts.bin.tmp", "wb") as f:
            for e in range(NE):
                g = dequantize(raw3[0][e], gate_t.tensor_type).reshape(FF, H)
                u = dequantize(raw3[1][e], up_t.tensor_type).reshape(FF, H)
                d = dequantize(raw3[2][e], down_t.tensor_type).reshape(H, FF)
                gu = q2_0(np.concatenate([g, u], axis=0)).reshape(2 * FF, H // 64 * 18)
                dn = q2_0(d).reshape(H, FF // 64 * 18)
                f.write(blob_of(gu, dn))
                if (e + 1) % 128 == 0:
                    print(f"  experts {e + 1}/{NE} ({time.time() - t0:.0f} s)", flush=True)
        (out / "experts.bin.tmp").replace(out / "experts.bin")
        eh = f32("nextn.eh_proj.weight")                     # (H, 2H): rows out, columns in
        emb, hid = (eh[:, :H], eh[:, H:]) if a.eh_order == "emb-first" else (eh[:, H:], eh[:, :H])
        dense = [
            ("fc_embedding.weight", "q8_0", emb), ("fc_hidden.weight", "q8_0", hid),
            ("pre_fc_norm_embedding.weight", "f32", f32("nextn.enorm.weight") + plus),
            ("pre_fc_norm_hidden.weight", "f32", f32("nextn.hnorm.weight") + plus),
            ("self_attn.q_proj.weight", "q8_0", f32("attn_q.weight")),
            ("self_attn.k_proj.weight", "q8_0", f32("attn_k.weight")),
            ("self_attn.v_proj.weight", "q8_0", f32("attn_v.weight")),
            ("self_attn.o_proj.weight", "q8_0", f32("attn_output.weight")),
            ("self_attn.q_norm.weight", "f32", f32("attn_q_norm.weight")),
            ("self_attn.k_norm.weight", "f32", f32("attn_k_norm.weight")),
            ("attn_hyper_connection.hc_norm.weight", "f32", f32("hc_attn_norm.weight")),
            ("attn_hyper_connection.input_mix_weight_down.weight", "bf16", f32("hc_attn_down.weight")),
            ("attn_hyper_connection.input_mix_weight_up.weight", "bf16", f32("hc_attn_up.weight")),
            ("attn_hyper_connection.block_inject_weight.weight", "bf16", f32("hc_attn_inject.weight")),
            ("mlp_hyper_connection.hc_norm.weight", "f32", f32("hc_ffn_norm.weight")),
            ("mlp_hyper_connection.input_mix_weight_down.weight", "bf16", f32("hc_ffn_down.weight")),
            ("mlp_hyper_connection.input_mix_weight_up.weight", "bf16", f32("hc_ffn_up.weight")),
            ("mlp_hyper_connection.block_inject_weight.weight", "bf16", f32("hc_ffn_inject.weight")),
            ("hyper_connection_mixer.hc_norm.weight", "f32", f32("nextn.hc_head_norm.weight")),
            ("hyper_connection_mixer.input_mix_weight_down.weight", "bf16", f32("nextn.hc_head_down.weight")),
            ("hyper_connection_mixer.input_mix_weight_up.weight", "bf16", f32("nextn.hc_head_up.weight")),
            ("mlp.gate.weight", "bf16", f32("ffn_gate_inp.weight")),
            ("mlp.shared_expert.gate_proj.weight", "q8_0", f32("ffn_gate_shexp.weight")),
            ("mlp.shared_expert.up_proj.weight", "q8_0", f32("ffn_up_shexp.weight")),
            ("mlp.shared_expert.down_proj.weight", "q8_0", f32("ffn_down_shexp.weight")),
            ("mlp.shared_expert_gate.weight", "bf16", f32("ffn_gate_inp_shexp.weight").reshape(1, -1)),
        ]
    lines, off = [], 0
    with open(out / "dense.bin", "wb") as f:
        for name, kind, x in dense:
            x = np.asarray(x, dtype=np.float32)
            if x.ndim == 1:
                rows, cols = 1, x.size
            else:
                rows, cols = x.shape[0], int(np.prod(x.shape[1:]))
            raw_b = q8_0(x.reshape(rows, cols)) if kind == "q8_0" else bf16(x) if kind == "bf16" else x.tobytes()
            pad = (-off) % 256
            f.write(b"\0" * pad)
            off += pad
            f.write(raw_b)
            lines.append(f"{name} {kind} {rows} {cols} {off} {len(raw_b)}")
            off += len(raw_b)
    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MTP draft layer ({arch}) from blk.{b} of {Path(a.gguf).name}: experts.bin "
          f"{(out / 'experts.bin').stat().st_size} B, dense.bin {off} B, {len(lines)} tensors -> {out} "
          f"({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
