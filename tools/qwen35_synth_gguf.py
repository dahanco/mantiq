"""tools/qwen35_synth_gguf.py - a tiny, structurally complete synthetic qwen35moe GGUF.

Phase 1 needs a model the reference forward pass and the
future engine kernels can run against without the 21.7 GB download: the same tensor set, the same
metadata keys, every structural path exercised (GDN layers AND full-attention layers, the 4:1
cadence, a GQA ratio > 1, a GatedDeltaNet with `n_v_heads != n_k_heads` so the k-head repeat runs,
a partial mRoPE narrower than the head so the pass-through tail runs, MoE with a shared expert, and
the MTP block), dimensions scaled down and values drawn from a seeded RNG (norm gains near 1,
everything else small; `ssm_a` negative, since the artifact stores -exp(A_log)).

The writer is deliberately minimal and self-contained (GGUF v3, F32 tensors only) so the synthetic
artifact has no dependency on conversion-time choices; the reference loader's dequantization path is
exercised through F32 and the repo's own reader validates the result.

Orientation (the one invariant to keep in mind when adding tensors here): a GGUF tensor's `ne[]`
array is what `create_tensor(..., {a, b, c})` in llama.cpp spells out, and the file stores those
dims in reverse. The reference loader reverses them back, so **a numpy array of shape S passed to
`write_gguf` comes back out of `tensor()` with shape S**. llama.cpp's own `ne[]` for a weight is
`(in, out)`, so the numpy shape is `(out, in)` - i.e. pass weights as `(out, in)`.

Usage:
    python tools/qwen35_synth_gguf.py --out qwen35-synth.gguf [--seed 1234] [--wide]

The tokenizer is deliberately absent from the default fixture: the oracle runs on token ids, and
the real tokenizer is a conversion concern (tools/strata_tokenizer.py), not a reference-model one.
`--vocab` adds the keys upstream llama.cpp needs to LOAD the model (Phase 2 parity).
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# GGUF metadata value types (ggml spec order)
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
GGML_F32 = 0

ALIGN = 32


def _s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_u32(v): return struct.pack("<I", U32) + struct.pack("<I", v)
def kv_i32(v): return struct.pack("<I", I32) + struct.pack("<i", v)
def kv_f32(v): return struct.pack("<I", F32) + struct.pack("<f", v)
def kv_bool(v): return struct.pack("<I", BOOL) + struct.pack("<B", 1 if v else 0)
def kv_str(v): return struct.pack("<I", STRING) + _s(v)


# array elements carry NO per-element type tag: the array header names the element type once
def kv_arr(elem_tag: int, elems: list[bytes]) -> bytes:
    return struct.pack("<I", ARRAY) + struct.pack("<I", elem_tag) + struct.pack("<Q", len(elems)) + b"".join(elems)


def arr_i32(vs): return kv_arr(I32, [struct.pack("<i", v) for v in vs])
def arr_f32(vs): return kv_arr(F32, [struct.pack("<f", v) for v in vs])
def arr_str(vs): return kv_arr(STRING, [_s(v) for v in vs])


def write_gguf(path: Path, metadata: list[tuple[str, bytes]], tensors: list[tuple[str, np.ndarray]]) -> None:
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    kvs = b"".join(_s(k) + v for k, v in metadata)
    infos, blobs, offset = b"", [], 0
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr, dtype="<f4")
        dims = list(reversed(arr.shape))                 # ne0 varies fastest
        infos += _s(name) + struct.pack("<I", len(dims))
        infos += b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<I", GGML_F32) + struct.pack("<Q", offset)
        blob = arr.tobytes()
        blobs.append(blob)
        offset += len(blob)
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN   # tensors start aligned, like gguf.cpp
    data_pad = (len(header) + len(kvs) + len(infos) + ALIGN - 1) // ALIGN * ALIGN
    pad = b"\0" * (data_pad - len(header) - len(kvs) - len(infos))
    body = b"".join(b.ljust(len(b) + (-len(b) % ALIGN), b"\0") for b in blobs)
    path.write_bytes(header + kvs + infos + pad + body)


# The seed-1234 fixture geometry: every structural path of qwen35moe, none of the size.
#   * 8 trunk layers + 1 MTP block (block_count 9), full_attention_interval 4 -> full attention at
#     il 3 and 7, GDN at 0,1,2,4,5,6, and the MTP block at 8 is a full-attention block too;
#   * n_head 4 / n_head_kv 2 (GQA ratio 2) and n_v_heads 8 / n_k_heads 4 (repeat ratio 2) - the two
#     ratios are the SAME number here on purpose, so a kernel that confuses the two mappings still
#     passes this fixture... `WIDE` below breaks that tie (GQA 2 vs repeat 3) and is the second half
#     of what the test actually runs;
#   * head_dim 64 with n_rot 32: exactly half the head is rotated, so both the rotation and the
#     copy-through tail are exercised;
#   * n_embd 128 vs n_embd/n_head 32 - head_dim CANNOT be derived from n_embd/n_head, so a loader
#     that forgets attention.key_length reads the wrong width immediately.
SMALL = dict(n_embd=128, n_vocab=512, n_layers=8, n_nextn=1, n_head=4, n_head_kv=2,
             head_dim=64, n_rot=32, sections=[11, 11, 10, 0], freq_base=1e7,
             d_conv=4, head_k_dim=32, n_k_heads=4, n_v_heads=8,
             n_expert=8, n_exp_used=3, n_ff_exp=48, n_ff_shexp=48, interval=4)

# The same tiny model with the two head ratios DECOUPLED (GQA ratio 2 = 8/4, k-head repeat ratio
# 3 = 6/2) and n_rot == head_dim, so the whole head rotates and the pass-through tail is empty.
WIDE = dict(SMALL, n_head=8, n_head_kv=4, n_k_heads=2, n_v_heads=6, n_rot=64)


def build(seed: int, vocab: bool = False) -> tuple[list[tuple[str, bytes]], list[tuple[str, np.ndarray]]]:
    return _build(SMALL, seed, vocab)


def build_wide(seed: int = 777, vocab: bool = False) -> tuple[list[tuple[str, bytes]], list[tuple[str, np.ndarray]]]:
    return _build(WIDE, seed, vocab)


def _build(g: dict, seed: int, vocab: bool) -> tuple[list[tuple[str, bytes]], list[tuple[str, np.ndarray]]]:
    n_embd, n_vocab, n_layers, n_nextn = g["n_embd"], g["n_vocab"], g["n_layers"], g["n_nextn"]
    n_head, n_head_kv, head_dim = g["n_head"], g["n_head_kv"], g["head_dim"]
    n_rot, sections, freq_base = g["n_rot"], g["sections"], g["freq_base"]
    d_conv, head_k_dim = g["d_conv"], g["head_k_dim"]
    n_k_heads, n_v_heads = g["n_k_heads"], g["n_v_heads"]
    n_expert, n_exp_used = g["n_expert"], g["n_exp_used"]
    n_ff_exp, n_ff_shexp, interval = g["n_ff_exp"], g["n_ff_shexp"], g["interval"]
    n_layer_all = n_layers + n_nextn

    # load_arch_tensors' own derivations
    head_v_dim = head_k_dim                       # head_v_dim == head_k_dim == ssm_d_state
    key_dim = head_k_dim * n_k_heads
    value_dim = head_v_dim * n_v_heads
    conv_dim = key_dim * 2 + value_dim
    ssm_d_inner = head_v_dim * n_v_heads
    q_width = head_dim * 2 * n_head               # [q | gate]
    assert n_v_heads % n_k_heads == 0, "build_layer_attn_linear asserts H_v % H_k == 0"
    assert n_head % n_head_kv == 0, "GQA needs n_head % n_head_kv == 0"
    assert sum(sections) <= head_dim, "ggml_mrope_cache_init asserts sect_dims <= ne0"
    assert n_rot <= head_dim and n_rot % 2 == 0, "ggml_rope asserts n_dims <= ne0 and even"

    is_recr = [i < n_layers and (i + 1) % interval != 0 for i in range(n_layer_all)]

    rng = np.random.default_rng(seed)

    def w(*shape, scale=0.05):                            # a plain weight
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm_w(n):                                        # an RMS gain, near 1
        return (1.0 + rng.standard_normal(n) * 0.02).astype(np.float32)

    md = [
        ("general.architecture", kv_str("qwen35moe")),
        ("general.name", kv_str("qwen35moe-synth")),
        ("general.alignment", kv_u32(ALIGN)),
        ("qwen35moe.block_count", kv_u32(n_layer_all)),
        ("qwen35moe.embedding_length", kv_u32(n_embd)),
        ("qwen35moe.nextn_predict_layers", kv_u32(n_nextn)),
        ("qwen35moe.attention.head_count", kv_u32(n_head)),
        ("qwen35moe.attention.head_count_kv", kv_u32(n_head_kv)),
        # attention.key_length drives n_embd_head_k AND the K-cache row width; it is NOT n_embd/n_head
        ("qwen35moe.attention.key_length", kv_u32(head_dim)),
        ("qwen35moe.attention.value_length", kv_u32(head_dim)),
        ("qwen35moe.attention.layer_norm_rms_epsilon", kv_f32(1e-5)),
        ("qwen35moe.rope.dimension_count", kv_u32(n_rot)),
        ("qwen35moe.rope.dimension_sections", arr_i32(sections)),
        ("qwen35moe.rope.freq_base", kv_f32(freq_base)),
        # gated delta net
        ("qwen35moe.ssm.conv_kernel", kv_u32(d_conv)),
        ("qwen35moe.ssm.inner_size", kv_u32(ssm_d_inner)),
        ("qwen35moe.ssm.state_size", kv_u32(head_k_dim)),
        ("qwen35moe.ssm.time_step_rank", kv_u32(n_v_heads)),
        ("qwen35moe.ssm.group_count", kv_u32(n_k_heads)),
        # MoE - every layer, there is no dense lead
        ("qwen35moe.expert_count", kv_u32(n_expert)),
        ("qwen35moe.expert_used_count", kv_u32(n_exp_used)),
        ("qwen35moe.expert_feed_forward_length", kv_u32(n_ff_exp)),
        ("qwen35moe.expert_shared_feed_forward_length", kv_u32(n_ff_shexp)),
        ("qwen35moe.full_attention_interval", kv_u32(interval)),
    ]
    if vocab:
        md += [
            ("qwen35moe.context_length", kv_u32(4096)),
            ("tokenizer.ggml.model", kv_str("gpt2")),
            ("tokenizer.ggml.pre", kv_str("qwen35")),
            ("tokenizer.ggml.tokens", arr_str(["tok%d" % i for i in range(n_vocab)])),
            ("tokenizer.ggml.scores", arr_f32([0.0] * n_vocab)),
            ("tokenizer.ggml.token_type", arr_i32([1] * n_vocab)),
            ("tokenizer.ggml.merges", arr_str([])),
        ]

    ts: list[tuple[str, np.ndarray]] = [
        ("token_embd.weight", w(n_vocab, n_embd)),
        ("output_norm.weight", norm_w(n_embd)),
        ("output.weight", w(n_vocab, n_embd)),
    ]
    for il in range(n_layer_all):
        p = f"blk.{il}."
        ts += [
            (p + "attn_norm.weight", norm_w(n_embd)),
            (p + "post_attention_norm.weight", norm_w(n_embd)),
        ]
        if not is_recr[il]:                                # full attention (trunk and the MTP block)
            ts += [
                (p + "attn_q.weight", w(q_width, n_embd)),         # [q | gate] fused
                (p + "attn_k.weight", w(head_dim * n_head_kv, n_embd)),
                (p + "attn_v.weight", w(head_dim * n_head_kv, n_embd)),
                (p + "attn_q_norm.weight", norm_w(head_dim)),
                (p + "attn_k_norm.weight", norm_w(head_dim)),
                (p + "attn_output.weight", w(n_embd, head_dim * n_head)),
            ]
        else:                                              # GatedDeltaNet
            ts += [
                (p + "attn_qkv.weight", w(conv_dim, n_embd)),      # [q | k | v]
                (p + "attn_gate.weight", w(value_dim, n_embd)),    # z
                (p + "ssm_conv1d.weight", w(conv_dim, d_conv)),
                (p + "ssm_dt.bias", (rng.standard_normal(n_v_heads).astype(np.float32) * 0.1)),
                # ssm_a holds -exp(A_log): negative by contract
                (p + "ssm_a", (-np.exp(rng.uniform(0.5, 1.5, n_v_heads))).astype(np.float32)),
                (p + "ssm_beta.weight", w(n_v_heads, n_embd)),
                (p + "ssm_alpha.weight", w(n_v_heads, n_embd)),
                (p + "ssm_norm.weight", norm_w(head_v_dim)),
                (p + "ssm_out.weight", w(n_embd, value_dim)),
            ]
        ts += [
            (p + "ffn_gate_inp.weight", w(n_expert, n_embd)),
            (p + "ffn_gate_exps.weight", w(n_expert, n_ff_exp, n_embd)),
            (p + "ffn_up_exps.weight", w(n_expert, n_ff_exp, n_embd)),
            (p + "ffn_down_exps.weight", w(n_expert, n_embd, n_ff_exp)),
            (p + "ffn_gate_inp_shexp.weight", (rng.standard_normal(n_embd).astype(np.float32) * 0.05)),
            (p + "ffn_gate_shexp.weight", w(n_ff_shexp, n_embd)),
            (p + "ffn_up_shexp.weight", w(n_ff_shexp, n_embd)),
            (p + "ffn_down_shexp.weight", w(n_embd, n_ff_shexp)),
        ]
        if il >= n_layers:                                 # the MTP block's own tensors
            ts += [
                (p + "nextn.eh_proj.weight", w(n_embd, 2 * n_embd)),
                (p + "nextn.enorm.weight", norm_w(n_embd)),
                (p + "nextn.hnorm.weight", norm_w(n_embd)),
                (p + "nextn.shared_head_norm.weight", norm_w(n_embd)),
            ]
    return md, ts


def selftest(path: Path, wide: bool = False) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gguf_reader import GGUFFile
    gf = GGUFFile(path)
    by = {t.name: t for t in gf.tensors}
    md = gf.metadata
    g = WIDE if wide else SMALL
    n_layers, n_nextn = g["n_layers"], g["n_nextn"]
    head_dim, n_head, n_head_kv = g["head_dim"], g["n_head"], g["n_head_kv"]
    n_k_heads, n_v_heads, head_k_dim, d_conv = g["n_k_heads"], g["n_v_heads"], g["head_k_dim"], g["d_conv"]
    conv_dim = head_k_dim * n_k_heads * 2 + head_k_dim * n_v_heads

    assert md["general.architecture"] == "qwen35moe", "architecture key lost"
    assert md["qwen35moe.rope.dimension_sections"] == g["sections"], "sections array mangled"
    assert md["qwen35moe.nextn_predict_layers"] == n_nextn, "nextn lost"
    assert list(by["blk.0.attn_qkv.weight"].shape) == [g["n_embd"], conv_dim], "attn_qkv (in, out)"
    assert list(by["blk.0.ssm_conv1d.weight"].shape) == [d_conv, conv_dim], "ssm_conv1d (d_conv, conv_dim)"
    assert list(by["blk.3.attn_q.weight"].shape) == [g["n_embd"], head_dim * 2 * n_head], "fused attn_q"
    assert list(by["blk.3.attn_output.weight"].shape) == [head_dim * n_head, g["n_embd"]], "attn_output"
    assert list(by["blk.0.ffn_gate_exps.weight"].shape) == [g["n_embd"], g["n_ff_exp"], g["n_expert"]], "experts"
    assert list(by["blk.0.ffn_down_exps.weight"].shape) == [g["n_ff_exp"], g["n_embd"], g["n_expert"]], "down experts"
    assert "blk.0.attn_qkv.weight" in by and "blk.3.attn_q.weight" in by, "layer cadence wrong"
    assert "blk.0.attn_q.weight" not in by and "blk.3.attn_qkv.weight" not in by, "cadence inverted"
    mtp = f"blk.{n_layers}."
    assert mtp + "attn_q.weight" in by, "MTP block must be full attention"
    assert mtp + "nextn.eh_proj.weight" in by and mtp + "nextn.shared_head_norm.weight" in by, "MTP nextn missing"
    assert f"blk.{n_layers - 1}.nextn.eh_proj.weight" not in by, "nextn leaked into the trunk"
    print(f"selftest ok: {len(gf.tensors)} tensors, {path.stat().st_size / 1e6:.2f} MB"
          f"{' (wide)' if wide else ''}")


def main() -> int:
    ap = argparse.ArgumentParser(description="write a tiny synthetic qwen35moe GGUF")
    ap.add_argument("--out", default="qwen35-synth.gguf")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vocab", action="store_true",
                    help="add the tokenizer metadata upstream llama.cpp needs to LOAD the model "
                         "(byte-different from the default; the fixture stays vocab-less)")
    ap.add_argument("--wide", action="store_true",
                    help="the WIDE geometry (decoupled GQA / k-head-repeat ratios, full-head RoPE)")
    args = ap.parse_args()
    md, ts = build_wide(args.seed, vocab=args.vocab) if args.wide else build(args.seed, vocab=args.vocab)
    out = Path(args.out)
    write_gguf(out, md, ts)
    selftest(out, wide=args.wide)
    return 0


if __name__ == "__main__":
    sys.exit(main())
