# Upstream Strata ports (Mantiq 1.7.5 work tree)

Two opt-in ports from upstream Strata `3142b66b` are present in this work tree. Both are off by default;
the released `v1.7.5` binaries do not contain them. The local A/B build is `build-ab`
(CUDA 13.4, `sm_75;80;86;89`, `STRATA_NATIVE_EXPERTS=ON`, `STRATA_PORTABLE=ON`).

## PLE compact read-ahead — `MNTQ_PLE_COMPACT`, `MNTQ_PLE_COPY_STREAM`

* `PleTable::gather_raw_iq4nl()` copies a chunk's 16 rows/token as raw IQ4_NL bytes (90 B/row) instead of
  dequantizing them on the host into 10,240 floats/token. Out-of-table rows are encoded as `d=0`, code 8 so
  the device dequant produces `+0.0f`, exactly matching the host `gather_batch()` path.
* `ple_iq4nl_dequant()` runs the upstream one-thread-per-value kernel on the compute stream.
* `MNTQ_PLE_COPY_STREAM=1` uploads those raw bytes on a non-blocking side stream. `ple_raw_free` prevents the
  single raw device staging buffer being overwritten before the previous chunk's dequant; `ple_copied` keeps
  the host-side double-buffer reuse guard correct, and `PleCopyDrain` drains the side stream on every exit.
* Requires an in-memory PLE table (`--ple-io mmap`, the default for a PLE-enabled qwen4exp model). Direct SSD
  mode has no mapping and automatically falls back to `gather_batch()`.
* Status: compiles and links (`iq_kernels.cu`/`ple.cu` syntax checks and the full CUDA build are clean); the
  flag strings and kernel are present in `build-ab\Release\mntq.exe`. **No local A/B measurement**: the active
  Qwopus/Qwen35Moe model has no `blk.1.ple_*` module (`PLE off`), and no qwen4exp PLE-table artifact is present
  on this machine. A PLE A/B needs a PLE-enabled model plus `--ple-gguf <table> --ple-io mmap`.

## EXPDEC shared-memory staging — `MNTQ_EXPDEC`, `MNTQ_EXPDEC_NC`

* `expdec_gu_kernel` / `expdec_gu_role_kernel` stage `NC` entries' q8_1 activations into dynamic shared memory,
  then apply each decoded weight block to all `NC` entries. `MNTQ_EXPDEC_NC` selects 1..8 (default 4; capped by
  a 40 KB shared-memory budget). The down projection stays on the default path.
* The k/lane order and `warp_sum` tree are unchanged from `native_gu_kernel`, so each column's sum is bitwise
  identical. A/B runs compare the `output :` line hash; all arms matched.
* Status: compiles and links; verified with B-A-B runs via `C:\tmp3\ab_flags.ps1`.

### Measured A/B (RTX 3070, Qwopus 35B A3B, `_p1024.txt`, 64 generated tokens)

| `MNTQ_EXPDEC_NC` | baseline tok/s | EXPDEC tok/s | change | output hash |
|---:|---:|---:|---:|---|
| 2 | 65.18 | 64.00 | -1.8% | identical |
| 4 | 63.78 | 64.75 | +1.5% | identical |
| 4 | 65.38 | 63.94 | -2.2% | identical |
| 4 | 64.63 | 65.12 | +0.8% | identical |
| 4 | 62.63 | 65.47 | +4.5% | identical |
| 8 | 66.40 | 64.48 | -2.9% | identical |

Conclusion: the port is output-identical but **performance-neutral within noise** on this model/GPU; the
short repetitive `tokens_2048.txt` prompt hits EOS after three tokens and produces misleadingly large swings,
so it is not used for the conclusion. Keep `MNTQ_EXPDEC` opt-in.

Raw logs: `C:\tmp3\_mntq_bench\expdec_ab_p1024_*.txt`, `expdec_ab_final_confirm.txt`.
