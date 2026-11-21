# Upstream Strata ports (Mantiq 1.7.6 work tree)

Three ports from upstream Strata `3142b66b` are present in this work tree. Guarded suffix-lookup windows are
promoted to the default on one-GPU CUDA; PLE compact/copy-stream and EXPDEC remain opt-in and off by default.
The released `v1.7.6` binaries contain the promoted lookup window and the EXPDEC/PLE code behind their flags.
The promoted CUDA build is `build-cuda`
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
  then apply each decoded weight block to all `NC` entries. `MNTQ_EXPDEC_NC` accepts 1..8; only 1, 2, 4 and 8 have
  kernels, so other values snap down to the next supported width (3->2, 5/6/7->4; default 4). The down projection
  stays on the default path.
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

## Guarded suffix-lookup window — promoted (`--lookup-max-t`, `MNTQ_LOOKUP_MAX_T`)

* Ported upstream Strata's long suffix-lookup verify window: a repeated n-gram in the context can fill a window
  wider than the MTP's, so one verification commits many more tokens on rewrites/agent edit turns. The MTP still
  loads only `--spec` rows (its MMVQ graphs support a small set of widths); `draft_after_commit()` splits the
  follow-up draft so the MTP never proposes 11 rows nobody filled.
* `policy.choose()` only takes a window wider than `max(t_mtp + 2, 4)` after that match-length bucket has shown a
  real continuation (`seen >= kProbes`, `q >= 0.95`). A coincidental match in ordinary prose cannot unlock it.
* Promotion: with no `--lookup-max-t` and no `MNTQ_LOOKUP_MAX_T`, the engine enables `12` only when one CUDA GPU
  is visible, there is no layer split, `--suffix-draft > 0`, `--spec >= 2` and `--mtp-max-t > 0`. Everything else
  keeps the old off-by-default behaviour. `--lookup-max-t 0` / `MNTQ_LOOKUP_MAX_T=0` still disables it. The
  dashboard configs and launchers also set `MNTQ_LOOKUP_MAX_T=12` explicitly.
* No prompt-read win was proven: prefill deltas in the A/B runs were noisy (`+2.9% .. +9.6%` on the edit prompt),
  so this promotion is for decode speed only.

### Final guarded A/B (RTX 3070, Qwopus 35B A3B Coder, `--spec 4 --suffix-draft 4 --mtp-max-t 2`, 128 new tokens)

| Prompt | rep | baseline decode | lookup decode | change | prefill change | output |
|---|---:|---:|---:|---:|---:|---|
| code rewrite | 1 | 74.42 | 86.42 | +16.1% | +2.9% | identical |
| code rewrite | 2 | 74.93 | 87.92 | +17.3% | +4.6% | identical |
| code rewrite | 3 | 73.77 | 87.96 | +19.2% | +9.6% | identical |
| normal control | 1 | 64.79 | 67.45 | +4.1% | +14.7% | pre-existing nondeterminism |
| normal control | 2 | 66.46 | 64.45 | -3.0% | -4.3% | identical |

The code-edit output hash was `4496D96897A63679` in every arm. The control has two hashes that also occur in
baseline arms (`A4A431D54B4238DD`, `3CC1D0E806B35BE9`), so a single mismatch there is not attributable to lookup.
The control took only 5 lookup windows with narrow `7/8` drafts; its decode rate is neutral within noise.

Baseline: `C:\tmp3\baseline-v175-expdec-mntq.exe`, SHA-256 `EBDE9940A3315B74F1073AAD6DE450088AF80BEEB88F8FC9BC4C27476C2B60AD`.
Tested guarded variant: SHA-256 `7403DF95930CA329A74EDF96CF476A103C1F4F7823B5A0C35BFE47F0B9035ABF`
(historical A/B artifact; not retained in the tree).
Promoted binary: `C:\tmp3\project-mantiq\build-cuda\Release\mntq.exe`, SHA-256
`81939C2CBF4B0AD83D9BB4A040ED1C506DD189FAC60BD795994866E5D1D6BA27`.

Raw logs: `C:\tmp3\_mntq_bench\lookup_ab_code3.log`, `lookup_ab_control2.log`; harness `C:\tmp3\lookup_ab.ps1`.
