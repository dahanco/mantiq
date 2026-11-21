## Mantiq 1.7.7 — the portable builds

Two self-contained builds. Extract, then run `START-MNTQ.bat`.

| Build | Engine | Runs on |
| --- | --- | --- |
| `Mantiq-Portable-CPU.zip` | `build-cpu/mntq-cpu.exe` | any x86-64 PC with AVX2 (BMI2 is disabled), static CRT — no GPU |
| `Mantiq-Portable-CUDA.zip` | `build/mntq.exe` | NVIDIA, compute capability 7.5+ (arch 75/80/86/89), driver with CUDA 13 support |

Both engines are statically linked against the MSVC runtime (`/MT`), so there is no Visual C++ redistributable to install. The CUDA package ships cuBLAS beside the engine; the NVIDIA driver supplies `nvcuda.dll`. The CUDA 13 runtime requires a recent driver; check NVIDIA's CUDA compatibility table if in doubt.

Each `.zip` ships with a `.sha256` file beside it; verify with `Get-FileHash -Algorithm SHA256` (PowerShell) or `sha256sum -c` (Linux).

| File | SHA-256 |
| --- | --- |
| `Mantiq-Portable-CPU.zip` | `9F5ADD800DE24F122754CD920DAC71AC89AF73190BFE08FE5FDFEDC91CB685ED` |
| `Mantiq-Portable-CUDA.zip` | `589AFFEA2F3A39FA3FB934B400E9CC795B2EE5D50E25E82DC6F4454998E7C66B` |

## What changed since 1.7.6

This is the audited release. The 1.7.6 feature set is unchanged; the fixes below come from a full project audit.

- **MTP draft widths are corrected.** `draft_after_commit` now drafts at the real verified window instead of a fixed `S_mtp`, and `S_mtp` is clamped to the MTP's loaded width. A request can no longer read stale residual rows or abort when `--mtp-max-t > --spec`.
- **Prompt progress is per-request.** `progress` and `prefill_tok_s_mean` are thread-local in multi mode, so concurrent requests cannot show each other's `prompt_progress`.
- **Restart and logging races are fixed.** A stale stdout pump can no longer mark a restarted engine dead; engine stderr and server stdout now share one log lock, so torn log lines are gone.
- **Prompt-cache defaults now ship.** Fresh installs get `--prompt-cache 32 --prompt-cache-every 2048` instead of the previous `6 / 16384`, matching the documented agent-session behavior.
- **Portable CPU is truly AVX2-only.** BMI2 is disabled for the portable build because ggml has no runtime dispatch for it.
- **Robustness fixes.** `keep_alive="-garbage"` is rejected; `lookup_size_mask` cannot shift by 32 or more; `checkpoint_restore` restores `ple_token`; EXPDEC `NC` values 3/5/6/7 snap to supported widths; PLE copy-stream failure nulls its event handle; the final native PP line no longer overstates fresh tokens by one.
- **Release tooling is versioned.** Build/pack/verify scripts and notes now live in the repository under `release/`, with a checklist. Stale `C:\tmp` paths, docs claims, zip verification, and the base decode CSV hash were corrected.

Decode remains inside the baseline gate on the shipping binary: 97.90 tok/s mean (93.66-100.85) versus the 94.84-98.65 baseline range, with output hash `02039B7067E348731EC4A6B9A29B10D0CC3EBC725693DC80A93DE29AA1D9BB49`. The 5-turn cache test reuses the whole prefix after the first cold read.
