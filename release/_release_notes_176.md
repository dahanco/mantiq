## Mantiq 1.7.6 — the portable builds

Two self-contained builds. Extract, then run `START-MNTQ.bat`.

| Build | Engine | Runs on |
| --- | --- | --- |
| `Mantiq-Portable-CPU.zip` | `build-cpu/mntq-cpu.exe` | any x86-64 PC with AVX2 (BMI2 is disabled), static CRT — no GPU |
| `Mantiq-Portable-CUDA.zip` | `build/mntq.exe` | NVIDIA, compute capability 7.5+ (arch 75/80/86/89), driver with CUDA 13 support |

Both engines are statically linked against the MSVC runtime (`/MT`), so there is no Visual C++ redistributable to install. The CUDA package ships cuBLAS beside the engine; the NVIDIA driver supplies `nvcuda.dll`. The CUDA 13 runtime requires a recent driver; check NVIDIA's CUDA compatibility table if in doubt.

Each `.zip` ships with a `.sha256` file beside it; verify with `Get-FileHash -Algorithm SHA256` (PowerShell) or `sha256sum -c` (Linux).

| File | SHA-256 |
| --- | --- |
| `Mantiq-Portable-CPU.zip` | `013672823A2DC73B6EE9BF199B9E61EB322497E2BBCB77F5A8A00B865BE46182` |
| `Mantiq-Portable-CUDA.zip` | `92AF430C95851FF4BF8C28BA55F08FA4A427D2CA81FAF20235A9AE2D03685BAB` |

## What changed since 1.7.5

- **Guarded suffix-lookup windows are promoted on one-GPU CUDA.** A repeated n-gram can open a verify window up to 12 tokens; the MTP stays narrow and a follow-up draft replays the accepted cells. A match bucket only unlocks a wide window after it proves a real continuation (`q >= 0.95`). Code-rewrite decode measured +16-19% with identical output hashes; ordinary prose stayed neutral.
- **EXPDEC shared-memory staging is ported** behind `MNTQ_EXPDEC` / `MNTQ_EXPDEC_NC` (output-identical, performance-neutral within noise here). PLE compact read-ahead is also ported behind flags but stays compile-tested only because this model has no PLE module.
- **Agent prompt reuse is tuned.** The pre-answer/generation checkpoint is always kept, and the native serve chain is tuned to `--prompt-cache 32 --prompt-cache-every 2048`. Five-turn opencode/API tests confirm that after the first cold read each turn reuses the whole prefix and reads only its new suffix.
- **Native qwen35moe prompt progress is reported correctly.** `[mntq] reading the prompt:` now shows fresh tokens freshly read, the fresh total, and the reused cached prefix instead of the whole re-sent prompt. The streaming `prompt_progress` fields (`total`, `cache`, `processed`, `time_ms`) match llama.cpp.
- Decode remains inside the baseline gate on the shipping binary: 98.44 tok/s mean (98.22-98.78) versus the 94.84-98.65 baseline range, with output hash `02039B7067E348731EC4A6B9A29B10D0CC3EBC725693DC80A93DE29AA1D9BB49`.
