## Mantiq 1.7.8 — the portable builds

Two self-contained builds. Extract, then run `START-MNTQ.bat`.

Both engines are statically linked against the MSVC runtime (`/MT`), so there is no Visual C++ redistributable to install. The CUDA package ships cuBLAS beside the engine; the NVIDIA driver supplies `nvcuda.dll`. Each `.zip` has a `.sha256` sidecar; verify with `Get-FileHash -Algorithm SHA256` or `sha256sum -c`.

| Build | Engine | Runs on |
| --- | --- | --- |
| `Mantiq-Portable-CPU.zip` | `build-cpu/mntq-cpu.exe` | any x86-64 PC with AVX2 (BMI2 disabled), static CRT |
| `Mantiq-Portable-CUDA.zip` | `build/mntq.exe` | NVIDIA, compute capability 7.5+ |

## What changed since 1.7.7

- Adaptive expert promotion is now tunable: `--adapt-gain F` and `--adapt-policy freq|lfru|nextuse`.
- `--expert-mirror` pins VRAM-resident experts in RAM when the RAM tier comes from the model file, so an adaptive swap never falls back to the SSD.
- Swaps now update the cache's own residency table; learned profiles save the live residency.
- Optional `MNTQ_ADAPT_DEBUG=1` prints candidate/victim counts.

Quality remains unchanged: greedy and sampled A/B tests produced identical output hashes. Decode gate: 100.83 tok/s mean (100.01-101.35), hash `02039B7067E348731EC4A6B9A29B10D0CC3EBC725693DC80A93DE29AA1D9BB49`, chunk 4096, arena 1936 MiB.

| File | SHA-256 |
| --- | --- |
| `Mantiq-Portable-CPU.zip` | `F2CBD739D678E983BC8641CAFDFE6E52C58F1BCE96C92F1BCE72D2F434769FF` |
| `Mantiq-Portable-CUDA.zip` | `FA4AEB311C6E725244FF4E5790584F4C91A7DC93792159A94594270AFB3F4E8D` |
