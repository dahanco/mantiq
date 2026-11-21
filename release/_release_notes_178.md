## Mantiq 1.7.8 — local sync build

Version bump only; no GitHub release is published yet.

Two self-contained builds. Extract, then run `START-MNTQ.bat`.

| Build | Engine | Runs on |
| --- | --- | --- |
| `Mantiq-Portable-CPU.zip` | `build-cpu/mntq-cpu.exe` | any x86-64 PC with AVX2 (BMI2 disabled), static CRT |
| `Mantiq-Portable-CUDA.zip` | `build/mntq.exe` | NVIDIA, compute capability 7.5+ |

## What changed since 1.7.7

- Adaptive expert promotion is now tunable: `--adapt-gain F` and `--adapt-policy freq|lfru|nextuse`.
- `--expert-mirror` pins VRAM-resident experts in RAM when the RAM tier comes from the model file, so an adaptive swap never falls back to the SSD.
- Swaps now update the cache's own residency table; learned profiles save the live residency.
- Optional `MNTQ_ADAPT_DEBUG=1` prints candidate/victim counts.

Quality remains unchanged: greedy and sampled A/B tests produced identical output hashes.

| File | SHA-256 |
| --- | --- |
| `Mantiq-Portable-CPU.zip` | `F2CBD739D678E983BC8641CAFDFE6E52C58F1BCE96C92F1BCE72D2F434769FF` |
| `Mantiq-Portable-CUDA.zip` | `FA4AEB311C6E725244FF4E5790584F4C91A7DC93792159A94594270AFB3F4E8D` |
