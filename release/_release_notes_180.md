## Mantiq 1.8.0 — trace-built expert residency

This release changes **pack data and the setup tooling, not the C++/CUDA engine**.
The shipped expert profile for `Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M` was stale: its
first 859 VRAM slots covered only 7.5-10.8% of the experts actually routed during
decode. Profiles are now ranked from each model's own routing trace, and setup
does that for every qwen35moe pack rather than for one hand-tuned model.

### What changed

- `models_cfg/packs/Qwopus3.6 35B A3B Coder MTP Q4_K_M/expert-profile.bin`
  is the new equal-weighted profile built from two `--dump-routing` traces
  (`tools/make_profile.py ... --reorder --equal --n-expert 256 --n-layer 40`).
- The same pack's `routing-trace.bin` is replaced with the equal-weighted
  concatenation, so a fresh setup regenerates 854 of the same 859 top slots even
  if the profile file is absent.
- Originals are kept beside them as `*.pre-combined`.
- `mntq.py` gains `--refresh-profiles`: it discovers every configured qwen35moe
  pack, reuses its routing trace (or captures one with a short synthetic-token
  decode when it has none), rebuilds the profile with `make_profile.py --reorder`
  and writes both files back. Setup calls it for each newly written qwen config,
  so a fresh install builds its own VRAM residency table. Set
  `MNTQ_SKIP_PROFILE_CAPTURE=1` to opt out. All five configured packs were
  refreshed; the other four already had traces matching their profiles, while
  Apex gained its first real profile (11.6% -> 18.2% VRAM hits, 75.0 -> 80.9 tok/s).
- `VERSION` is 1.8.0 in the main tree and both portable packages; the
  `MNTQ-BUILD.json` version fields are synced. No engine source changed, so the
  build-stamp source hashes are unchanged.

### Measured result (RTX 3070 8 GB, Q4_K_M, 256-token greedy decode)

| Workload | Before | After | Change | VRAM expert hits |
|---|---:|---:|---:|---:|
| `tokens_lookup_code.txt` | ~104.8 tok/s | ~116.6 tok/s | **+11.3%** | 10.8% → 36.7% |
| `tokens_5312.txt` | ~62.9 tok/s | ~75.5 tok/s | **+20.0%** | 6.4% → 56.4% |
| Final gate, shipped config | 87.5 tok/s (earlier session) | **116.39 tok/s** (114.84-118.19) | — | — |

Output hash `02039B7067E348731EC4A6B9A29B10D0CC3EBC725693DC80A93DE29AA1D9BB49`
was identical in every arm and in all three final gate runs. Arena 1936 MiB and
chunk 4096 were unchanged. The paired A/B numbers are the reliable gain
estimate; the absolute comparison to the earlier 87.5 session includes machine
drift. A code-only profile was also tested (+23% on code, +3.7% out of sample)
and rejected in favour of the equal-weighted one.

### Caveats

- The profile is trained on two prompt shapes. `--expert-profile-save` will keep
  learning at runtime; for a different workload, recapture with `--dump-routing`
  and rebuild with `tools/make_profile.py` (see the release notes in the repo).
- The general refresh learns from existing routing traces; a model whose trace is
  stale should be recaptured with `mntq.py --refresh-profiles` after a workload change.
- Both portable binaries were rebuilt from this tree and embed 1.8.0; the
  verifier confirmed `has1.8.0=True` for `mntq-cpu.exe` and `mntq.exe`.

| File | SHA-256 |
| --- | --- |
| `Mantiq-Portable-CPU.zip` | `CC282E5421AC16F8883A7DD46DF4291130D08D74DD0920DE58DACB03928DD717` |
| `Mantiq-Portable-CUDA.zip` | `3D29F64AD98FE9F1C8E4FE49BEF14E544F52A96D270C09ED4719F9E97B207D71` |
