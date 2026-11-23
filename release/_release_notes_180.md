## Mantiq 1.8.0 — trace-built expert residency

This release changes **data, not engine code**. The shipped expert profile for
`Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M` was stale: its first 859 VRAM slots covered
only 7.5-10.8% of the experts actually routed during decode. The profile is now
ranked from real decode routing traces, so the VRAM tier starts with the experts
this model and workload actually use.

### What changed

- `models_cfg/packs/Qwopus3.6 35B A3B Coder MTP Q4_K_M/expert-profile.bin`
  is the new equal-weighted profile built from two `--dump-routing` traces
  (`tools/make_profile.py ... --reorder --equal --n-expert 256 --n-layer 40`).
- The same pack's `routing-trace.bin` is replaced with the equal-weighted
  concatenation, so a fresh setup regenerates 854 of the same 859 top slots even
  if the profile file is absent.
- Originals are kept beside them as `*.pre-combined`.
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
- This is a pack data change. Other model packs keep their own profiles.
- Both portable binaries were rebuilt from this tree and embed 1.8.0; the
  verifier confirmed `has1.8.0=True` for `mntq-cpu.exe` and `mntq.exe`.

| File | SHA-256 |
| --- | --- |
| `Mantiq-Portable-CPU.zip` | `24EE884A6423FBF6E66E95E3AC61D7E1C95AB18BE31BA21213FF97337B5EA1DB` |
| `Mantiq-Portable-CUDA.zip` | `1ADD9317C42E28FD1FC806C964F786ED2B88F998462A9AA8052E543445F5A3F1` |
