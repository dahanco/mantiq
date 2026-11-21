# Mantiq release checklist

The scripts in this directory are the exact ones used to build the 1.7.6 portable packages. They are kept in git so
the release is reviewable. They currently assume the checkout is `C:\tmp3\project-mantiq`, the portable Python runtime
is `C:\tmp3\portable-runtime\python`, and the output tree is `C:\tmp3\dist`; change those three paths at the top of the
scripts if the checkout moves.

## Steps

1. Set `VERSION` and commit it. Both `CMakeLists.txt` and the build stamps read that file.
2. `release\build_release_176.bat` - configures and builds `build-cuda\Release\mntq.exe` and
   `build-cpu-portable\Release\mntq-cpu.exe` with `STRATA_PORTABLE=ON` (`/MT`).
3. `powershell -File release\pack-portable-cpu.ps1` and `...\pack-portable-cuda.ps1` - copy sources, engine, and the
   portable Python into `C:\tmp3\dist\Mantiq-Portable-{CPU,CUDA}` and write `MNTQ-BUILD.json` stamps.
4. `powershell -File release\_repack.ps1` - zip both trees and write LF-terminated `.sha256` sidecars.
5. `python release\_zip_verify.py` - check structure, VERSION, embedded engine version, and hash.
6. Re-run `decode_mntq.ps1 -Exe ...\build-cuda\Release\mntq.exe` and the 5-turn prompt-cache harness against the
   shipping binary. Confirm the output hash `02039B7067E348731EC4A6B9A29B10D0CC3EBC725693DC80A93DE29AA1D9BB49`,
   chunk 4096, arena 1936 MiB, and that each turn after the first reuses the cached prefix.
7. Fill the hashes in `release\_release_notes_176.md`, tag `vX.Y.Z`, push, and create the GitHub release with both
   zips and both sidecars.

## Notes

- The CUDA zip contains `sm_75/80/86/89` only; it does not run on compute capability 7.0/7.2 cards.
- The CPU zip is AVX2-only; BMI2 is disabled for the portable build because ggml has no runtime dispatch for it.
- No Vulkan package is produced by these scripts. The published Vulkan zip is an older 1.6.2 experiment.
