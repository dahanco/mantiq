$ErrorActionPreference = 'Stop'
$src = 'C:\tmp3\project-mantiq'
$dst = 'C:\tmp3\dist\Mantiq-Portable-CPU'
$py  = 'C:\Users\doko\AppData\Local\Programs\Python\Python314'
$runtime = 'C:\tmp3\portable-runtime\python'   # the portable Python both packages ship (interpreter + packages)

if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
New-Item -ItemType Directory -Path $dst | Out-Null

# ---- the program's own files (the engine's source is kept: the start checks its hash, nothing is built)
foreach ($f in @('mntq.py','setup.py','chat.py','START-MNTQ.bat','VERSION','CMakeLists.txt',
                 'LICENSE','LICENSE.deepseek-v41','LICENSE.minimax-m31','LICENSE.qwen-community-1.0','README.md')) {
    Copy-Item (Join-Path $src $f) $dst
}
foreach ($d in @('serve','tools','src','include','cmake','assets')) {
    robocopy (Join-Path $src $d) (Join-Path $dst $d) /E /XD __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
}
robocopy (Join-Path $src 'third_party\ggml')      (Join-Path $dst 'third_party\ggml')      /E /NFL /NDL /NJH /NJS /NP | Out-Null
robocopy (Join-Path $src 'third_party\wheels')    (Join-Path $dst 'third_party\wheels')    /E /NFL /NDL /NJH /NJS /NP | Out-Null
robocopy (Join-Path $src 'third_party\llama.cpp\gguf-py') (Join-Path $dst 'third_party\llama.cpp\gguf-py') /E /XD __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
# third-party license files tracked in the tree (these directories carry a LICENSE only: no code is shipped)
foreach ($d in @('third_party\beellama.cpp','third_party\exllamav3')) {
    robocopy (Join-Path $src $d) (Join-Path $dst $d) /E /NFL /NDL /NJH /NJS /NP | Out-Null
}

# ---- no models_cfg: the package carries no expert profile, and no model of any kind.  A qwen35moe config's --spec
# needs the device residency table a profile builds, so mntq.py writes one for the pack's own geometry as the setup
# asks for it (write_expert_profile: tools/make_profile.py --no-base)

# ---- the engine, already compiled (/MT: no Visual C++ runtime to install)
New-Item -ItemType Directory -Path (Join-Path $dst 'build-cpu') -Force | Out-Null
Copy-Item (Join-Path $src 'build-cpu-portable\Release\mntq-cpu.exe') (Join-Path $dst 'build-cpu\mntq-cpu.exe')

# ---- the build stamp.  mntq.py's refresh_engine compares its "src" against the hash of the source the package
# ships and recompiles the engine when they differ: without the stamp (or with another tree's) every start tried
# to build again, and with no llama.cpp source here that build could only warn.
$stampPy = Join-Path $env:TEMP 'mntq-portable-stamp.py'
@'
import json, pathlib, sys, time
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import setup
meta = {"backend": "cpu", "src": setup.source_hash(setup.ENGINE_SOURCES),
        "llama": str(root / "third_party" / "llama.cpp"),
        "version": (root / "VERSION").read_text(encoding="utf-8").strip(),
        "date": time.strftime("%Y-%m-%d %H:%M"), "targets": ["mntq-cpu"]}
(root / "build-cpu" / "MNTQ-BUILD.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print("stamp: src=" + meta["src"] + " version=" + meta["version"])
'@ | Set-Content -Path $stampPy -Encoding UTF8
& (Join-Path $py 'python.exe') $stampPy $dst
Remove-Item $stampPy -Force

# ---- Python, inside the package: nothing to install, nothing on PATH.  It comes from the portable runtime kept
# in $runtime (a copy of the PC's Python with the packages Mantiq, the dashboard server and the pack builder need -
# numpy included), so a repack never depends on the project's .venv, which may not be there any more.
robocopy $runtime (Join-Path $dst 'python') /E /NFL /NDL /NJH /NJS /NP | Out-Null

Write-Output ("python + packages: {0:N1} MB" -f ((Get-ChildItem (Join-Path $dst 'python') -Recurse -File | Measure-Object Length -Sum).Sum / 1MB))
Write-Output ("whole tree:        {0:N1} MB" -f ((Get-ChildItem $dst -Recurse -File | Measure-Object Length -Sum).Sum / 1MB))
