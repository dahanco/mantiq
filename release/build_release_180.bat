@echo off
rem Rebuild both portable release binaries for Mantiq 1.8.0 (CUDA + portable CPU).
rem STRATA_PORTABLE=ON for both: the packages ship a statically linked MSVC runtime (/MT).
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )
set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"
set "SRC=C:\tmp3\project-mantiq"
set "LLAMA=%SRC%\third_party\llama.cpp"

for %%D in (build-cuda build-cpu-portable) do (
  if exist "%SRC%\%%D\CMakeCache.txt" del /q "%SRC%\%%D\CMakeCache.txt"
  if exist "%SRC%\%%D\CMakeFiles" rmdir /s /q "%SRC%\%%D\CMakeFiles"
)

echo === [1/4] CUDA configure (VERSION 1.8.0, portable /MT) ===
"%CMAKE%" -S "%SRC%" -B "%SRC%\build-cuda" -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_CUDA_ARCHITECTURES=75;80;86;89 ^
  -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe" ^
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4" ^
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_GGML_DIR="%LLAMA%"
if errorlevel 1 ( echo CUDA configure failed & exit /b 1 )

echo === [2/4] CUDA build (target mntq, Release) ===
"%CMAKE%" --build "%SRC%\build-cuda" --target mntq --config Release -j 6
if errorlevel 1 ( echo CUDA build failed & exit /b 1 )

echo === [3/4] CPU portable configure ===
"%CMAKE%" -S "%SRC%" -B "%SRC%\build-cpu-portable" -G "Visual Studio 17 2022" -A x64 ^
  -DMNTQ_ENABLE_CPU=ON -DMNTQ_CPU_PORTABLE=ON -DSTRATA_PORTABLE=ON -DMNTQ_GGML_DIR="%LLAMA%"
if errorlevel 1 ( echo CPU configure failed & exit /b 1 )

echo === [4/4] CPU portable build (target mntq-cpu, Release) ===
"%CMAKE%" --build "%SRC%\build-cpu-portable" --target mntq-cpu --config Release -j 16
if errorlevel 1 ( echo CPU build failed & exit /b 1 )

echo === DONE ===
