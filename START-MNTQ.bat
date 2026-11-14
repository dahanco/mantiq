@echo off
rem Mantiq for Windows (experimental): the first run checks the PC, compiles the engine, asks before it
rem downloads the model, builds the pack and starts the dashboard; later runs just start it.
rem START-MNTQ.bat --help lists the options. Like ./mntq.sh, nothing is installed system-wide: a missing Python,
rem compiler or CUDA toolkit is reported with where to get it.
setlocal
title Mantiq
cd /d "%~dp0"
rem stop a previous run first: its server still holds the port (this run would then exit with code 2) and its engine
rem holds the memory
call :freeport 7711
rem a portable package carries its own Python (python\python.exe) together with Mantiq's packages: use it and skip
rem the private .venv entirely (nothing to install, and the Python on the PC is never touched)
if exist "python\python.exe" (
  "python\python.exe" mntq.py %*
  if errorlevel 1 pause
  exit /b
)
rem a .venv from an earlier run that failed half-way has a python but no pip: start it again
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m pip --version >nul 2>nul || rmdir /s /q .venv
)
if exist ".venv\Scripts\python.exe" goto run

call :findpy
if defined PY goto venv
echo.
echo  64-bit Python 3.10 or newer is needed. Install it, then double-click START-MNTQ.bat again:
echo    winget install -e --id Python.Python.3.12 --scope user
echo  or https://www.python.org/downloads/ (tick "Add python.exe to PATH").
echo.
pause
exit /b 1

:venv
rem a private environment inside this folder, so nothing is installed into the system Python
%PY% -m venv .venv
if exist ".venv\Scripts\python.exe" goto run
echo  Could not create the Python environment in .venv
pause
exit /b 1

:run
".venv\Scripts\python.exe" mntq.py %*
if errorlevel 1 pause
exit /b

:findpy
rem the py launcher first, then python on PATH (not the Microsoft Store stub), then the usual per-user folders
set "PY="
py -3 -c "import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3" & goto :eof
python -c "import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python" & goto :eof
for %%V in (313 312 311 310) do if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"" & goto :eof
goto :eof

:freeport
rem %1 = the port.  End every process of a previous run (the dashboard's server.py, the mntq.py that started it, the
rem telemetry children, the engine).  netstat reports PID 0 for an inherited listening socket, so the processes are
rem matched by their path / command line under this folder, not by the port's owner - and only python*.exe / mntq.exe,
rem so this window and PowerShell itself are never matched.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'mntq.exe' -or $_.Name -like 'python*.exe') -and ($_.ExecutablePath -like '%~dp0*' -or $_.CommandLine -like '*%~dp0*') } | ForEach-Object { Write-Host ('  stopping ' + $_.Name + ' PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
rem wait for Windows to release the port (a killed process's socket can linger a moment)
set "tries=0"
:fp_wait
ping -n 2 127.0.0.1 >nul
netstat -ano | findstr /c:":%~1 " | findstr /c:"LISTENING" >nul
if errorlevel 1 goto :eof
set /a tries+=1
if %tries% lss 6 goto fp_wait
echo  Port %~1 is still held - a process may be running as administrator; end it, then start again
goto :eof
