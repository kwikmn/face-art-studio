@echo off
setlocal
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH="
if not exist "venv\Scripts\python.exe" (
  echo Run Setup-FaceArt.cmd first. See docs\INSTALL.md.
  pause
  exit /b 1
)
"venv\Scripts\python.exe" -s scripts\launch_studio.py %*
set "result=%errorlevel%"
if not "%result%"=="0" pause
exit /b %result%
