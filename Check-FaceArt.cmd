@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH="
if not exist "venv\Scripts\python.exe" (
  echo Run Setup-FaceArt.cmd first.
  pause
  exit /b 1
)
"venv\Scripts\python.exe" -s scripts\check_install.py %*
set "result=%errorlevel%"
pause
exit /b %result%
