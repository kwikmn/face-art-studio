@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo Run Setup-FaceArt.cmd first.
  pause
  exit /b 1
)
"venv\Scripts\python.exe" -s scripts\check_install.py %*
set "result=%errorlevel%"
pause
exit /b %result%
