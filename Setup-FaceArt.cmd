@echo off
setlocal
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH="
where py >nul 2>nul
if errorlevel 1 goto fallback
py -3.10 -s -c "import sys" >nul 2>nul
if errorlevel 1 goto fallback
py -3.10 -s scripts\setup_windows.py %*
goto result
:fallback
where python >nul 2>nul
if errorlevel 1 goto locations
python -s scripts\setup_windows.py %*
goto result
:locations
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
  "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" -s scripts\setup_windows.py %*
  goto result
)
if exist "%ProgramFiles%\Python310\python.exe" (
  "%ProgramFiles%\Python310\python.exe" -s scripts\setup_windows.py %*
  goto result
)
goto missing
:result
set "result=%errorlevel%"
if not "%result%"=="0" echo Setup needs attention. See the message above and docs\INSTALL.md.
if /i not "%~1"=="--check-only" if /i not "%~1"=="--dry-run" pause
exit /b %result%
:missing
echo Install Python 3.10 64-bit with the Python launcher from python.org first.
echo See docs\INSTALL.md for the official link and remaining prerequisites.
pause
exit /b 1
