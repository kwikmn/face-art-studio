@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH="
where py >nul 2>nul
if errorlevel 1 goto fallback
py -3.10 -s -c "import struct,sys; print('FACEART_PYTHON_310_X64_OK') if sys.version_info[:2]==(3,10) and sys.version_info[:3]>=(3,10,6) and struct.calcsize('P')==8 else sys.exit(1)" 2>nul | findstr.exe /x /c:"FACEART_PYTHON_310_X64_OK" >nul
if errorlevel 1 goto fallback
py -3.10 -s scripts\setup_windows.py %*
goto result
:fallback
set "FACEART_PYTHON="
for /f "delims=" %%P in ('where.exe python 2^>nul') do call :accept_python "%%P"
call :accept_python "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
call :accept_python "%ProgramFiles%\Python310\python.exe"
if not defined FACEART_PYTHON goto missing
call "%FACEART_PYTHON%" -s scripts\setup_windows.py %*
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
:accept_python
if defined FACEART_PYTHON exit /b 0
if not exist "%~1" exit /b 1
if /i "%~dp1"=="%LOCALAPPDATA%\Microsoft\WindowsApps\" exit /b 1
call "%~1" -s -c "import struct,sys; print('FACEART_PYTHON_310_X64_OK') if sys.version_info[:2]==(3,10) and sys.version_info[:3]>=(3,10,6) and struct.calcsize('P')==8 else sys.exit(1)" 2>nul | findstr.exe /x /c:"FACEART_PYTHON_310_X64_OK" >nul
if errorlevel 1 exit /b 1
set "FACEART_PYTHON=%~1"
exit /b 0
