@echo off
REM ---------------------------------------------------------------
REM  Builds VehicleSpeedTracker.exe  (run this on Windows)
REM  Double-click this file, or run it from Command Prompt.
REM ---------------------------------------------------------------
cd /d "%~dp0"

echo Installing dependencies...
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :error

echo Building the application (this takes a few minutes)...
python -m PyInstaller --noconfirm --onefile --windowed --name VehicleSpeedTracker app.py
if errorlevel 1 goto :error

echo.
echo DONE.  Your app is here:  dist\VehicleSpeedTracker.exe
echo Tip: copy the exe to any folder, together with your sample videos.
pause
exit /b 0

:error
echo.
echo Something went wrong. Check that Python is installed and "python" works in Command Prompt.
pause
exit /b 1
