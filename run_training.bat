@echo off
echo Starting ECG Training...
echo Logs will be saved to training_output.txt and training_error.txt
echo.

cd /d "%~dp0implementation"

REM Get timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%_%dt:~8,6%"

REM Run Python with full path
"D:\Anaconda3\envs\envSSL_ECG\python.exe" main.py > "..\training_output_%timestamp%.txt" 2> "..\training_error_%timestamp%.txt"

echo.
echo Training completed. Check the log files for results.
pause
