@echo off
rem Q-Bench cause-isolation runner (paper experiment): double-click to start. Closing this window stops the run.
setlocal
set QB_MODE=paper
set QB_NEED_GB=100
cd /d "%~dp0"
set "GITBASH=%ProgramFiles%\Git\bin\bash.exe"
if not exist "%GITBASH%" (
  echo Git Bash not found: %GITBASH%
  pause
  exit /b 1
)
"%GITBASH%" ./run_qbench.sh
echo.
echo Runner exited with code %ERRORLEVEL%.
pause
