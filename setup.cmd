@echo off
rem One-time setup on a new Windows machine: creates .venv, installs packages, gets the pose model.
rem Needs Python 3.10+ from python.org (tick "Add python.exe to PATH") and internet for this step only.
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PY% -m venv .venv || goto :fail
)
echo Installing packages (a few minutes the first time)...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
if not exist "models\pose_landmarker_full.task" (
    echo Downloading the pose model...
    ".venv\Scripts\python.exe" -c "import config, urllib.request; config.MODEL_PATH.parent.mkdir(exist_ok=True); urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)" || goto :fail
)
echo Running the tests...
".venv\Scripts\python.exe" -m pytest -q || goto :fail
echo.
echo Setup complete. Double-click run.cmd to start the app.
pause
exit /b 0
:fail
echo.
echo Setup failed. See the message above.
pause
exit /b 1
