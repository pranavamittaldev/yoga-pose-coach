@echo off
rem Launch Yoga Pose Coach. Extra arguments are passed through, e.g. run.cmd --source clip.mp4
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py %*
