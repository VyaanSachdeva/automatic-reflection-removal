@echo off
rem Launches the Review-2 demo and opens it in the browser at http://127.0.0.1:7860
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" demo_app.py
pause
