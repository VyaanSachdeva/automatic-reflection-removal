@echo off
rem Launches the Review-2 demo and opens it in the browser at http://127.0.0.1:7860
cd /d "%~dp0"
set PYTHONUTF8=1
echo.
echo Starting the raindrop demo. This takes about 30 seconds...
echo Your browser will open on its own at http://127.0.0.1:7860
echo Keep this window open while you use the demo. Closing it stops the demo.
echo.
".venv\Scripts\python.exe" demo_app.py
echo.
echo The demo has stopped.
pause
