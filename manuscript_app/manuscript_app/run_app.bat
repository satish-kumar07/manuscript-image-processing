@echo off
REM Starts the Manuscript Cleaner website on this computer: http://127.0.0.1:8000
cd /d "%~dp0"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
