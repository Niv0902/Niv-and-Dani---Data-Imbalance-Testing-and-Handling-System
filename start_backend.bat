@echo off
echo Starting ImbalanceKit backend...
cd /d "%~dp0backend"
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001
