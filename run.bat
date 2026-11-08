@echo off
title ULTRON - Sovereign Local AI Assistant
echo ===================================================
echo             ULTRON LOCAL AI ASSISTANT             
echo ===================================================
echo Checking environment...

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe!
    echo Please make sure the .venv directory is present.
    pause
    exit /b 1
)

echo Starting Ultron Web Interface on http://127.0.0.1:5000 ...
echo [NOTE] Ensure 'ollama serve' is running in the background for local Llama 3 generation.
echo.

start http://127.0.0.1:5000
.\.venv\Scripts\python.exe ui\app.py

pause

