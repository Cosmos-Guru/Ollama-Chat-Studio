@echo off
title Ollama Chat Studio
cd /d "%~dp0"

REM ── First-run setup: create venv and install dependencies ──
if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo  ============================================
    echo   First-run setup — creating virtual environment...
    echo  ============================================
    echo.
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo  ERROR: Python not found. Install Python 3.10+ from https://www.python.org/downloads/
        echo  Make sure "Add Python to PATH" is checked during installation.
        echo.
        pause
        exit /b 1
    )
    call ".venv\Scripts\activate.bat"
    echo  Installing dependencies...
    pip install -r requirements.txt --quiet
    echo.
    echo  Setup complete!
    echo.
) else (
    call ".venv\Scripts\activate.bat"
)

REM ── Check if Ollama is reachable ──
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo.
    echo  WARNING: Ollama does not appear to be running on localhost:11434
    echo  Start Ollama first, or the app will not be able to connect.
    echo  Download Ollama: https://ollama.com/download
    echo.
)

REM ── Launch the app ──
echo  Starting Ollama Chat Studio...
echo.
python ollama_chat.py

pause
