@echo off
REM COTT Calculator launcher for Windows
REM Tries conda first, then venv, then system Python.

cd /d "%~dp0"

REM --- Option 1: Conda environment ---
where conda >nul 2>&1
if %ERRORLEVEL% equ 0 (
    call conda activate traction 2>nul
    if %ERRORLEVEL% equ 0 (
        echo Using conda environment: traction
        python calculator.py
        exit /b
    )
    echo Conda found but 'traction' env missing. Creating it...
    call conda env create -f environment.yml
    call conda activate traction
    python calculator.py
    exit /b
)

REM --- Option 2: Local venv ---
if exist ".venv\Scripts\python.exe" (
    echo Using existing venv
    .venv\Scripts\python.exe calculator.py
    exit /b
)

REM --- Option 3: Create venv with system Python ---
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo No environment found. Creating .venv...
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo Environment ready. Launching calculator...
.venv\Scripts\python.exe calculator.py
