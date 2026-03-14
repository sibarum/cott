#!/usr/bin/env bash
# COTT Calculator launcher for macOS / Linux
# Tries conda first, then venv, then system Python.

set -e
cd "$(dirname "$0")"

# --- Option 1: Conda environment ---
if command -v conda &>/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    if conda activate traction 2>/dev/null; then
        echo "Using conda environment: traction"
        python calculator.py
        exit 0
    fi
    echo "Conda found but 'traction' env missing. Creating it..."
    conda env create -f environment.yml
    conda activate traction
    python calculator.py
    exit 0
fi

# --- Option 2: Existing venv ---
if [ -f ".venv/bin/python" ]; then
    echo "Using existing venv"
    .venv/bin/python calculator.py
    exit 0
fi

# --- Option 3: Create venv with system Python ---
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python not found. Please install Python 3.10+."
    echo "  macOS:  brew install python"
    echo "  Ubuntu: sudo apt install python3 python3-venv python3-tk"
    exit 1
fi

# On Linux, tkinter may need a separate package
if [ "$(uname)" = "Linux" ]; then
    "$PYTHON" -c "import tkinter" 2>/dev/null || {
        echo "Warning: tkinter not found. Install it with:"
        echo "  sudo apt install python3-tk   (Debian/Ubuntu)"
        echo "  sudo dnf install python3-tkinter   (Fedora)"
        exit 1
    }
fi

echo "No environment found. Creating .venv..."
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo
echo "Environment ready. Launching calculator..."
.venv/bin/python calculator.py
