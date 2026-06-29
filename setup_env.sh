#!/usr/bin/env bash
# Set up the Python environment. Run from the project root in Git Bash:
#     bash setup_env.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"   # override with PYTHON=py if needed on Windows

echo ">> Creating virtual environment in .venv"
"$PYTHON" -m venv .venv

# Git Bash uses the Windows venv layout (Scripts/), not bin/
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate          # Windows / Git Bash
else
    source .venv/bin/activate              # Linux / macOS
fi

echo ">> Upgrading pip and installing requirements"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo ">> Done. Activate later with:"
echo "     source .venv/Scripts/activate    # Git Bash on Windows"
echo "     source .venv/bin/activate         # Linux/macOS"
