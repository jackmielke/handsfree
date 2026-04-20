#!/bin/bash
# One-shot setup: creates .venv and installs deps.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. install it via 'brew install python' or xcode-select."
  exit 1
fi

python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cat <<'EOF'

✓ setup complete.

to run:
  cd ~/dev/handsfree
  source .venv/bin/activate
  python main.py

first run will:
  1. download the face landmarker model (~3 MB)
  2. ask for camera access (grant to Terminal / your IDE)
  3. ask for accessibility access so pyautogui can move the cursor

(System Settings > Privacy & Security > Camera, and > Accessibility)
EOF
