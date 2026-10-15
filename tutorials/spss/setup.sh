#!/usr/bin/env bash
# Install what the tutorial needs (Linux: Google Colab, WSL, Ubuntu).
#   1. Python packages (requirements.txt)  -> the Python given by $PYTHON (default python3), via pip or uv
#   2. syllabify (Kyle Gorman, MIT)        -> third_party/syllabify (pinned commit)
# Usage: bash setup.sh [START] [END]   (default 1 2; e.g. "bash setup.sh 2" fetches syllabify only)
# Step 2 reads config.yaml and so needs step 1 (pyyaml) to have run once.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
PYTHON=${PYTHON:-python3}
SYLLABIFY_COMMIT=d816db784436e9de87ec1ef9bd11b8e229853710   # 2019-02-15
START=${1:-1}
END=${2:-2}
if ! [[ "$START" =~ ^[1-2]$ && "$END" =~ ^[1-2]$ ]] || [ "$START" -gt "$END" ]; then
  echo "Usage: bash setup.sh [START] [END]   (steps 1-2)"; exit 1
fi
step() { [ "$1" -ge "$START" ] && [ "$1" -le "$END" ]; }

if step 1; then
  echo "== 1. Python packages"
  if "$PYTHON" -m pip --version > /dev/null 2>&1; then
    "$PYTHON" -m pip install -q -r requirements.txt
  elif command -v uv > /dev/null; then          # uv-created venvs have no pip
    uv pip install -q --python "$PYTHON" -r requirements.txt
  else
    echo "no pip or uv for $PYTHON"; exit 1
  fi
fi

if step 2; then
  echo "== 2. syllabify"
  TP=$("$PYTHON" src/tutorial_lib.py paths.third_party)   # from config.yaml
  if [ ! -d "$TP/syllabify" ]; then
    git clone -q https://github.com/kylebgorman/syllabify.git "$TP/syllabify"
    git -C "$TP/syllabify" checkout -q "$SYLLABIFY_COMMIT"
  fi
  git -C "$TP/syllabify" log -1 --format="syllabify %h (%cs)"
fi

echo "== done (steps $START-$END)"
