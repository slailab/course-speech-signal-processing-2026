#!/usr/bin/env bash
# Run the tutorial: 1 linguistic features -> 2 feature table -> 3 figures.
# Needs the outputs of the preprocess tutorial (manifest with TextGrids) and setup.sh.
# Usage: bash run.sh [START] [END]   (default 1 3)
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
export PYTHON=${PYTHON:-python3}
START=${1:-1}
END=${2:-3}
if ! [[ "$START" =~ ^[1-3]$ && "$END" =~ ^[1-3]$ ]] || [ "$START" -gt "$END" ]; then
  echo "Usage: bash run.sh [START] [END]   (steps 1-3)"; exit 1
fi
step() { [ "$1" -ge "$START" ] && [ "$1" -le "$END" ]; }

if step 1; then echo "########## Step 1. Linguistic features"; "$PYTHON" src/extract_linguistic.py; fi
if step 2; then echo "########## Step 2. Feature table";      "$PYTHON" src/show_features.py; fi
if step 3; then echo "########## Step 3. Figures";            "$PYTHON" src/plot_features.py; fi
