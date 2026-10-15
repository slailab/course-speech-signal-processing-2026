#!/usr/bin/env bash
# Run the tutorial: 1 audio -> 2 text -> 3 alignment -> 4 figures.
# The config is $CONFIG (default configs/example_dataset_16k.yaml); its report
# settings decide how much is printed and whether figures are drawn.
# Run setup.sh once before.
# Usage: [CONFIG=configs/<name>.yaml] bash run.sh [START] [END]
#        (default 1 4; e.g. "bash run.sh 4" redraws the figures)
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
export PYTHON=${PYTHON:-python3}   # also used by the bash steps to read the config
START=${1:-1}
END=${2:-4}
if ! [[ "$START" =~ ^[1-4]$ && "$END" =~ ^[1-4]$ ]] || [ "$START" -gt "$END" ]; then
  echo "Usage: [CONFIG=configs/<name>.yaml] bash run.sh [START] [END]   (steps 1-4)"; exit 1
fi
step() { [ "$1" -ge "$START" ] && [ "$1" -le "$END" ]; }
echo "config: $("$PYTHON" src/tutorial_lib.py name)"

if step 1; then echo "########## Step 1. Audio preprocessing";   "$PYTHON" src/process_audio.py; fi
if step 2; then echo "########## Step 2. Text processing";       "$PYTHON" src/process_text.py; fi
if step 3; then echo "########## Step 3. Forced alignment (MFA)"; bash src/align.sh; "$PYTHON" src/show_alignment.py; fi
if step 4; then echo "########## Step 4. Figures";               "$PYTHON" src/plot_audio.py; "$PYTHON" src/plot_alignment.py; fi
