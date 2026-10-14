#!/usr/bin/env bash
# Install everything the tutorial needs (Linux: Google Colab, WSL, Ubuntu).
#   1. Python packages (requirements.txt)          -> the Python given by $PYTHON (default python3), via pip or uv
#   2. ming024/FastSpeech2: text cleaners, lexicon -> third_party/FastSpeech2
#   3. ITU-T STL: P.56 level meter (actlev)        -> third_party/bin/actlev  (needs gcc)
#   4. Montreal Forced Aligner (tested version $MFA_VERSION) + English models
#                                                  -> micromamba env "mfa" (~/.local/bin/micromamba)
# Usage: bash setup.sh [START] [END]   (default 1 4; e.g. "bash setup.sh 4" installs MFA only)
# Steps 2-4 read config.yaml and so need step 1 (pyyaml) to have run once.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
MFA_VERSION=${MFA_VERSION:-3.4.3}
PYTHON=${PYTHON:-python3}
START=${1:-1}
END=${2:-4}
if ! [[ "$START" =~ ^[1-4]$ && "$END" =~ ^[1-4]$ ]] || [ "$START" -gt "$END" ]; then
  echo "Usage: bash setup.sh [START] [END]   (steps 1-4)"; exit 1
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
if [ "$END" -ge 2 ]; then
  TP=$("$PYTHON" src/tutorial_lib.py paths.third_party)   # from config.yaml (needs pyyaml from step 1)
  mkdir -p "$TP/bin"
fi

if step 2; then
  echo "== 2. FastSpeech2 (english_cleaners, librispeech-lexicon)"
  [ -d "$TP/FastSpeech2" ] || git clone -q --depth 1 https://github.com/ming024/FastSpeech2.git "$TP/FastSpeech2"
fi

if step 3; then
  echo "== 3. ITU-T STL -> actlev"
  if [ ! -x "$TP/bin/actlev" ]; then
    [ -d "$TP/STL" ] || git clone -q --depth 1 https://github.com/openitu/STL.git "$TP/STL"
    S=$TP/STL/src
    gcc -O2 -w -I"$S/utl" -I"$S/g711" -o "$TP/bin/actlev" "$S/sv56/actlevel.c" "$S/sv56/sv-p56.c" "$S/utl/ugst-utl.c" -lm
  fi
  "$TP/bin/actlev" 2>&1 | head -1 || true
fi

if step 4; then
  echo "== 4. Montreal Forced Aligner"
  MAMBA=${MICROMAMBA:-$HOME/.local/bin/micromamba}
  export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
  if [ ! -x "$MAMBA" ]; then
    mkdir -p "$(dirname "$MAMBA")"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$(dirname "$(dirname "$MAMBA")")" bin/micromamba
  fi
  [ -d "$MAMBA_ROOT_PREFIX/envs/${MFA_ENV:-mfa}" ] || "$MAMBA" create -q -y -n "${MFA_ENV:-mfa}" -c conda-forge "montreal-forced-aligner=$MFA_VERSION"
  "$MAMBA" run -n "${MFA_ENV:-mfa}" mfa model download acoustic "$("$PYTHON" src/tutorial_lib.py mfa.acoustic_model)"
  "$MAMBA" run -n "${MFA_ENV:-mfa}" mfa model download g2p "$("$PYTHON" src/tutorial_lib.py mfa.g2p_model)"
  echo "MFA $("$MAMBA" run -n "${MFA_ENV:-mfa}" mfa version)"
fi

echo "== done (steps $START-$END)"
