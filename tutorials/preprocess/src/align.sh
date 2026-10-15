#!/usr/bin/env bash
# Forced alignment with MFA (guides: docs/03_alignment.md; OOV handling in
# docs/02_text_processing.md).
#   OOV words -> G2P -> dictionary -> align -> TextGrid -> manifest
# Inputs come from process_audio.py and process_text.py (audio/<speaker>/<name>.flac
# + .lab, mfa/lexicon.txt, manifest.jsonl).
# Usage: bash src/align.sh
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
# Value from config.yaml (paths come back absolute)
cfg() { "${PYTHON:-python3}" tutorial_lib.py "$1"; }

WORK=$(cfg paths.work)
ACOUSTIC_MODEL=$(cfg mfa.acoustic_model)
G2P_MODEL=$(cfg mfa.g2p_model)
CORPUS=$(cfg corpus_dir)   # output audio + .lab; sub-folders = speakers
M=$WORK/mfa
# MFA runs inside its own micromamba environment (installed by setup.sh)
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
mfa() { "${MICROMAMBA:-$HOME/.local/bin/micromamba}" run -n "${MFA_ENV:-mfa}" mfa "$@"; }
# MFA keeps its database and features in <temporary_directory>/<corpus folder
# name>: use a directory of our own (on the Linux filesystem; sqlite locking is
# unreliable on /mnt/c)
TMP=${MFA_TMP:-$HOME/.cache/mfa_tutorial}
mkdir -p "$TMP"

# Find the words that are not in the lexicon (OOV)
echo "== 3-1. OOV words (not in the lexicon)"
mfa find_oovs "$CORPUS" "$M/lexicon.txt" "$M/oovs" --clean -t "$TMP" > "$M/find_oovs.log" 2>&1
OOV_LIST=$(ls "$M"/oovs/oovs_found*.txt 2>/dev/null | head -1 || true)
if [ -n "$OOV_LIST" ] && [ -s "$OOV_LIST" ]; then  # MFA writes no list when there are no OOVs
  cat "$OOV_LIST"
  # G2P gives pronunciation candidates; all of them are merged into the
  # lexicon and the aligner picks one
  echo "== 3-2. G2P pronunciations for the OOV words"
  mfa g2p "$OOV_LIST" "$G2P_MODEL" "$M/oov.dict" --clean -t "$TMP" > "$M/g2p.log" 2>&1
  cat "$M/oov.dict"
  cat "$M/lexicon.txt" "$M/oov.dict" > "$M/dictionary.txt"
else
  echo "(none)"
  rm -f "$M/oov.dict"
  cp "$M/lexicon.txt" "$M/dictionary.txt"
fi

# Align and write one TextGrid per utterance, with the
# same speaker folder layout as the corpus (+ alignment_analysis.csv)
echo "== 3-3. Alignment"
mfa align "$CORPUS" "$M/dictionary.txt" "$ACOUSTIC_MODEL" "$WORK/textgrids" --clean -t "$TMP" > "$M/align.log" 2>&1
ls "$WORK/textgrids"
echo "logs: $M/{find_oovs,g2p,align}.log"

# Record textgrid and text_info.phonemes in the output manifest
echo "== 3-4. Manifest"
"${PYTHON:-python3}" record_alignment.py
