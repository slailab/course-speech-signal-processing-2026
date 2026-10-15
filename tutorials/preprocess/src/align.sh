#!/usr/bin/env bash
# Forced alignment with MFA (guides: docs/03_alignment.md; OOV handling in
# docs/02_text_processing.md).
#   OOV words -> G2P -> dictionary -> align -> TextGrid -> manifest
# Inputs come from process_audio.py and process_text.py (audio/<speaker>/<name>.flac
# + .lab, mfa/lexicon.txt, manifest.jsonl). MFA runs with the config's jobs; with
# report.verbose the OOV words and their G2P pronunciations are printed, otherwise
# only their numbers.
# Usage: [CONFIG=configs/<name>.yaml] bash src/align.sh
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
# Value from the config (paths come back absolute)
cfg() { "${PYTHON:-python3}" tutorial_lib.py "$1"; }

WORK=$(cfg paths.work)
ACOUSTIC_MODEL=$(cfg mfa.acoustic_model)
G2P_MODEL=$(cfg mfa.g2p_model)
CORPUS=$(cfg corpus_dir)   # output audio + .lab; sub-folders = speakers
JOBS=$(cfg jobs)
VERBOSE=$(cfg report.verbose)   # True / False
M=$WORK/mfa
mkdir -p "$M"
# MFA runs inside its own micromamba environment (installed by setup.sh)
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
mfa() { "${MICROMAMBA:-$HOME/.local/bin/micromamba}" run -n "${MFA_ENV:-mfa}" mfa "$@"; }
# MFA keeps its database and features in <temporary_directory>/<corpus folder
# name>: use a directory of our own per config (on the Linux filesystem; sqlite
# locking is unreliable on /mnt/c)
TMP=${MFA_TMP:-$HOME/.cache/mfa_tutorial}/$(cfg name)
mkdir -p "$TMP"
echo "config $(cfg name): $JOBS jobs, temporary files in $TMP"

# Find the words that are not in the lexicon (OOV)
echo "== OOV words (not in the lexicon)"
mfa find_oovs "$CORPUS" "$M/lexicon.txt" "$M/oovs" --clean -t "$TMP" -j "$JOBS" \
  > "$M/find_oovs.log" 2>&1
OOV_LIST=$(ls "$M"/oovs/oovs_found*.txt 2>/dev/null | head -1 || true)
if [ -n "$OOV_LIST" ] && [ -s "$OOV_LIST" ]; then  # MFA writes no list when there are no OOVs
  if [ "$VERBOSE" = True ]; then cat "$OOV_LIST"; else echo "$(wc -l < "$OOV_LIST") OOV words"; fi
  # G2P gives pronunciation candidates; all of them are merged into the
  # lexicon and the aligner picks one
  echo "== G2P pronunciations for the OOV words"
  mfa g2p "$OOV_LIST" "$G2P_MODEL" "$M/oov.dict" --clean -t "$TMP" -j "$JOBS" \
    > "$M/g2p.log" 2>&1
  if [ "$VERBOSE" = True ]; then cat "$M/oov.dict"; else echo "$(wc -l < "$M/oov.dict") pronunciations"; fi
  cat "$M/lexicon.txt" "$M/oov.dict" > "$M/dictionary.txt"
else
  echo "(none)"
  rm -f "$M/oov.dict"
  cp "$M/lexicon.txt" "$M/dictionary.txt"
fi

# Align and write one TextGrid per utterance, with the
# same speaker folder layout as the corpus (+ alignment_analysis.csv)
echo "== Alignment"
mfa align "$CORPUS" "$M/dictionary.txt" "$ACOUSTIC_MODEL" "$WORK/textgrids" --clean -t "$TMP" \
  -j "$JOBS" > "$M/align.log" 2>&1
echo "$(find "$WORK/textgrids" -name '*.TextGrid' | wc -l) TextGrids in $WORK/textgrids"
echo "logs: $M/{find_oovs,g2p,align}.log"

# Record textgrid and text_info.phonemes in the output manifest
echo "== Manifest"
"${PYTHON:-python3}" record_alignment.py
