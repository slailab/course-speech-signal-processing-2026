#!/usr/bin/env bash
# Download LJSpeech 1.1 (public domain; 13,100 utterances of one speaker, 22.05 kHz,
# about 2.6 GB) and write its manifest.
#   tutorials/data/raw/LJSpeech-1.1/   the original layout as extracted, never modified
#   tutorials/data/raw/LJSpeech-1.1/manifest.jsonl
# Skips the download when the dataset is already there.
# Usage: bash download.sh [RAW_DIR]   (default: tutorials/data/raw)
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
RAW=${1:-$HERE/../../raw}
URL=https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

mkdir -p "$RAW"
if [ ! -d "$RAW/LJSpeech-1.1/wavs" ]; then
  echo "== downloading $URL"
  curl -L --fail -o "$RAW/LJSpeech-1.1.tar.bz2" "$URL"
  echo "== extracting"
  tar -xjf "$RAW/LJSpeech-1.1.tar.bz2" -C "$RAW"
  rm "$RAW/LJSpeech-1.1.tar.bz2"
fi
echo "== manifest"
"${PYTHON:-python3}" "$HERE/make_manifest.py" "$RAW/LJSpeech-1.1"
