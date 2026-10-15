"""Shared helpers for the SPSS tutorial steps.

Config, manifest and reading the forced alignment (TextGrid).
"""

import json
import os
import sys
from pathlib import Path

import tgt
import yaml

# Tutorial root: config.yaml, docs/, third_party/
ROOT = Path(__file__).resolve().parent.parent
# One symbol for every silence: pauses marked by the aligner and the context
# outside the utterance
SIL = "sil"
# MFA writes silence as an empty label
SILENCE_LABELS = {"", "sil", "sp"}


# ----------------------------------------------------------------- config / manifest


def load_config(path=ROOT / "config.yaml"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("manifest", "third_party"):
        cfg["paths"][key] = (ROOT / cfg["paths"][key]).resolve()
    return cfg


def config_value(key):
    """Value of a dotted config key, e.g. "paths.third_party" (used by the bash scripts)."""
    value = load_config()
    for part in key.split("."):
        value = value[part]
    return value


def read_manifest(path):
    """JSON Lines manifest: one utterance (a JSON object) per line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)


def audio_root(entries):
    """Common folder of all audio files in the manifest, e.g. "audio".

    Sub-folders below it are speakers.
    """
    folders = [Path(e["audio"]).parent.as_posix() for e in entries]
    return Path(os.path.commonpath(folders))


def utt_id(entry, root):
    """Utterance id: audio path below the audio root without extension.

    E.g. "0001/example_0001", the same id as in the preprocess tutorial.
    """
    return Path(entry["audio"]).with_suffix("").relative_to(root).as_posix()


def file_label(uid):
    """Utterance id as a flat file name: "0001/example_0001" -> "0001_example_0001"."""
    return uid.replace("/", "_")


# ----------------------------------------------------------------- alignment


def read_alignment(path):
    """Phones and words of an MFA TextGrid.

    Returns (phones, times, words):
      phones: labels as in the lexicon (ARPAbet with stress) or SIL; adjacent
              silences are merged
      times:  (start, end) in seconds for each phone; contiguous over the file
      words:  (word, start, end) for each non-silent word
    """
    tg = tgt.io.read_textgrid(str(path), include_empty_intervals=True)
    phones, times = [], []
    for iv in tg.get_tier_by_name("phones").intervals:
        label = SIL if iv.text.strip() in SILENCE_LABELS else iv.text.strip()
        if phones and label == SIL and phones[-1] == SIL:
            times[-1] = (times[-1][0], iv.end_time)
            continue
        phones.append(label)
        times.append((iv.start_time, iv.end_time))
    words = [
        (iv.text.strip(), iv.start_time, iv.end_time)
        for iv in tg.get_tier_by_name("words").intervals
        if iv.text.strip() not in SILENCE_LABELS
    ]
    return phones, times, words


def phone_word_index(times, words):
    """Index into `words` for each phone; None for phones outside any word (silences)."""
    idx = []
    for start, end in times:
        mid = (start + end) / 2
        idx.append(next((k for k, (_, ws, we) in enumerate(words) if ws <= mid < we), None))
    return idx


def load_syllabify(cfg):
    """syllabify() of Kyle Gorman's syllabify (MIT), fetched by setup.sh."""
    sys.path.insert(0, str(cfg["paths"]["third_party"] / "syllabify"))
    from syllabify import syllabify

    return syllabify


# python src/tutorial_lib.py <dotted.key>  -> prints a config value
if __name__ == "__main__":
    print(config_value(sys.argv[1]))
