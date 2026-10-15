"""Print the linguistic features of every utterance as a table, one row per phone.

Each row decodes the stored 239-dim vector back into readable values: quin-phone
context, syllable stress, positions and counts (guide: docs/01_linguistic_features.md).
Usage: python src/show_features.py
"""

import numpy as np
from linguistic import NUMERIC, QUIN
from tutorial_lib import audio_root, load_config, read_manifest, utt_id

# Categorical groups shown as columns: (column title, feature name prefix)
GROUPS = [(q, f"{q}=") for q in QUIN] + [
    ("stress p/c/n", None),
    ("syl", "phone_in_syl="),
    ("ph/word", "phone_in_word="),
    ("syl/word", "syl_in_word="),
    ("word/phr", "word_in_phrase="),
    ("phr/utt", "phrase_in_utt="),
]
SHORT = {"single": "S", "begin": "B", "middle": "M", "end": "E", "none": "-"}
# Syllables whose stress is encoded: previous, current, next
SYLLABLES = ("prev", "cur", "next")
# Column titles of the numeric features, in the order of linguistic.NUMERIC
NUMERIC_TITLES = [
    "#ph/syl", "#syl/word", "#ph/word", "#word/phr",
    "#syl/phr", "#syl/utt", "#word/utt", "#phr/utt",
]  # fmt: skip


def active(row, names, prefix):
    """Value of the active (1) feature whose name starts with prefix, or "" if none."""
    for v, name in zip(row, names, strict=True):
        if v and name.startswith(prefix):
            return name[len(prefix) :]
    return ""


def main():
    cfg = load_config()
    work = cfg["paths"]["manifest"].parent
    names = (work / cfg["linguistic"]["dir"] / "feature_names.txt").read_text().split()
    entries = read_manifest(cfg["paths"]["manifest"])
    root = audio_root(entries)
    for entry in entries:
        if not entry.get("linguistic"):
            continue
        data = np.load(work / entry["linguistic"])
        phones, ling = data["phones"], data["ling"]
        print(f"== {utt_id(entry, root)}  ({len(phones)} phones x {ling.shape[1]} features)")
        header = ["phone"] + [title for title, _ in GROUPS] + NUMERIC_TITLES
        print("  " + " ".join(f"{h:>9s}" for h in header))
        for p, row in zip(phones, ling, strict=True):
            cells = [p]
            for title, prefix in GROUPS:
                if prefix is None:  # stress of the previous / current / next syllable
                    stress = [active(row, names, f"{w}_syl_stress=") for w in SYLLABLES]
                    cells.append("/".join(SHORT.get(s, s) for s in stress))
                else:
                    value = active(row, names, prefix)
                    cells.append(SHORT.get(value, value) if title != "syl" else value[:3])
            cells += [str(int(v)) for v in row[-len(NUMERIC) :]]
            print("  " + " ".join(f"{c:>9s}" for c in cells))


if __name__ == "__main__":
    main()
