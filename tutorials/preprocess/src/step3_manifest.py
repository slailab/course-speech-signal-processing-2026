"""Step 3 (end). Add the alignment to the manifest.

Fills textgrid (path) and text_info.phonemes: the aligned phones, words
separated by " | " and pauses written as "sil",
e.g. "S EH1 D | M IH1 S T ER0 | sil | ER1 W AH0 N | ...".
Called by step3_align.sh.
Usage: python src/step3_manifest.py
"""

import tgt
from tutorial_lib import (
    audio_root,
    load_config,
    output_manifest,
    read_manifest,
    utt_id,
    write_manifest,
)


def phoneme_string(tg):
    """Aligned phones of a TextGrid, grouped by word ("S EH1 D | ... | sil")."""
    phones = tg.get_tier_by_name("phones").intervals
    words = []
    for w in tg.get_tier_by_name("words").intervals:
        # An empty word label is a silence inserted by the aligner
        if not w.text:
            words.append("sil")
            continue
        # The phones whose interval lies inside this word (small tolerance for
        # rounding of the times)
        ph = [
            p.text
            for p in phones
            if p.text and p.start_time >= w.start_time - 1e-6 and p.end_time <= w.end_time + 1e-6
        ]
        words.append(" ".join(ph))
    return " | ".join(words)


def main():
    cfg = load_config()
    work, root = cfg["paths"]["work"], audio_root(cfg)
    entries = read_manifest(output_manifest(cfg))
    for entry in entries:
        # MFA keeps the corpus layout: textgrids/<speaker>/<name>.TextGrid
        path = work / "textgrids" / f"{utt_id(entry, root)}.TextGrid"
        if not path.exists():  # not aligned
            continue
        tg = tgt.io.read_textgrid(str(path), include_empty_intervals=True)
        entry["textgrid"] = path.relative_to(work).as_posix()
        entry["text_info"]["phonemes"] = phoneme_string(tg)
    write_manifest(output_manifest(cfg), entries)
    n = sum(e.get("textgrid") is not None for e in entries)
    print(f"manifest.jsonl: textgrid and phonemes for {n} of {len(entries)} utterances")


if __name__ == "__main__":
    main()
