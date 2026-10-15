"""Extract phoneme-level linguistic features (guide: docs/01_linguistic_features.md).

TextGrid -> phones and words -> syllables and phrases -> (N, 239) features.

Reads the output manifest of the preprocess tutorial and writes, next to it,
with the same folder layout as the audio:
  <linguistic.dir>/<id>.npz         phones (N,), ling (N, 239) float32
  <linguistic.dir>/feature_names.txt  the 239 dimension names, one per line
and the field "linguistic" (path of the .npz) into the manifest. Utterances
without a TextGrid or with an unknown word (spn) are skipped.
Usage: python src/extract_linguistic.py
"""

from pathlib import Path

import numpy as np
from linguistic import N_FEATURES, annotate, feature_names, phone_features
from tutorial_lib import (
    SIL,
    audio_root,
    load_config,
    load_syllabify,
    phone_word_index,
    read_alignment,
    read_manifest,
    utt_id,
    write_manifest,
)


def print_structure(phones, word_idx, words, syllabify):
    """Print the syllables of each word and the phrases, as found by annotate()."""
    _, syllables, word_phones, phrases, _, utt = annotate(phones, word_idx, syllabify)
    print(
        f"  {len(phones)} phones ({phones.count(SIL)} sil), {utt['n_syls']} syllables, "
        f"{utt['n_words']} words, {utt['n_phrases']} phrases"
    )
    for k in sorted(word_phones):
        sylls = [s for s in syllables if s["word"] == k]
        text = " . ".join(" ".join(phones[i] for i in s["phones"]) for s in sylls)
        print(f"    {words[k][0]:12s} {text}")
    for j, ph in enumerate(phrases):
        print(f"    phrase {j}: {' '.join(words[k][0] for k in sorted({word_idx[i] for i in ph}))}")


def main():
    cfg = load_config()
    manifest = cfg["paths"]["manifest"]
    work = manifest.parent
    out_dir = work / cfg["linguistic"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    syllabify = load_syllabify(cfg)
    entries = read_manifest(manifest)
    root = audio_root(entries)

    n_ok = 0
    for entry in entries:
        uid = utt_id(entry, root)
        print(f"== {uid}")
        # Skip utterances that cannot be used
        if not entry.get("textgrid"):
            print("  no TextGrid -> skipped")
            continue
        phones, times, words = read_alignment(work / entry["textgrid"])
        if "spn" in phones:
            print("  unknown word (spn) in the alignment -> skipped")
            continue

        # Phone -> word membership, then the features
        word_idx = phone_word_index(times, words)
        print_structure(phones, word_idx, words, syllabify)
        ling = phone_features(phones, word_idx, syllabify)
        assert ling.shape == (len(phones), N_FEATURES)

        # Save with the same folder layout as the audio and record it in the manifest
        out = out_dir / f"{uid}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, phones=np.array(phones), ling=ling)
        entry["linguistic"] = Path(out).relative_to(work).as_posix()
        n_ok += 1

    (out_dir / "feature_names.txt").write_text("\n".join(feature_names()) + "\n")
    write_manifest(manifest, entries)
    print(f"\nwrote {out_dir} ({n_ok} of {len(entries)} utterances) and the manifest field")


if __name__ == "__main__":
    main()
