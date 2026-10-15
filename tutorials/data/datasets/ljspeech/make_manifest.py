"""Write manifest.jsonl for LJSpeech 1.1, in the course manifest format.

The manifest is written at the root of the dataset folder; source_audio points into the
original layout (wavs/<id>.wav), which is never modified. Fields filled by preprocessing
are null. Uses the Python standard library only.

metadata.csv has three columns: id | transcription | normalized transcription (numbers,
ordinals and monetary units written out as words). text takes the normalized column, as
FastSpeech2 does for LJSpeech; the rest of text normalization (abbreviations, case,
punctuation) is done by the preprocess tutorial. text_org keeps the transcription.
Usage: python make_manifest.py <LJSpeech-1.1 folder>
"""

import argparse
import csv
import json
from pathlib import Path

SAMPLING_RATE = 22050


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", type=Path, help="the extracted LJSpeech-1.1 folder")
    root = ap.parse_args().root

    entries, missing = [], 0
    with open(root / "metadata.csv", encoding="utf-8", newline="") as f:
        # The transcriptions contain quotes: read them literally
        for utt, transcription, normalized in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            audio = f"wavs/{utt}.wav"
            if not (root / audio).exists():
                missing += 1
                continue
            entries.append({
                "audio": None,
                "source_audio": audio,
                "sampling_rate": None,
                "sampling_rate_org": SAMPLING_RATE,
                "duration": None,
                "text_info": {
                    "text": normalized,
                    "text_org": transcription,
                    "normalized_text": None,
                    "phonemes": None,
                },
                "speaker_info": {"speaker_id": "LJ", "gender": "female"},
                "textgrid": None,
                "language": "en",
                "dataset": "LJSpeech",
            })  # fmt: skip

    out = root / "manifest.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    print(f"wrote {out} ({len(entries)} utterances, {missing} missing audio files)")


if __name__ == "__main__":
    main()
