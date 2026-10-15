"""Print the MFA alignment (TextGrid) as a word / phone table.

Covers every utterance in the output manifest (guide: docs/03_alignment.md).
Usage: python src/show_alignment.py
"""

import tgt
from tutorial_lib import audio_root, load_config, output_manifest, read_manifest, utt_id


def main():
    cfg = load_config()
    if not cfg["report"]["verbose"]:
        print(f"report.verbose is off in config {cfg['name']}: no alignment tables")
        return
    work, root = cfg["paths"]["work"], audio_root(cfg)
    for entry in read_manifest(output_manifest(cfg)):
        if not entry.get("textgrid"):
            continue
        tg = tgt.io.read_textgrid(str(work / entry["textgrid"]), include_empty_intervals=True)
        phones = tg.get_tier_by_name("phones").intervals
        print(f"== {utt_id(entry, root)}")
        print(f"  {'start':>6s} {'end':>6s}  {'word':12s} phones")
        # One row per word interval (empty label = silence) with the phones
        # inside it
        for w in tg.get_tier_by_name("words").intervals:
            ph = [
                p.text
                for p in phones
                if p.start_time >= w.start_time - 1e-6 and p.end_time <= w.end_time + 1e-6
            ]
            label = w.text or f"(silence {round((w.end_time - w.start_time) * 1000)} ms)"
            phone_str = " ".join(p for p in ph if p)
            print(f"  {w.start_time:6.2f} {w.end_time:6.2f}  {label:12s} {phone_str}")


if __name__ == "__main__":
    main()
