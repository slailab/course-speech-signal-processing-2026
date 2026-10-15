"""Text processing (guide: docs/02_text_processing.md).

raw text -> text normalization -> words -> pronunciation lexicon.
G2P for OOV words and the alignment are done in align.sh.

Reads the manifest written by process_audio.py and writes the MFA inputs (a .lab
next to each output audio file, e.g. audio/0001/example_0001.lab) and mfa/lexicon.txt,
and text_info.normalized_text into the manifest. Utterances are processed in parallel
(config: jobs). With report.verbose, every normalization sub-step and the dictionary
lookup of each word are printed (OOV words are marked); otherwise only progress.
Usage: [CONFIG=configs/<name>.yaml] python src/process_text.py
"""

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tutorial_lib import (
    audio_root,
    english_cleaners_module,
    load_config,
    mfa_safe,
    output_manifest,
    print_progress,
    read_lexicon,
    read_manifest,
    utt_id,
    write_manifest,
)

# Set in each worker process by init_worker()
_cfg = _cleaners = _lexicon = _root = None


def init_worker(cfg, lexicon, root):
    global _cfg, _cleaners, _lexicon, _root
    _cfg, _lexicon, _root = cfg, lexicon, root
    _cleaners = english_cleaners_module(cfg)


def process(entry):
    """One utterance -> (normalized text, printed lines)."""
    cl, verbose = _cleaners, _cfg["report"]["verbose"]
    text = entry["text_info"]["text"]
    lines = [f"== {utt_id(entry, _root)}", f"  raw                 : {text}"]

    # Text normalization: english_cleaners = these five sub-steps in this
    # order; print the result of each
    t = text
    for step in (
        cl.convert_to_ascii,
        cl.lowercase,
        cl.expand_numbers,
        cl.expand_abbreviations,
        cl.collapse_whitespace,
    ):
        new = step(t)
        lines.append(f"  {step.__name__:20s}: {new}{'' if new != t else '   (no change)'}")
        t = new

    # Word splitting: remove brackets and quotes that MFA would keep attached
    # to words, then write the .lab file next to the audio (MFA input)
    t = mfa_safe(t)
    (_cfg["paths"]["work"] / Path(entry["audio"]).with_suffix(".lab")).write_text(
        t + "\n", encoding="utf-8"
    )
    lines.append(f"  -> .lab             : {t}")

    # Lexicon lookup (printed only); OOV words are handled by G2P in align.sh
    if verbose:
        lines.append("  lexicon lookup:")
        for w in re.findall(r"[a-z']+", t):
            prons = _lexicon.get(w)
            lines.append(f"    {w:12s} {' | '.join(prons) if prons else 'OOV -> G2P in align.sh'}")
    return t, lines


def main():
    cfg = load_config()
    p = cfg["paths"]
    lexicon = read_lexicon(cfg["text"]["lexicon"])  # {word: [pronunciation, ...]}
    # Optional extra pronunciations: a word listed there replaces its lexicon entries
    if cfg["text"].get("lexicon_extra"):
        extra = read_lexicon(cfg["text"]["lexicon_extra"])
        lexicon.update(extra)
        print(f"lexicon_extra: {len(extra)} words from {cfg['text']['lexicon_extra'].name}")
    # The manifest written by process_audio.py: only utterances that passed it
    entries = read_manifest(output_manifest(cfg))
    root = audio_root(cfg)
    jobs = max(1, min(cfg["jobs"], len(entries)))  # no more processes than utterances
    print(f"config {cfg['name']}: {len(entries)} utterances, {jobs} jobs")

    # The lexicon is only needed in the workers for the verbose lookup
    worker_lexicon = lexicon if cfg["report"]["verbose"] else {}
    with ProcessPoolExecutor(
        jobs, initializer=init_worker, initargs=(cfg, worker_lexicon, root)
    ) as ex:
        for i, (entry, (t, lines)) in enumerate(
            zip(entries, ex.map(process, entries, chunksize=64), strict=True), 1
        ):
            entry["text_info"]["normalized_text"] = t
            if cfg["report"]["verbose"]:
                print("\n".join(lines))
            else:
                print_progress(i, len(entries))

    write_manifest(output_manifest(cfg), entries)
    # Copy of the lexicon in the MFA folder; align.sh adds the G2P pronunciations
    (p["work"] / "mfa").mkdir(exist_ok=True)
    lexicon_path = p["work"] / "mfa" / "lexicon.txt"
    with open(lexicon_path, "w", encoding="utf-8") as f:
        for word, prons in lexicon.items():
            f.writelines(f"{word}\t{pron}\n" for pron in prons)
    print(f"\nwrote .lab files, manifest.jsonl and {lexicon_path} ({len(lexicon)} words)")


if __name__ == "__main__":
    main()
