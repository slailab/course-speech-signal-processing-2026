"""Audio preprocessing (guide: docs/01_audio_preprocessing.md).

read -> resample -> high-pass (DC removal) -> VAD trim -> P.56 level.

Reads the given manifest and writes into paths.work, with the same folder
layout as the input:
  <source path>.<audio.format>
      16-bit PCM, e.g. audio/0001/example_0001.flac
  manifest.jsonl
      the given fields + audio, sampling_rate, duration (utterances that passed)
  audio_log.csv
      trim / level / exclusion record of every utterance
Utterances are processed in parallel (config: jobs). With report.verbose, the result
of every step is printed for every utterance; otherwise only progress and a summary.
Usage: [CONFIG=configs/<name>.yaml] python src/process_audio.py
"""

import copy
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from tutorial_lib import (
    INT16_MAX,
    active_speech_level,
    audio_root,
    highpass,
    load_config,
    load_vad,
    output_audio,
    output_manifest,
    print_progress,
    read_manifest,
    to_target_rate,
    utt_id,
    vad_trim_bounds,
    write_manifest,
)

FIELDS = [
    "utt_id", "status", "in_sr", "dur_in_s", "dur_out_s", "trim_head_s", "trim_tail_s",
    "act_level_db", "gain_db", "peak",
]  # fmt: skip

# Set in each worker process by init_worker()
_cfg = _vad = _root = None


def init_worker(cfg, root):
    """Per process: one thread for torch, the config and the VAD model."""
    global _cfg, _vad, _root
    import torch

    torch.set_num_threads(1)
    _cfg, _root = cfg, root
    _vad = load_vad() if cfg["audio"]["trim"] == "vad" else None


def process(entry):
    """One utterance -> (log row, output manifest entry or None, printed lines)."""
    a, p = _cfg["audio"], _cfg["paths"]
    sr = a["target_sr"]
    uid = utt_id(entry, _root)
    lines = [f"== {uid}"]

    # Read as float in [-1, 1) and convert to mono at target_sr
    y, in_sr = sf.read(p["manifest"].parent / entry["source_audio"], dtype="float64")
    y = to_target_rate(y, in_sr, sr)
    lines.append(
        f"  read      : {entry['source_audio']}, {in_sr} Hz -> {sr} Hz, {len(y) / sr:.3f} s"
    )
    row = {"utt_id": uid, "status": "ok", "in_sr": in_sr, "dur_in_s": round(len(y) / sr, 3)}

    # High-pass filter: remove DC and low-frequency noise (zero-phase)
    y = highpass(y, sr, a["hpf"]["cutoff_hz"], a["hpf"]["order"])
    lines.append(
        f"  high-pass : Butterworth order {a['hpf']['order']}, "
        f"{a['hpf']['cutoff_hz']} Hz, zero-phase"
    )

    # VAD trim: keep from the first to the last speech segment found by the
    # VAD; pauses inside the sentence are kept
    if a["trim"] == "vad":
        v = a["vad"]
        bounds = vad_trim_bounds(y, sr, _vad, v["threshold"], v["margin_ms"], v["zero_pad_s"])
        if bounds is None:
            lines.append("  VAD trim  : no speech found -> excluded")
            return {**row, "status": "no_speech"}, None, lines
        start, end = bounds
        lines.append(
            f"  VAD trim  : keep {start / sr:.3f}-{end / sr:.3f} s "
            f"(head {start / sr:.3f} s, tail {(len(y) - end) / sr:.3f} s removed)"
        )
        row.update(trim_head_s=round(start / sr, 3), trim_tail_s=round((len(y) - end) / sr, 3))
        y = y[start:end]

    # Level normalization: measure the P.56 active speech level with actlev,
    # then apply the gain to reach the target level ourselves (as a float)
    level = active_speech_level(y, sr, p["third_party"] / "bin" / "actlev")
    if level is None:
        lines.append("  P.56 level: no speech activity -> excluded")
        return {**row, "status": "no_activity"}, None, lines
    gain_db = a["level"]["target_dbov"] - level
    y = y * 10 ** (gain_db / 20)
    peak = float(np.max(np.abs(y)))
    lines.append(
        f"  P.56 level: {level:.2f} dBov -> {a['level']['target_dbov']:.0f} dBov "
        f"(gain {gain_db:+.2f} dB, peak {peak:.3f})"
    )
    row.update(
        dur_out_s=round(len(y) / sr, 3),
        act_level_db=level,
        gain_db=round(gain_db, 2),
        peak=round(peak, 3),
    )
    # Saving as 16-bit would clip: drop the utterance instead
    if peak > INT16_MAX:
        lines.append("  -> clipping at 16 bit: excluded")
        return {**row, "status": "clipping"}, None, lines

    # Save with the same folder layout as the input and record the new
    # fields in the output manifest (the given fields are kept)
    out = copy.deepcopy(entry)
    out.update(audio=output_audio(entry, _cfg), sampling_rate=sr, duration=round(len(y) / sr, 3))
    (p["work"] / out["audio"]).parent.mkdir(parents=True, exist_ok=True)
    sf.write(p["work"] / out["audio"], y, sr, subtype="PCM_16")
    return row, out, lines


def main():
    cfg = load_config()
    p = cfg["paths"]
    root = audio_root(cfg)
    given = read_manifest(p["manifest"])
    jobs = max(1, min(cfg["jobs"], len(given)))  # no more processes than utterances
    print(f"config {cfg['name']}: {len(given)} utterances, {jobs} jobs")

    rows, entries = [], []
    # map() returns the results in manifest order, whatever order they finish in
    with ProcessPoolExecutor(jobs, initializer=init_worker, initargs=(cfg, root)) as ex:
        for i, (row, out, lines) in enumerate(ex.map(process, given, chunksize=16), 1):
            rows.append(row)
            if out is not None:
                entries.append(out)
            if cfg["report"]["verbose"]:
                print("\n".join(lines))
            else:
                print_progress(i, len(given))

    p["work"].mkdir(parents=True, exist_ok=True)
    with open(p["work"] / "audio_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    write_manifest(output_manifest(cfg), entries)
    status = dict(Counter(r["status"] for r in rows))
    print(f"\nstatus: {status}")
    print(f"wrote {p['work'] / root}, manifest.jsonl ({len(entries)} utterances) and audio_log.csv")


if __name__ == "__main__":
    main()
