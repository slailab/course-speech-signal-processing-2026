"""Step 1. Audio preprocessing (guide: docs/01_audio_preprocessing.md).

read -> resample -> high-pass (DC removal) -> VAD trim -> P.56 level.

Reads the given manifest and writes into paths.work, with the same folder
layout as the input:
  <source path>.<audio.format>
      16-bit PCM, e.g. audio/0001/example_0001.flac
  manifest.jsonl
      the given fields + audio, sampling_rate, duration (utterances that passed)
  audio_log.csv
      trim / level / exclusion record of every utterance
Usage: python src/step1_audio.py
"""

import copy
import csv

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
    read_manifest,
    to_target_rate,
    utt_id,
    vad_trim_bounds,
    write_manifest,
)


def main():
    cfg = load_config()
    a, p = cfg["audio"], cfg["paths"]
    sr = a["target_sr"]
    actlev = p["third_party"] / "bin" / "actlev"  # P.56 level meter (setup.sh)
    vad = load_vad() if a["trim"] == "vad" else None
    root = audio_root(cfg)

    # rows: one log line per utterance (audio_log.csv), including excluded ones
    # entries: output manifest lines, only for utterances that passed
    rows, entries = [], []
    for entry in read_manifest(p["manifest"]):
        uid = utt_id(entry, root)
        path = p["manifest"].parent / entry["source_audio"]
        print(f"== {uid}")

        # Read as float in [-1, 1) and convert to mono at target_sr
        y, in_sr = sf.read(path, dtype="float64")
        y = to_target_rate(y, in_sr, sr)
        print(f"  read      : {entry['source_audio']}, {in_sr} Hz -> {sr} Hz, {len(y) / sr:.3f} s")
        row = {
            "utt_id": uid,
            "status": "ok",
            "in_sr": in_sr,
            "dur_in_s": round(len(y) / sr, 3),
        }

        # High-pass filter: remove DC and low-frequency noise (zero-phase)
        y = highpass(y, sr, a["hpf"]["cutoff_hz"], a["hpf"]["order"])
        print(
            f"  high-pass : Butterworth order {a['hpf']['order']}, "
            f"{a['hpf']['cutoff_hz']} Hz, zero-phase"
        )

        # VAD trim: keep from the first to the last speech segment found by the
        # VAD; pauses inside the sentence are kept
        if a["trim"] == "vad":
            v = a["vad"]
            bounds = vad_trim_bounds(y, sr, vad, v["threshold"], v["margin_ms"], v["zero_pad_s"])
            if bounds is None:
                print("  VAD trim  : no speech found -> excluded")
                rows.append({**row, "status": "no_speech"})
                continue
            start, end = bounds
            print(
                f"  VAD trim  : keep {start / sr:.3f}-{end / sr:.3f} s "
                f"(head {start / sr:.3f} s, tail {(len(y) - end) / sr:.3f} s removed)"
            )
            row.update(
                trim_head_s=round(start / sr, 3),
                trim_tail_s=round((len(y) - end) / sr, 3),
            )
            y = y[start:end]

        # Level normalization: measure the P.56 active speech level with actlev,
        # then apply the gain to reach the target level ourselves (as a float)
        level = active_speech_level(y, sr, actlev)
        if level is None:
            print("  P.56 level: no speech activity -> excluded")
            rows.append({**row, "status": "no_activity"})
            continue
        gain_db = a["level"]["target_dbov"] - level
        y = y * 10 ** (gain_db / 20)
        peak = float(np.max(np.abs(y)))
        print(
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
            print("  -> clipping at 16 bit: excluded")
            rows.append({**row, "status": "clipping"})
            continue

        # Save with the same folder layout as the input and record the new
        # fields in the output manifest (the given fields are kept)
        out = copy.deepcopy(entry)
        out.update(
            audio=output_audio(entry, cfg),
            sampling_rate=sr,
            duration=round(len(y) / sr, 3),
        )
        (p["work"] / out["audio"]).parent.mkdir(parents=True, exist_ok=True)
        sf.write(p["work"] / out["audio"], y, sr, subtype="PCM_16")
        rows.append(row)
        entries.append(out)

    fields = [
        "utt_id",
        "status",
        "in_sr",
        "dur_in_s",
        "dur_out_s",
        "trim_head_s",
        "trim_tail_s",
        "act_level_db",
        "gain_db",
        "peak",
    ]
    with open(p["work"] / "audio_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    write_manifest(output_manifest(cfg), entries)
    print(
        f"\nwrote {p['work'] / root}, manifest.jsonl "
        f"({len(entries)} of {len(rows)} utterances) and audio_log.csv"
    )


if __name__ == "__main__":
    main()
