"""Figure for guide 03 (docs/03_alignment.md), saved as figures/align_<id>.png.

Waveform + log-mel of the aligned (preprocessed) audio with the MFA word and
phone intervals underneath. "/" in the id is replaced by "_" in the file name.
Same axes style as plot_audio.py: amplitude -1..1, 80 mel bins, fixed log-mel
color range.

Usage: python src/plot_alignment.py
"""

import librosa
import matplotlib
import numpy as np
import soundfile as sf
import tgt

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tutorial_lib import (
    FIGURES as OUT,
    MEL_RANGE,
    audio_root,
    file_label,
    load_config,
    log_mel,
    output_manifest,
    read_manifest,
    utt_id,
)


def draw_tier(ax, intervals, label, stagger=False):
    """Interval labels between boundary lines; silences shaded.

    stagger: alternate the label height so that labels of short neighbouring
    intervals do not overlap.
    """
    for k, iv in enumerate(intervals):
        if iv.text:
            y = (0.3 if k % 2 else 0.7) if stagger else 0.5
            ax.text(
                (iv.start_time + iv.end_time) / 2,
                y,
                iv.text,
                ha="center",
                va="center",
                fontsize=8,
            )
        else:  # silence
            ax.axvspan(iv.start_time, iv.end_time, color="0.85")
        ax.axvline(iv.start_time, color="0.4", lw=0.6)
    ax.axvline(intervals[-1].end_time, color="0.4", lw=0.6)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center")


def main():
    cfg = load_config()
    if not cfg["report"]["figures"]:
        print(f"report.figures is off in config {cfg['name']}: no figures")
        return
    work, m, root = cfg["paths"]["work"], cfg["mel"], audio_root(cfg)
    OUT.mkdir(exist_ok=True)
    for entry in read_manifest(output_manifest(cfg)):
        if not entry.get("textgrid"):
            continue
        name = file_label(utt_id(entry, root))
        y, sr = sf.read(work / entry["audio"])
        tg = tgt.io.read_textgrid(str(work / entry["textgrid"]), include_empty_intervals=True)
        mel = log_mel(y, sr, m)
        dur = len(y) / sr

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(14, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1.4, 0.35, 0.5]},
        )
        ax_w, ax_m, ax_word, ax_ph = axes
        ax_w.plot(np.arange(len(y)) / sr, y, lw=0.4, color="tab:blue")
        ax_w.set_ylim(-1, 1)
        ax_w.set_ylabel("amplitude")
        ax_m.imshow(
            mel.T,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=MEL_RANGE[0],
            vmax=MEL_RANGE[1],
            extent=[0, len(mel) * m["hop_size"] / sr, -0.5, m["num_mels"] - 0.5],
        )
        # Center frequencies of the mel filters (inner band edges)
        edges = librosa.mel_frequencies(m["num_mels"] + 2, fmin=m["fmin"], fmax=m["fmax"] or sr / 2)
        freqs = edges[1:-1]
        ticks = [f for f in (250, 500, 1000, 2000, 4000, 8000) if f <= freqs[-1]]
        ax_m.set_yticks(
            [int(np.argmin(abs(freqs - f))) for f in ticks],
            [f"{f / 1000:g}k" for f in ticks],
        )
        ax_m.set_ylabel("mel, Hz")
        words = tg.get_tier_by_name("words").intervals
        phones = tg.get_tier_by_name("phones").intervals
        for ax in (ax_w, ax_m):  # word boundaries on the signal panels
            for iv in words:
                ax.axvline(
                    iv.start_time,
                    color="white" if ax is ax_m else "0.5",
                    lw=0.6,
                    ls="--",
                )
        draw_tier(ax_word, words, "words")
        draw_tier(ax_ph, phones, "phones", stagger=True)
        ax_ph.set_xlabel("time [s]")
        ax_ph.set_xlim(0, dur)
        fig.savefig(OUT / f"align_{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"figures/align_{name}.png")


if __name__ == "__main__":
    main()
