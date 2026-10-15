"""Figures of the linguistic features (guide: docs/01_linguistic_features.md).

Saved in figures/ ("/" in the id is replaced by "_" in the file names):
  structure_<id>.png  the structure the features are built from, on the time axis of
                      the alignment: phrases, words, syllables (stress) and phones
                      (onset / nucleus / coda)
  features_<id>.png   the (N, 239) feature matrix: one column per phone, rows grouped
                      as quin-phone identity, syllable stress, positions and counts;
                      counts are scaled to 0-1 per row
Usage: python src/plot_features.py
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from linguistic import NUMERIC, PHONES, QUIN, annotate, feature_names
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from tutorial_lib import (
    ROOT,
    SIL,
    audio_root,
    file_label,
    load_config,
    load_syllabify,
    phone_word_index,
    read_alignment,
    read_manifest,
    utt_id,
)

# Figures drawn by run.sh (not tracked by git)
OUT = ROOT / "figures"

# Colors (reference palette of the dataviz guide): categorical slots 1-3 for the
# position of a phone in its syllable, an ordinal gray ramp for syllable stress
# (darker = stronger), hatching for pauses
INK, MUTED, SURFACE, NEUTRAL = "#0b0b0b", "#898781", "#fcfcfb", "#f0efec"
ONC_COLOR = {"onset": "#2a78d6", "nucleus": "#eb6834", "coda": "#1baf7a"}
STRESS_COLOR = {"0": "#e1e0d9", "2": "#898781", "1": "#52514e", "none": NEUTRAL}
STRESS_TEXT = {"0": INK, "2": INK, "1": SURFACE, "none": INK}
SIL_HATCH = "#c3c2b7"
# Sequential blue for the feature matrix: 0 = surface, 1 = darkest
MATRIX_CMAP = LinearSegmentedColormap.from_list("blue", [SURFACE, "#cde2fb", "#3987e5", "#104281"])
# Row groups of the 239 dims: (label, size); the first five are the quin-phone blocks
GROUPS = [(q, len(PHONES)) for q in QUIN] + [
    ("stress", 12),
    ("syl pos", 3),
    ("positions", 16),
    ("counts", len(NUMERIC)),
]
N_QUIN = len(QUIN) * len(PHONES)  # 200


def cell(ax, start, end, y, label, face, text=INK, rotate=False, hatch=None):
    """One interval of a tier: a filled box with a 2-pt surface gap and its label."""
    box = (start, y), end - start, 0.9
    if hatch:
        ax.add_patch(Rectangle(*box, facecolor=face, edgecolor=SIL_HATCH, hatch=hatch, lw=0))
        ax.add_patch(Rectangle(*box, fill=False, edgecolor=SURFACE, lw=2))
    else:
        ax.add_patch(Rectangle(*box, facecolor=face, edgecolor=SURFACE, lw=2))
    ax.text(
        (start + end) / 2, y + 0.45, label, ha="center", va="center", fontsize=7,
        color=text, rotation=90 if rotate else 0,
    )  # fmt: skip


def plot_structure(path, phones, times, words, word_idx, syllabify):
    info, syllables, word_phones, phrases, _, _ = annotate(phones, word_idx, syllabify)
    # Tiers from the bottom: phones (y=0), syllables, words, phrases (y=3)
    fig, ax = plt.subplots(figsize=(14, 3.6))
    for i, p in enumerate(phones):
        if p == SIL:  # a pause cuts through every tier
            for y in range(4):
                cell(ax, *times[i], y, "sil", SURFACE, hatch="////")
        else:
            cell(ax, *times[i], 0, p, ONC_COLOR[info[i]["onc"]], rotate=True)
    for syl in syllables:
        s, e = times[syl["phones"][0]][0], times[syl["phones"][-1]][1]
        cell(ax, s, e, 1, syl["stress"], STRESS_COLOR[syl["stress"]], STRESS_TEXT[syl["stress"]])
    for k, idx in word_phones.items():
        cell(ax, times[idx[0]][0], times[idx[-1]][1], 2, words[k][0], NEUTRAL)
    for j, ph in enumerate(phrases):
        cell(ax, times[ph[0]][0], times[ph[-1]][1], 3, f"phrase {j}", NEUTRAL)

    ax.set_xlim(times[0][0], times[-1][1])
    ax.set_ylim(0, 4)
    ax.set_yticks([0.45, 1.45, 2.45, 3.45], ["phones", "syllables\n(stress)", "words", "phrases"])
    ax.set_xlabel("time [s]")
    ax.tick_params(colors=MUTED, labelcolor=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    legend = [Patch(color=c, label=r) for r, c in ONC_COLOR.items()]
    legend += [Patch(color=STRESS_COLOR[s], label=f"stress {s}") for s in ("0", "2", "1")]
    ax.legend(
        handles=legend, ncol=6, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False,
        fontsize=8,
    )  # fmt: skip
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_matrix(path, phones, ling):
    x = ling.T.copy()  # (239, N)
    counts = x[-len(NUMERIC) :]
    peak = counts.max(axis=1, keepdims=True)
    x[-len(NUMERIC) :] = np.divide(counts, peak, out=np.zeros_like(counts), where=peak > 0)

    # Two panels: the 200 quin-phone rows on top, the other 39 rows taller below
    # so that every row can carry its name
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(13, 11), sharex=True, gridspec_kw={"height_ratios": [5, 4]}
    )
    edges = np.cumsum([0] + [n for _, n in GROUPS])
    for ax, lo, hi in ((top, 0, N_QUIN), (bottom, N_QUIN, len(x))):
        im = ax.imshow(
            x[lo:hi], aspect="auto", cmap=MATRIX_CMAP, vmin=0, vmax=1, interpolation="nearest",
            extent=(-0.5, len(phones) - 0.5, hi - 0.5, lo - 0.5),
        )  # fmt: skip
        # Group boundaries and labels on the left
        inside = [(a, b, g, n) for a, b, (g, n) in zip(edges[:-1], edges[1:], GROUPS, strict=True)]
        inside = [(a, b, g, n) for a, b, g, n in inside if lo <= a < hi]
        for a, *_ in inside[1:]:
            ax.axhline(a - 0.5, color=MUTED, lw=0.8)
        centers = [(a + b) / 2 - 0.5 for a, b, *_ in inside]
        ax.set_yticks(centers, [f"{g} ({n})" for *_, g, n in inside])
        ax.tick_params(colors=MUTED, labelcolor=INK)
    # Name of every row in the bottom panel
    right = bottom.secondary_yaxis("right")
    right.set_yticks(np.arange(N_QUIN, len(x)), feature_names()[N_QUIN:], fontsize=6)
    right.tick_params(colors=MUTED, labelcolor=INK)
    bottom.set_xticks(np.arange(len(phones)), phones, rotation=90, fontsize=7)
    bottom.set_xlabel("phone")
    bar = fig.colorbar(im, ax=[top, bottom], fraction=0.02, pad=0.14)
    bar.set_label("value (0/1; counts scaled to 0-1 per row)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = load_config()
    work = cfg["paths"]["manifest"].parent
    syllabify = load_syllabify(cfg)
    entries = read_manifest(cfg["paths"]["manifest"])
    root = audio_root(entries)
    OUT.mkdir(exist_ok=True)
    for entry in entries:
        if not entry.get("linguistic"):
            continue
        name = file_label(utt_id(entry, root))
        phones, times, words = read_alignment(work / entry["textgrid"])
        word_idx = phone_word_index(times, words)
        plot_structure(OUT / f"structure_{name}.png", phones, times, words, word_idx, syllabify)
        ling = np.load(work / entry["linguistic"])["ling"]
        plot_matrix(OUT / f"features_{name}.png", phones, ling)
        print(f"figures/structure_{name}.png, figures/features_{name}.png")


if __name__ == "__main__":
    main()
