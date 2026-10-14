"""Figures for guide 01 (docs/01_audio_preprocessing.md), saved in figures/.

"/" in the id is replaced by "_" in the file names.

  <id>_0_input.png, <id>_1_hpf.png, <id>_2_trim.png, <id>_3_level.png
      Waveform + log-mel after each step. Shared axes: time 0..input duration
      (trimmed signals stay at their original time, removed parts shaded),
      amplitude -1..1, 80 mel bins and a fixed log-mel color range.
      No title / colorbar.
  vad_<id>.png
      silero-vad speech probability, threshold and trimmed parts.

Usage: python src/plot_audio.py
"""

import librosa
import matplotlib
import numpy as np
import soundfile as sf
import torch
from scipy import signal

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tutorial_lib import (
    FIGURES as OUT,
    MEL_RANGE,
    VAD_SR,
    active_speech_level,
    file_label,
    highpass,
    load_config,
    load_corpus,
    load_vad,
    log_mel,
    to_target_rate,
    vad_trim_bounds,
)


def audio_steps(y, sr, cfg, vad):
    """Yield (key, signal, start offset in samples) after each preprocessing step."""
    a = cfg["audio"]
    yield "0_input", y, 0
    y = highpass(y, sr, a["hpf"]["cutoff_hz"], a["hpf"]["order"])
    yield "1_hpf", y, 0
    start = 0
    if a["trim"] == "vad":
        v = a["vad"]
        start, end = vad_trim_bounds(y, sr, vad, v["threshold"], v["margin_ms"], v["zero_pad_s"])
        y = y[start:end]
        yield "2_trim", y, start
    level = active_speech_level(y, sr, cfg["paths"]["third_party"] / "bin" / "actlev")
    yield "3_level", y * 10 ** ((a["level"]["target_dbov"] - level) / 20), start


def plot_step(path, y, offset, n_in, sr, m):
    hop, t0 = m["hop_size"], offset / sr
    mel = log_mel(y, sr, m)
    fig, (ax_w, ax_m) = plt.subplots(
        2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]}
    )
    ax_w.plot(t0 + np.arange(len(y)) / sr, y, lw=0.4, color="tab:blue")
    ax_w.set_ylim(-1, 1)
    ax_w.set_ylabel("amplitude")
    ax_m.imshow(
        mel.T,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=MEL_RANGE[0],
        vmax=MEL_RANGE[1],
        extent=[t0, t0 + len(mel) * hop / sr, -0.5, m["num_mels"] - 0.5],
    )
    # librosa.filters.mel uses n_mels + 2 band edges; the inner ones are the
    # center frequencies of the mel filters
    edges = librosa.mel_frequencies(m["num_mels"] + 2, fmin=m["fmin"], fmax=m["fmax"] or sr / 2)
    freqs = edges[1:-1]
    ticks = [f for f in (250, 500, 1000, 2000, 4000, 8000) if f <= freqs[-1]]
    ax_m.set_yticks(
        [int(np.argmin(abs(freqs - f))) for f in ticks],
        [f"{f / 1000:g}k" for f in ticks],
    )
    ax_m.set_ylim(-0.5, m["num_mels"] - 0.5)
    ax_m.set_ylabel(f"mel ({m['num_mels']} bins), Hz")
    ax_m.set_xlabel("time [s]")
    ax_m.set_xlim(0, n_in / sr)
    if offset or len(y) < n_in:  # shade the parts removed by trimming
        for ax in (ax_w, ax_m):
            for s, e in ((0, t0), (t0 + len(y) / sr, n_in / sr)):
                if e > s:
                    ax.axvspan(
                        s,
                        e,
                        color="0.85",
                        zorder=0 if ax is ax_w else 3,
                        alpha=1 if ax is ax_w else 0.6,
                    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def vad_figure(cfg, vad, utt_id, y, sr):
    a, v = cfg["audio"], cfg["audio"]["vad"]
    y = highpass(y, sr, a["hpf"]["cutoff_hz"], a["hpf"]["order"])
    y16 = signal.resample_poly(y, VAD_SR, sr) if sr != VAD_SR else y
    # padding in whole 32 ms chunks for plotting
    z = round(v["zero_pad_s"] * VAD_SR / 512) * 512
    x = torch.from_numpy(np.concatenate([np.zeros(z), y16, np.zeros(z)]).astype(np.float32))
    vad.reset_states()
    p = np.array([vad(x[i : i + 512], VAD_SR).item() for i in range(0, len(x) - 511, 512)])
    p = p[z // 512 :][: len(y16) // 512]
    t = (np.arange(len(p)) + 0.5) * 512 / VAD_SR
    start, end = vad_trim_bounds(y, sr, vad, v["threshold"], v["margin_ms"], v["zero_pad_s"])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    a1.plot(np.arange(len(y)) / sr, y, lw=0.4)
    a1.set_ylim(-1, 1)
    a1.set_ylabel("amplitude")
    a2.step(t, p, where="mid")
    a2.axhline(v["threshold"], color="tab:red", lw=0.8)
    a2.text(
        t[-1],
        v["threshold"] + 0.03,
        f"threshold {v['threshold']}",
        color="tab:red",
        fontsize=9,
        ha="right",
    )
    a2.set_ylim(0, 1.05)
    a2.set_ylabel("speech prob.")
    a2.set_xlabel("time [s]")
    a2.set_xlim(0, len(y) / sr)
    for ax in (a1, a2):
        ax.axvspan(0, start / sr, color="0.85", zorder=0)
        ax.axvspan(end / sr, len(y) / sr, color="0.85", zorder=0)
    fig.savefig(OUT / f"vad_{utt_id}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"vad_{utt_id}.png  keep {start / sr:.3f}-{end / sr:.3f} s of {len(y) / sr:.3f} s")


def main():
    cfg = load_config()
    OUT.mkdir(exist_ok=True)
    sr = cfg["audio"]["target_sr"]
    vad = load_vad() if cfg["audio"]["trim"] == "vad" else None
    for utt_id, (path, _) in load_corpus(cfg).items():
        name = file_label(utt_id)
        y, in_sr = sf.read(path, dtype="float64")
        y = to_target_rate(y, in_sr, sr)
        for key, sig, offset in audio_steps(y, sr, cfg, vad):
            plot_step(OUT / f"{name}_{key}.png", sig, offset, len(y), sr, cfg["mel"])
            print(f"{name}_{key}.png")
        if vad is not None:
            vad_figure(cfg, vad, name, y, sr)


if __name__ == "__main__":
    main()
