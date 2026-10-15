"""Shared helpers for the tutorial steps.

Config, manifest, audio processing, text and log-mel.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from scipy import signal

# Repository root: config.yaml, data/, docs/
ROOT = Path(__file__).resolve().parent.parent
# Full scale of a 16-bit file in [-1, 1) float
INT16_MAX = 32767 / 32768
# silero-vad accepts 8 or 16 kHz input only
VAD_SR = 16000


# ----------------------------------------------------------------- config / corpus


def load_config(path=ROOT / "config.yaml"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("manifest", "work", "third_party"):
        cfg["paths"][key] = (ROOT / cfg["paths"][key]).resolve()
    for key in ("lexicon", "lexicon_extra"):
        if cfg["text"].get(key):  # lexicon_extra is optional
            cfg["text"][key] = (ROOT / cfg["text"][key]).resolve()
    return cfg


def config_value(key):
    """Value of a dotted config key, e.g. "mfa.acoustic_model".

    Used by the bash scripts.
    """
    value = load_config()
    for part in key.split("."):
        value = value[part]
    return value


def read_manifest(path):
    """JSON Lines manifest: one utterance (a JSON object) per line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)


def audio_root(cfg):
    """Common folder of all source audio in the given manifest, e.g. "audio".

    Relative to the manifest's folder. Sub-folders below it are speakers. The
    outputs mirror the input layout, so work / audio_root is the corpus folder
    for MFA.
    """
    entries = read_manifest(cfg["paths"]["manifest"])
    folders = [Path(e["source_audio"]).parent.as_posix() for e in entries]
    return Path(os.path.commonpath(folders))


def utt_id(entry, root):
    """Utterance id: source audio path below the audio root without extension.

    E.g. "0001/example_0001".
    """
    return Path(entry["source_audio"]).with_suffix("").relative_to(root).as_posix()


def output_audio(entry, cfg):
    """Output audio path (relative): the source path with the audio.format suffix."""
    suffix = "." + cfg["audio"]["format"]
    return Path(entry["source_audio"]).with_suffix(suffix).as_posix()


def output_manifest(cfg):
    """Manifest written by the steps: the given fields plus the results.

    Its audio / textgrid paths are relative to its own folder (paths.work).
    """
    return cfg["paths"]["work"] / "manifest.jsonl"


def load_corpus(cfg):
    """{utt_id: (source audio path, raw text)} from the given manifest.

    source_audio is relative to the folder of paths.manifest.
    """
    m, root = cfg["paths"]["manifest"], audio_root(cfg)
    return {
        utt_id(e, root): (m.parent / e["source_audio"], e["text_info"]["text"])
        for e in read_manifest(m)
    }


def file_label(uid):
    """Utterance id as a flat file name: "0001/example_0001" -> "0001_example_0001"."""
    return uid.replace("/", "_")


# ----------------------------------------------------------------- audio (step 1)


def to_target_rate(y, sr, target_sr):
    """Mono float signal at target_sr.

    Channels are averaged; soxr very-high-quality resampling.
    """
    import soxr

    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr == target_sr:
        return y
    return soxr.resample(y, sr, target_sr, quality="VHQ")


def highpass(y, sr, cutoff_hz, order):
    """Zero-phase Butterworth high-pass (effective order 2 * order, -6 dB at cutoff)."""
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    # Forward and backward filtering: no phase delay
    return signal.sosfiltfilt(sos, y)


def vad_trim_bounds(y, sr, model, threshold, margin_ms, zero_pad_s):
    """(start, end) sample indices at sr, from the first to the last speech segment.

    Speech segments are found by silero-vad; margin_ms is added on each side.
    Returns None if no speech is found.
    """
    import torch
    from silero_vad import get_speech_timestamps

    # silero-vad works at 8 or 16 kHz only
    y16 = signal.resample_poly(y, VAD_SR, sr) if sr != VAD_SR else y
    # silero-vad is recurrent and misses speech right at the start of its input:
    # pad zeros first
    z = round(zero_pad_s * VAD_SR)
    x = torch.from_numpy(np.concatenate([np.zeros(z), y16, np.zeros(z)]).astype(np.float32))
    # Speech segments as [{"start": ..., "end": ...}, ...] in samples at VAD_SR
    ts = get_speech_timestamps(x, model, threshold=threshold, sampling_rate=VAD_SR, speech_pad_ms=0)
    if not ts:
        return None
    # Start of the first segment and end of the last one, back at sr, without the
    # zero padding, plus the margin; everything in between (pauses) is kept
    margin = round(sr * margin_ms / 1000)
    start = round((ts[0]["start"] - z) * sr / VAD_SR) - margin
    end = round((ts[-1]["end"] - z) * sr / VAD_SR) + margin
    return max(0, start), min(len(y), end)


def active_speech_level(y, sr, actlev):
    """ITU-T P.56 active speech level [dBov], measured by the ITU-T STL `actlev`.

    Returns None if no speech activity is found.
    """
    # actlev reads 16-bit little-endian raw PCM from a file
    pcm = np.clip(np.round(y * 32768), -32768, 32767).astype("<i2")
    with tempfile.NamedTemporaryFile(suffix=".raw") as f:
        pcm.tofile(f.name)
        out = subprocess.run(
            [str(actlev), "-q", "-sf", str(sr), f.name],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    # The output contains a line such as "ActLev[dB]: -25.481"
    m = re.search(r"ActLev\[dB\]:\s*(-?[\d.]+)", out)
    return float(m.group(1)) if m else None


def load_vad():
    import warnings

    # torch.jit.load inside silero-vad
    warnings.filterwarnings("ignore", category=FutureWarning)
    from silero_vad import load_silero_vad

    return load_silero_vad()


# ----------------------------------------------------------------- text (step 2)


def english_cleaners_module(cfg):
    """keithito/tacotron cleaners as shipped in ming024/FastSpeech2 (MIT)."""
    sys.path.insert(0, str(cfg["paths"]["third_party"] / "FastSpeech2"))
    from text import cleaners

    return cleaners


def mfa_safe(text):
    """Remove punctuation that MFA would keep attached to words.

    Removes (), [], double quotes, and single quotes around words (apostrophes
    inside words such as you'll are kept).
    """
    # Brackets and double quotes
    text = re.sub(r'[()\[\]"]', " ", text)
    # Single quotes not inside a word, e.g. 'hello' -> hello, but you'll stays
    text = re.sub(r"(?<![\w])'|'(?![\w])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_lexicon(path):
    """{word: [pronunciation, ...]} from "word<TAB>phones" lines.

    A word with several pronunciations has several lines.
    """
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            word, pron = line.split(maxsplit=1)
            entries.setdefault(word.lower(), []).append(pron.strip())
    return entries


# ----------------------------------------------------------------- figures

# Figures drawn by run.sh (not tracked by git; docs/figures keeps the copies
# used in the guides)
FIGURES = ROOT / "figures"
# Fixed log-mel color range for all figures; floor ln(1e-5) = -11.5
MEL_RANGE = (-11.5, 2.0)


def log_mel(y, sr, m):
    """(T, num_mels) natural-log magnitude mel spectrogram, T = len(y) // hop.

    Frame i is centered at (i + 0.5) * hop: reflect padding of (n_fft - hop) / 2,
    no centering; slaney mel, floor 1e-5.
    """
    import librosa

    pad = (m["n_fft"] - m["hop_size"]) // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    spec = np.abs(
        librosa.stft(
            yp,
            n_fft=m["n_fft"],
            hop_length=m["hop_size"],
            win_length=m["win_size"],
            window="hann",
            center=False,
        )
    )
    basis = librosa.filters.mel(
        sr=sr, n_fft=m["n_fft"], n_mels=m["num_mels"], fmin=m["fmin"], fmax=m["fmax"]
    )
    return np.log(np.maximum(basis @ spec, 1e-5)).T


# python src/tutorial_lib.py <dotted.key>  -> prints a config value
if __name__ == "__main__":
    # Output audio folder = MFA corpus (used by align.sh)
    if sys.argv[1] == "corpus_dir":
        cfg = load_config()
        print(cfg["paths"]["work"] / audio_root(cfg))
    else:
        print(config_value(sys.argv[1]))
