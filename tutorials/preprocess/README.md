# TTS Data Preprocessing Tutorial

Run, step by step, the process that turns example speech samples into TTS training data.

| Step | Content | Document | Run |
|---|---|---|---|
| 1 | Audio preprocessing: DC removal (HPF), leading/trailing silence removal (VAD), level normalization (P.56) | [01_audio_preprocessing.md](docs/01_audio_preprocessing.md) | `python src/step1_audio.py` |
| 2 | Text processing: text normalization (TN), pronunciation lexicon, OOV | [02_text_processing.md](docs/02_text_processing.md) | `python src/step2_text.py` |
| 3 | Forced alignment: MFA, TextGrid | [03_alignment.md](docs/03_alignment.md) | `bash src/step3_align.sh` |

## Data

Data is kept in `tutorials/data/`, shared by all tutorials of the course: `raw/` holds the given data and `preprocess/` the outputs. The paths are set in `config.yaml` (`paths.manifest`, `paths.work`).

```
../data/raw/example_dataset_16k/
├── manifest.jsonl                  list of utterances: one utterance per line (JSON)
└── audio/
    ├── 0001/example_0001.flac      folder = speaker
    └── 0002/example_0001.flac
```

One line of `manifest.jsonl` (wrapped for readability):

```json
{"audio": null, "source_audio": "audio/0001/example_0001.flac", "sampling_rate": null, "sampling_rate_org": 16000, "duration": null,
 "text_info": {"text": "Said Mr. Irwine with his stately cordiality",
               "text_org": "SAID MISTER IRWINE WITH HIS STATELY CORDIALITY",
               "normalized_text": null, "phonemes": null},
 "speaker_info": {"speaker_id": "0001", "gender": "female"},
 "textgrid": null, "language": "en", "dataset": "LibriSpeech"}
```

- The code reads two fields: `source_audio` (path relative to the manifest's folder) and `text_info.text` (raw text).
- Sub-folders of the audio folder (`audio/`) are speakers. The aligner (MFA) tells speakers apart by folder.
- The utterance id is the path below the audio folder without the extension (`0001/example_0001`). File names only need to be unique within a speaker folder.
- Outputs are written to `../data/preprocess/example_dataset_16k/` with the same folder layout as the input.
- Fields that are `null` are filled by preprocessing. Each step adds the following values to the output manifest (`../data/preprocess/example_dataset_16k/manifest.jsonl`). Paths there are relative to the output manifest's folder.

| Step | Fields filled |
|---|---|
| 1 | `audio` (preprocessed audio, format set by `audio.format` in `config.yaml`), `sampling_rate`, `duration` |
| 2 | `text_info.normalized_text` |
| 3 | `textgrid`, `text_info.phonemes` (aligned phone sequence, words separated by `\|`, silence as `sil`) |

**Source:** the example speech is two utterances from [LibriSpeech](https://www.openslr.org/12) test-clean (CC BY 4.0): `0001/example_0001` = `2094-142345-0044`, `0002/example_0001` = `4970-29093-0000`. LibriSpeech transcripts are normalized upper-case text (`text_org`), so the written form (`text`) was restored to demonstrate text normalization.

## Environment

Runs on Linux: **Google Colab**, WSL (Windows), Ubuntu, etc. Requires git, gcc and curl (all available on Colab).

Tested with **Python 3.11** (recommended). Other versions may work, but the pinned package versions in `requirements.txt` may not install on them.

## Installation (once)

Download only this tutorial from the course repository, then run `setup.sh` (see the [course README](../../README.md) for other options):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/slailab/course-speech-signal-processing-2026.git course
cd course
git sparse-checkout set tutorials/preprocess tutorials/data
cd tutorials/preprocess
bash setup.sh
```

All commands below are run from this folder (`tutorials/preprocess`).

What `setup.sh` does (about 5-10 minutes, mostly the MFA installation):

| Step | Item | Location |
|---|---|---|
| 1 | Python packages (`requirements.txt`) | current Python environment (set with `PYTHON=...`; installed with pip or uv) |
| 2 | FastSpeech2: text normalization (`english_cleaners`), pronunciation lexicon (librispeech-lexicon) | `third_party/FastSpeech2/` |
| 3 | ITU-T STL: P.56 level meter `actlev` (built with gcc) | `third_party/bin/actlev` |
| 4 | Montreal Forced Aligner 3.4.3 + English models | micromamba environment `mfa` (`~/.local/bin/micromamba`, `~/micromamba`) |

To install only some steps, give the first and last step. Items already installed are skipped.

```bash
bash setup.sh 3      # step 3 to the end
bash setup.sh 2 3    # steps 2-3 only
```

## Running

All at once:

```bash
bash run.sh          # 1 audio → 2 text → 3 alignment → 4 figures
bash run.sh 3        # step 3 to the end
bash run.sh 4 4      # figures only
```

One script at a time:

```bash
python src/step1_audio.py        # 1. audio preprocessing
python src/step2_text.py         # 2. text processing
bash src/step3_align.sh          # 3. forced alignment (MFA)
python src/show_alignment.py     #    alignment as a table
python src/plot_audio.py         # figures: waveform and mel after each audio step, VAD probability
python src/plot_alignment.py     # figures: word and phone intervals over the waveform and mel
```

Parameters (data paths, sampling rate, filter, VAD, target level, lexicon, MFA models) are set in `config.yaml`.

**On Colab:** prefix commands with `!` in a cell (use `%cd` instead of `cd`), and listen to the audio like this:

```python
from IPython.display import Audio

Audio("../data/preprocess/example_dataset_16k/audio/0001/example_0001.flac")
```

## Outputs

Written with the same folder layout as the input.

```
../data/preprocess/example_dataset_16k/
├── manifest.jsonl                       output manifest (filled by steps 1-3)
├── audio_log.csv                        step 1 log (trimmed parts, level, gain, reason for exclusion)
├── audio/0001/example_0001.flac         step 1 output (16 kHz, -26 dBov)
├── audio/0001/example_0001.lab          step 2 output (normalized transcript)
├── audio/0002/...
├── textgrids/0001/example_0001.TextGrid step 3 output (+ alignment_analysis.csv)
├── textgrids/0002/...
└── mfa/
    ├── lexicon.txt                      pronunciation lexicon
    ├── oovs/, oov.dict                  OOV list and G2P pronunciations
    ├── dictionary.txt                   lexicon used for alignment
    └── *.log                            MFA logs
docs/figures/                            figures (file names use the id with "/" replaced by "_": 0001_example_0001_*.png)
```

## Code style

Only needed if you change the code: see "Code style" in the [course README](../../README.md).

## Files

```
├── README.md
├── config.yaml          paths and parameters
├── requirements.txt     Python packages for the tutorial
├── setup.sh             installation
├── run.sh               run everything
├── docs/                tutorial documents 01-03, figures/
├── src/
│   ├── tutorial_lib.py  shared functions (resampling, HPF, VAD trim, P.56, text cleanup, log-mel)
│   ├── step1_audio.py, step2_text.py, step3_align.sh      steps
│   ├── step3_manifest.py                                  writes the alignment into the manifest (called by step3_align.sh)
│   └── show_alignment.py, plot_audio.py, plot_alignment.py viewing results, figures
└── third_party/         external tools fetched by setup.sh
../data/                 data shared by all tutorials (raw/: example data, preprocess/: outputs)
```
