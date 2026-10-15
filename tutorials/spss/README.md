# Statistical Parametric Speech Synthesis (SPSS) Tutorial

*Work in progress: more steps (duration and acoustic models, synthesis, evaluation) will be added.*

Build a DNN-based statistical parametric speech synthesis system step by step, starting from the outputs of the [preprocess](../preprocess/) tutorial.

| Step | Content | Document | Run |
|---|---|---|---|
| 1 | Linguistic features: phones, syllables, words and phrases → 239 dims per phone | [01_linguistic_features.md](docs/01_linguistic_features.md) | `python src/extract_linguistic.py` |

## Data

The input is the output of the preprocess tutorial in `tutorials/data/preprocess/`: its output manifest, with the alignment (`textgrid`) of each utterance. The path is set in `config.yaml` (`paths.manifest`). Run the preprocess tutorial first.

The outputs are written next to that manifest, with the same folder layout as the audio, and recorded in it:

| Step | Output | Field added to the manifest |
|---|---|---|
| 1 | `linguistic/<id>.npz` (`phones` (N,), `ling` (N, 239)), `linguistic/feature_names.txt` | `linguistic` |

## Environment

Runs on Linux: **Google Colab**, WSL (Windows), Ubuntu, etc. Requires git.

Tested with **Python 3.11** (recommended). Other versions may work, but the pinned package versions in `requirements.txt` may not install on them.

## Installation (once)

Download this tutorial, the preprocess tutorial and the shared data folder from the course repository (see the [course README](../../README.md) for other options):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/slailab/course-speech-signal-processing-2026.git course
cd course
git sparse-checkout set tutorials/preprocess tutorials/spss tutorials/data
cd tutorials/spss
bash setup.sh
```

All commands below are run from this folder (`tutorials/spss`).

What `setup.sh` does:

| Step | Item | Location |
|---|---|---|
| 1 | Python packages (`requirements.txt`) | current Python environment (set with `PYTHON=...`; installed with pip or uv) |
| 2 | syllabify (Kyle Gorman, MIT): syllabification of ARPAbet words | `third_party/syllabify/` (pinned commit) |

To install only some steps, give the first and last step, e.g. `bash setup.sh 2`.

## Running

All at once:

```bash
bash run.sh          # 1 linguistic features → 2 feature table → 3 figures
bash run.sh 3        # figures only
```

One script at a time:

```bash
python src/extract_linguistic.py   # 1. linguistic features
python src/show_features.py        # 2. feature table, one row per phone
python src/plot_features.py        # 3. figures: structure on the time axis, feature matrix
```

## Outputs

```
../data/preprocess/example_dataset_16k/
├── manifest.jsonl                         + field "linguistic"
└── linguistic/
    ├── feature_names.txt                  the 239 dimension names
    ├── 0001/example_0001.npz              phones (N,), ling (N, 239)
    └── 0002/...
figures/                                   figures drawn by run.sh (file names use the id with "/" replaced by "_")
```

`figures/` is not tracked by git; the figures shown in the guides are kept in `docs/figures/`.

## Code style

Only needed if you change the code: see "Code style" in the [course README](../../README.md).

## Files

```
├── README.md
├── config.yaml          paths
├── requirements.txt     Python packages for the tutorial
├── setup.sh             installation
├── run.sh               run everything
├── docs/                tutorial documents, figures/
├── src/
│   ├── tutorial_lib.py         shared functions (config, manifest, reading the alignment)
│   ├── linguistic.py           the 239 linguistic features
│   ├── extract_linguistic.py   extracts the features of every utterance
│   └── show_features.py, plot_features.py   feature table, figures
└── third_party/         external code fetched by setup.sh
```
