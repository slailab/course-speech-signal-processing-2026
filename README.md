# Special Topics in Speech Signal Processing (2026)

Tutorials and lecture materials for the course *Special Topics in Speech Signal Processing* at Yonsei University, Fall 2026.

## Tutorials

Each tutorial has its own README, setup script and requirements, and shares no code with the others. Data is kept in `tutorials/data/`, shared by all tutorials, so that a dataset is stored (or downloaded) only once and a tutorial can use the outputs of another:

```
tutorials/
├── data/             see tutorials/data/README.md (datasets, manifest format)
│   ├── datasets/     download and manifest scripts per dataset
│   ├── raw/          given data (raw/example_dataset_16k/ is included; others are downloaded)
│   └── preprocess/   outputs of the tutorials (not tracked by git)
├── preprocess/
└── spss/
```

| Tutorial | Content |
|---|---|
| [preprocess](tutorials/preprocess/) | Preparing TTS training data: audio preprocessing, text processing, forced alignment |
| [spss](tutorials/spss/) | Statistical parametric speech synthesis with DNNs (work in progress) |

More tutorials will be added during the semester.

## Lecture materials

The lecture slides (PDF) will be published in `lectures/` at the end of the semester.

## Getting a tutorial

To download only the tutorial you need (here: `preprocess`) together with the shared data folder:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/slailab/course-speech-signal-processing-2026.git course
cd course
git sparse-checkout set tutorials/preprocess tutorials/data
cd tutorials/preprocess
```

Then follow the README of the tutorial. On Google Colab, run the same commands in a cell, prefixing each line with `!` (use `%cd` instead of `cd`).

To download everything instead: `git clone --depth 1 https://github.com/slailab/course-speech-signal-processing-2026.git`.

## Code style

Only needed if you change the code. The Python code of all tutorials is formatted and checked with [ruff](https://docs.astral.sh/ruff/) (black-compatible formatter, line length 100; settings in `pyproject.toml`), and [pre-commit](https://pre-commit.com/) runs these checks on every commit (`.pre-commit-config.yaml`). Run the commands from the repository root:

```bash
pip install -r requirements-dev.txt   # ruff, pre-commit (pinned versions)
pre-commit install                    # once per clone: run the checks on every commit

ruff format .                         # format
ruff check .                          # lint (add --fix to fix what can be fixed automatically)
pre-commit run --all-files            # all checks on all files in the repository
```

## License

- Code: [MIT](LICENSE)
- Documents and lecture materials: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Example data: see the README of each tutorial (e.g. LibriSpeech, CC BY 4.0)
