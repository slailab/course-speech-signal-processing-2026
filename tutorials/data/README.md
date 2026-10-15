# Data

Data shared by all tutorials of the course. A dataset is stored (or downloaded) only once, and a tutorial can use the outputs of another.

```
tutorials/data/
├── datasets/                   scripts per dataset (tracked by git)
│   └── ljspeech/
│       ├── download.sh         download, extract and write the manifest
│       ├── make_manifest.py    original metadata → manifest.jsonl
│       └── lexicon_extra.txt   pronunciation fixes for OOV words that G2P gets wrong
├── raw/                        given data, never modified (only the example is tracked by git)
│   ├── example_dataset_16k/    two example utterances
│   └── LJSpeech-1.1/           downloaded: the original layout as extracted + manifest.jsonl
└── preprocess/                 outputs of the tutorials (not tracked by git)
```

## Datasets

| Dataset | Get it | Content | License |
|---|---|---|---|
| `example_dataset_16k` | included | 2 utterances from LibriSpeech test-clean, 2 speakers, 16 kHz | CC BY 4.0 |
| LJSpeech 1.1 | `bash datasets/ljspeech/download.sh` | 13,100 utterances of one speaker (about 24 hours), 22.05 kHz, about 2.6 GB | public domain |

## Manifest

Each dataset folder has a `manifest.jsonl` at its root: one utterance per line (JSON). The tutorials read only the manifest, never the original metadata, so every dataset looks the same to them.

```json
{"audio": null, "source_audio": "wavs/LJ002-0026.wav", "sampling_rate": null, "sampling_rate_org": 22050, "duration": null,
 "text_info": {"text": "in the three years between eighteen thirteen and eighteen sixteen,",
               "text_org": "in the three years between 1813 and 1816,",
               "normalized_text": null, "phonemes": null},
 "speaker_info": {"speaker_id": "LJ", "gender": "female"},
 "textgrid": null, "language": "en", "dataset": "LJSpeech"}
```

| Field | Meaning |
|---|---|
| `source_audio` | audio file, relative to the manifest's folder (inside the original layout) |
| `sampling_rate_org` | sampling rate of the original audio |
| `text_info.text` | raw text, the input of text normalization (for LJSpeech: its normalized column, with numbers written out) |
| `text_info.text_org` | transcription as given by the dataset |
| `speaker_info` | speaker id and gender |
| `language`, `dataset` | language code and dataset name |
| `audio`, `sampling_rate`, `duration`, `text_info.normalized_text`, `text_info.phonemes`, `textgrid` | `null`; filled by the preprocess tutorial in its output manifest |

- The utterance id is the audio path below the common audio folder, without the extension (e.g. `LJ001-0001`, or `0001/example_0001` with speaker folders).
- Sub-folders of the audio folder are speakers: the aligner (MFA) tells speakers apart by folder.

## Adding a dataset

1. Add `datasets/<name>/download.sh`: download into `raw/` and keep the original layout as extracted.
2. Add `datasets/<name>/make_manifest.py`: write `manifest.jsonl` at the dataset root with the fields above; `source_audio` points into the original layout.
3. Add a row to the table above (content, size, license).
4. Point a tutorial's `config.yaml` (`paths.manifest`) to the new manifest.
