# Tutorial 2. Text Processing

A TTS model does not read text directly; it reads a **phone sequence**. In this tutorial we turn raw text into phones in five steps:

1. **Text normalization** (TN)
2. **Word splitting**
3. **Pronunciation lexicon lookup**
4. **G2P (grapheme-to-phoneme) for out-of-vocabulary words**
5. **Choosing pronunciations and inserting pauses** (done by forced alignment, covered in detail in Tutorial 3)


## 0. Reading the raw text

We read the raw text (transcript) from the dataset's manifest (`text_info.text`) through `tutorial_lib.load_corpus`.

```python
from tutorial_lib import load_config, load_corpus

cfg = load_config()  # config.yaml
for utt_id, (audio_path, text) in load_corpus(cfg).items():
    print(utt_id, text)
```

```
0001/example_0001 Said Mr. Irwine with his stately cordiality
0002/example_0001 You'll never dig it out of the Astor library
```

## 1. Text normalization (TN)

### What and why

Text normalization rewrites numbers, abbreviations and symbols as **the words that are actually spoken**. We need it because the pronunciation lexicon has no entries such as "Mr." or "1880", only words such as "mister" and "eighteen eighty".

### Code

We use [`english_cleaners`](https://github.com/keithito/tacotron/blob/master/text/cleaners.py) (keithito/tacotron, MIT license), the rule-based normalizer used by many open-source TTS projects such as Tacotron 2 and FastSpeech 2. `setup.sh` fetches the copy that ships with [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2). It applies five sub-steps in order:

```python
import sys

sys.path.insert(0, str(cfg["paths"]["third_party"] / "FastSpeech2"))
from text import cleaners

t = text
for step in (
    cleaners.convert_to_ascii,  # unidecode: café -> cafe
    cleaners.lowercase,  # lower case
    cleaners.expand_numbers,  # inflect: $5 -> five dollars, 1880 -> eighteen eighty
    cleaners.expand_abbreviations,  # mr. -> mister, dr. -> doctor
    cleaners.collapse_whitespace,
):  # collapse repeated whitespace
    t = step(t)
```

### Result

`0001/example_0001`:

| Sub-step | Result |
|---|---|
| Raw text | Said Mr. Irwine with his stately cordiality |
| `convert_to_ascii` | Said Mr. Irwine with his stately cordiality (no change) |
| `lowercase` | said mr. irwine with his stately cordiality |
| `expand_numbers` | said mr. irwine with his stately cordiality (no change) |
| `expand_abbreviations` | said **mister** irwine with his stately cordiality |
| `collapse_whitespace` | said mister irwine with his stately cordiality (no change) |

`0002/example_0001`:

| Sub-step | Result |
|---|---|
| Raw text | You'll never dig it out of the Astor library |
| `convert_to_ascii` | You'll never dig it out of the Astor library (no change) |
| `lowercase` | you'll never dig it out of the astor library |
| `expand_numbers` | you'll never dig it out of the astor library (no change) |
| `expand_abbreviations` | you'll never dig it out of the astor library (no change) |
| `collapse_whitespace` | you'll never dig it out of the astor library (no change) |

## 2. Word splitting

### What and why

To look words up in the lexicon, we first split the normalized text into words. The text is split on whitespace and punctuation is dropped, while apostrophes stay part of the word (`you'll`). The aligner (MFA) does the actual splitting internally.

One thing we must handle ourselves: `english_cleaners` leaves brackets and quotes in place. If they stick to a word, MFA sees tokens such as `(which` and treats them as unknown words. So we remove them before passing the text to MFA.

### Code

```python
from tutorial_lib import mfa_safe

t = mfa_safe(t)  # remove (), [], double quotes, single quotes around words
lab_path = cfg["paths"]["work"] / "audio" / f"{utt_id}.lab"  # next to the audio file
lab_path.write_text(t + "\n")  # input of the aligner (MFA)
```

### Result

```
0001/example_0001 said | mister | irwine | with | his | stately | cordiality
0002/example_0001 you'll | never | dig | it | out | of | the | astor | library
```

## 3. Pronunciation lexicon lookup

### What and why

Next, we look up the phone sequence of each word in a pronunciation lexicon. We use **librispeech-lexicon** (about 200k words), which is written in **ARPAbet** (see Appendices A and C).

- Vowels carry a stress digit: 0 = unstressed, 1 = primary stress, 2 = secondary stress.
- **A word can have more than one pronunciation.** The lexicon only lists the candidates; which one the speaker actually used can only be told from the audio (Section 5).

### Code

```python
from tutorial_lib import read_lexicon

lexicon = read_lexicon(cfg["text"]["lexicon"])  # {word: [pronunciation, ...]}
for word in t.split():
    print(word, lexicon.get(word, "OOV"))
```

### Result

`0001/example_0001`:

| Word | Pronunciation candidates in the lexicon |
|---|---|
| said | `S EH1 D` |
| mister | `M IH1 S T ER0` |
| irwine | **not in the lexicon (OOV)** |
| with | `W IH0 DH` / `W IH0 TH` / `W IH1 DH` / `W IH1 TH` |
| his | `HH IH0 Z` / `HH IH1 Z` |
| stately | `S T EY1 T L IY0` |
| cordiality | `K AO1 R D IY0 AE1 L AH0 T IY0` |

`0002/example_0001`:

| Word | Pronunciation candidates in the lexicon |
|---|---|
| you'll | `Y UW1 L` |
| never | `N EH1 V ER0` |
| dig | `D IH1 G` |
| it | `IH0 T` / `IH1 T` |
| out | `AW1 T` |
| of | `AH0 V` / `AH1 V` |
| the | `DH AH0` / `DH AH1` / `DH IY0` |
| astor | `AE1 S T ER0` |
| library | `L AY1 B R EH2 R IY0` |

## 4. G2P for out-of-vocabulary words

### What and why

Words that are not in the lexicon are called OOV (out-of-vocabulary) words. For these, a **G2P model** predicts the pronunciation from the spelling. In our examples, the name `irwine` is an OOV word. If we left it as is, the aligner would treat the word as an unknown sound (`spn`).

### Code

We use the MFA G2P model (`mfa.g2p_model` in `config.yaml`; see Appendices B and D).

```bash
# CORPUS: folder of audio and .lab files, LEXICON: pronunciation lexicon, G2P_MODEL: mfa.g2p_model in config.yaml
mfa find_oovs "$CORPUS" "$LEXICON" "$OOV_DIR"                                 # find OOV words
mfa g2p "$OOV_DIR/oovs_found_lexicon.txt" "$G2P_MODEL" "$OOV_DICT"            # generate OOV pronunciations
cat "$LEXICON" "$OOV_DICT" > "$DICTIONARY"                                    # merge into the lexicon
```

### Result

The only OOV word is `irwine` in `0001/example_0001`; `0002/example_0001` has none.

```
irwine  ER1 W AH0 N
irwine  ER1 W AY2 N
```

Like the lexicon, G2P can return more than one candidate. When the spelling alone is not conclusive, it lists several candidates in order of probability.

**We do not pick a candidate at this point.** All candidates go into the lexicon, and the aligner chooses one by listening to the audio (Section 5).

## 5. Choosing pronunciations and inserting pauses

### What and why

There are two things the text alone cannot tell us:

- which pronunciation the speaker used for a word with several candidates
- where the speaker paused

**Forced alignment** answers both questions by looking at the audio. We run it in Tutorial 3; here we look at how it makes the choice:

- For each word, the aligner expands all of its pronunciation candidates from the lexicon, and it allows an optional silence between words. Together these form a graph of possible paths.
- It then picks the path with the **highest acoustic model score**, that is, the pronunciations and silence positions that best match the actual sound (Viterbi search).
- Our lexicon has no pronunciation probabilities, so all candidates are equally likely and the choice is effectively made by the acoustic score alone.
- The choice is made separately for each utterance, so the same word can end up with different pronunciations in different utterances.

### Result

For the words with several candidates, the aligner chose:

`0001/example_0001`:

| Word | Candidates | Pronunciation chosen by the aligner |
|---|---|---|
| irwine | 2 (G2P) | `ER1 W AH0 N` |
| with | 4 | `W IH0 TH` |
| his | 2 | `HH IH1 Z` |

`0002/example_0001`:

| Word | Candidates | Pronunciation chosen by the aligner |
|---|---|---|
| it | 2 | `IH1 T` |
| of | 2 | `AH0 V` |
| the | 3 | `DH IY0` |

The second syllable of `irwine` is spoken as a weak schwa (`AH0`), so `ER1 W AH0 N` was chosen.

The final phone sequence is written to `text_info.phonemes` in the output manifest (words are separated by `|`, and `sil` marks a silence inserted by the aligner):

```
0001/example_0001  S EH1 D | M IH1 S T ER0 | sil | ER1 W AH0 N | sil | W IH0 TH | HH IH1 Z | S T EY1 T L IY0 | K AO1 R D IY0 AE1 L AH0 T IY0 | sil
0002/example_0001  sil | Y UW1 L | N EH1 V ER0 | D IH1 G | IH1 T | AW1 T | AH0 V | DH IY0 | AE1 S T ER0 | L AY1 B R EH2 R IY0 | sil
```

The raw text of `0001/example_0001` has no commas, yet the speaker paused after "Mr." (30 ms) and after "Irwine" (270 ms), as if reading "Said Mr. Irwine, with his stately cordiality". `0002/example_0001` has no pause inside the sentence.

Note the difference between training and synthesis. During training we have the audio, so the aligner can decide pronunciations and pauses. During synthesis we only have the text, so we must fall back on rules, such as taking the first lexicon candidate (or, for G2P, the most probable one) and using punctuation for pauses. As a result, the phone sequence of a sentence in the training data can differ from the one we feed in at synthesis time.


## Summary

The example outputs below are for `0001/example_0001`.

| Step | Tool | Output |
|---|---|---|
| 0. Raw text | `text_info.text` in the manifest | `Said Mr. Irwine with his stately cordiality` |
| 1. TN | `english_cleaners` (`unidecode`, `inflect`) | `said mister irwine with his stately cordiality` |
| 2. Word splitting | `mfa_safe`, MFA (internal) | `said / mister / irwine / ...` |
| 3. Lexicon lookup | librispeech-lexicon (ARPAbet) | pronunciation candidates per word |
| 4. OOV G2P | MFA `g2p` | `irwine` → pronunciation candidates |
| 5. Pronunciations, pauses | MFA `align` | phone sequence + `sil` (TextGrid, `text_info.phonemes` in the manifest) |

## Running it yourself

Run the commands from the repository folder. See the [README](../README.md) for installation.

- `python src/process_text.py` runs steps 0-3. It prints the result of each TN sub-step and the lexicon lookup of each word, and saves `../data/preprocess/example_dataset_16k/audio/<id>.lab` (next to the audio file) and `mfa/lexicon.txt`. It also fills `text_info.normalized_text` in the output manifest.
- Steps 4-5 are run by `bash src/align.sh` (Tutorial 3), which also fills `text_info.phonemes` in the output manifest.
- The functions live in `src/tutorial_lib.py` (`load_corpus`, `mfa_safe`, `read_lexicon`).
- To change the raw text, edit `text_info.text` in the dataset manifest; to change the lexicon, edit `text.lexicon` in `config.yaml`.

---

## Appendix A. ARPAbet

**ARPAbet** writes the phonemes of American English in plain ASCII. It was created for the ARPA Speech Understanding Project in the 1970s and is used by the CMU Pronouncing Dictionary (CMUdict) and the lexicons built on it, including librispeech-lexicon. Unlike IPA, it has no special symbols, which makes it easy to handle in code.

- 39 phonemes: 15 vowels and 24 consonants
- Vowels carry a stress digit: `0` unstressed, `1` primary stress, `2` secondary stress. For example, `AH0` is a weak "uh" (schwa) and `AH1` a stressed "uh".

**Vowels (15)**

| ARPAbet | IPA | Example | | ARPAbet | IPA | Example |
|---|---|---|---|---|---|---|
| AA | ɑ | **o**dd | | IH | ɪ | **i**t |
| AE | æ | **a**t | | IY | i | **ea**t |
| AH | ʌ, ə | h**u**t, **a**bout | | OW | oʊ | **oa**t |
| AO | ɔ | **ou**ght | | OY | ɔɪ | t**oy** |
| AW | aʊ | c**ow** | | UH | ʊ | h**oo**d |
| AY | aɪ | h**i**de | | UW | u | tw**o** |
| EH | ɛ | **E**d | | ER | ɝ, ɚ | h**ur**t |
| EY | eɪ | **a**te | | | | |

**Consonants (24)**

| ARPAbet | IPA | Example | | ARPAbet | IPA | Example |
|---|---|---|---|---|---|---|
| B | b | **b**e | | N | n | **kn**ee |
| CH | tʃ | **ch**eese | | NG | ŋ | pi**ng** |
| D | d | **d**ee | | P | p | **p**ee |
| DH | ð | **th**ee | | R | ɹ | **r**ead |
| F | f | **f**ee | | S | s | **s**ea |
| G | g | **g**reen | | SH | ʃ | **sh**e |
| HH | h | **h**e | | T | t | **t**ea |
| JH | dʒ | **g**ee | | TH | θ | **th**eta |
| K | k | **k**ey | | V | v | **v**ee |
| L | l | **l**ee | | W | w | **w**e |
| M | m | **m**e | | Y | j | **y**ield |
| Z | z | **z**ee | | ZH | ʒ | sei**z**ure |

## Appendix B. MFA (Montreal Forced Aligner)

[MFA](https://montreal-forced-aligner.readthedocs.io/) is a forced alignment tool: given speech and its transcript, it finds **the start and end times of every word and phone** (McAuliffe et al., Interspeech 2017, MIT license). It is built on the Kaldi speech recognition toolkit.

**Inputs and outputs**

| | Content |
|---|---|
| Input | audio + transcript (`.lab`), pronunciation lexicon, acoustic model |
| Output | TextGrid (Praat format) with a `words` tier and a `phones` tier |

**How it works**

1. Extract MFCCs from the audio every 10 ms.
2. Build a graph of phone candidates from the transcript using the lexicon. All pronunciation candidates of a word are included, and optional silences are placed between words.
3. Find the path that best matches the audio with the GMM-HMM acoustic model (Viterbi alignment). This step decides the pronunciations and whether there are pauses.
4. Adapt to the speaker (fMLLR) and align once more.

**Models and version used** (`mfa.*` in `config.yaml`)

- Acoustic model: `english_us_arpa` (trained on about 982 hours of LibriSpeech from 2,484 speakers; ARPAbet + stress)
- G2P model: `english_us_arpa` (based on Pynini/WFST)
- MFA 3.4.3, installed from conda-forge by `setup.sh`
- The alignment resolution is 10 ms.

## Appendix C. librispeech-lexicon and other data

### What is librispeech-lexicon?

- An English pronunciation lexicon published on [OpenSLR](https://www.openslr.org/11/) together with LibriSpeech. It covers the **200k words** of the LibriSpeech language model vocabulary (206,508 lines including pronunciation variants).
- Words found in CMUdict use the CMUdict pronunciation; the rest were generated by G2P.
- We use the copy included in the [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) repository, which `setup.sh` fetches. Since it is an existing resource, there is no need to build our own.

### What about data other than LibriSpeech?

**For English data, you can use it as is.** The lexicon is not tied to LibriSpeech; it is a general lexicon covering 200k English words. Words it does not contain can be filled in with the methods in Appendix D.

| Data | OOV words |
|---|---|
| The 2 example utterances in this tutorial | 1 (irwine) |
| All of LJSpeech (13,100 utterances) | 411 |

Other English lexicons work as well:

| Lexicon | Notes |
|---|---|
| [CMUdict](https://github.com/cmusphinx/cmudict) | about 130k words, ARPAbet; the basis of librispeech-lexicon |
| MFA `english_us_arpa` dictionary | `mfa model download dictionary english_us_arpa` |

**There is one requirement: the phone notation of the lexicon must match the phone set of the acoustic model.** The `english_us_arpa` acoustic model uses ARPAbet with stress, so any ARPAbet lexicon will do. To use an IPA lexicon (e.g. MFA `english_mfa`), you also need to switch the acoustic model to `english_mfa`.

**For other languages**, you need a lexicon and an acoustic model for that language.

- MFA provides lexicons and models for many languages (e.g. `korean_mfa` for Korean).
- If no suitable model exists, you can train one on your own data: `mfa train` for the acoustic model and `mfa train_g2p` for G2P.

## Appendix D. Handling OOV words

### Processing order (`src/align.sh`)

1. **Find:** `mfa find_oovs` lists the words that are not in the lexicon.
2. **Generate:** `mfa g2p` generates pronunciation candidates for each OOV word.
3. **Merge:** the candidates are merged with the original lexicon into the lexicon used for alignment.
4. **Choose:** during alignment, MFA picks the candidate that matches the audio.

### How many candidates does G2P produce?

For the 411 OOV words of LJSpeech, the `english_us_arpa` G2P produced 945 candidates in total:

| Candidates per word | 1 | 2 | 3 | 4 | 5 | 6 or more |
|---|---|---|---|---|---|---|
| Words | 190 | 114 | 47 | 20 | 15 | 25 |

Several candidates are not a problem, because the aligner chooses among them using the audio. The real problem is when **all of the candidates are wrong**.

### When G2P gets it wrong: examples from LJSpeech

| OOV word (occurrences) | Cause | G2P result | Actual pronunciation |
|---|---|---|---|
| `misess` (247) | TN rewrote "Mrs." into a spelling that is not in the lexicon (the abbreviation list of `english_cleaners` maps mrs → misess) | `M AY1 Z EH1 S` and 6 more | missus: `M IH1 S IH0 Z` |
| `fbi` (165) | acronym that TN does not spell out letter by letter | `F B IY0` only | eff-bee-eye: `EH1 F B IY1 AY1` |
| `prs` (48) | acronym (Protective Research Section) | `P R Z`, `P Z` | `P IY1 AA1 R EH1 S` |

A G2P model that only looks at the spelling cannot handle acronyms or the odd spellings produced by normalization. Such words can be fixed in one of these ways:

| Method | Example |
|---|---|
| **Add the word to the lexicon directly** | `misess  M IH1 S IH0 Z`, `fbi  EH1 F B IY1 AY1` |
| Change the TN rules | mrs → missus; spell out upper-case acronyms (FBI → f b i) |
| Use a different G2P | [`g2p_en`](https://github.com/Kyubyong/g2p) (neural, based on CMUdict, ARPAbet) |

A practical approach is to sort the OOV list by number of occurrences (`mfa/oovs/oov_counts_*.txt`) and check the G2P results from the top. In LJSpeech, 30% of all OOV occurrences came from just 9 words (misess, fbi, prs, fpcc, ussr, ...), so adding only those 9 words to the lexicon made a large difference.
