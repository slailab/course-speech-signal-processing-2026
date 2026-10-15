"""Phoneme-level linguistic features (239 dims; guide: docs/01_linguistic_features.md).

Input is an aligned phone sequence: ARPAbet phones with stress digits plus SIL, and for
each phone the index of the word it belongs to (None for SIL). Syllables come from
syllabify, applied within each word; phrases are runs of phones between SILs.
"""

import numpy as np
from tutorial_lib import SIL

# 39 ARPAbet phones + sil
PHONES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F",
    "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S",
    "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", SIL,
]  # fmt: skip
# Quin-phone context: two phones before, current, two phones after
QUIN = ["LL", "L", "C", "R", "RR"]
STRESS = ["0", "1", "2", "none"]
ONC = ["onset", "nucleus", "coda"]
POS4 = ["single", "begin", "middle", "end"]
NUMERIC = [
    "n_phones_in_syl", "n_syls_in_word", "n_phones_in_word", "n_words_in_phrase",
    "n_syls_in_phrase", "n_syls_in_utt", "n_words_in_utt", "n_phrases_in_utt",
]  # fmt: skip


def feature_names():
    """Names of the 239 dimensions, in order: 231 categorical (0/1) + 8 numeric."""
    names = [f"{q}={p}" for q in QUIN for p in PHONES]
    names += [f"{w}_syl_stress={s}" for w in ("prev", "cur", "next") for s in STRESS]
    names += [f"phone_in_syl={c}" for c in ONC]
    for unit in ("phone_in_word", "syl_in_word", "word_in_phrase", "phrase_in_utt"):
        names += [f"{unit}={p}" for p in POS4]
    return names + NUMERIC


N_FEATURES = len(feature_names())  # 239


def base_phone(p):
    """Phone identity without the stress digit: "EH1" -> "EH"."""
    return p[:-1] if p[-1].isdigit() else p


def pos4(i, n):
    """Position of item i among n: single / begin / middle / end."""
    if n == 1:
        return "single"
    if i == 0:
        return "begin"
    return "end" if i == n - 1 else "middle"


def annotate(phones, word_idx, syllabify):
    """Utterance structure: syllables, words and phrases, and per-phone membership.

    Returns (info, syllables, words, phrases, phrase_of, utt):
      info:      per phone, {"onc": onset/nucleus/coda, "syl": syllable index}
      syllables: [{"phones": [phone indices], "stress": "0"/"1"/"2"/"none", "word": k}]
      words:     {word index: [phone indices]}
      phrases:   [[phone indices]], maximal runs of non-SIL phones
      phrase_of: {phone index: phrase index}
      utt:       numbers of syllables, words and phrases in the utterance
    """
    info = [{} for _ in phones]

    # Words -> syllables: syllabify each word on its own
    words = {}
    for i, k in enumerate(word_idx):
        if k is not None:
            words.setdefault(k, []).append(i)
    syllables = []
    for k in sorted(words):
        pos = words[k]
        try:
            sylls = syllabify([phones[i] for i in pos])
        except ValueError:
            # No vowel (e.g. a reduced word): one syllable made of onsets only
            sylls = [([phones[i] for i in pos], [], [])]
        cursor = 0
        for onset, nucleus, coda in sylls:
            members = []
            for role, part in (("onset", onset), ("nucleus", nucleus), ("coda", coda)):
                for _ in part:
                    info[pos[cursor]]["onc"] = role
                    members.append(pos[cursor])
                    cursor += 1
            # Stress of the syllable = stress digit of its vowel
            stress = next((phones[i][-1] for i in members if phones[i][-1].isdigit()), "none")
            syllables.append({"phones": members, "stress": stress, "word": k})
        assert cursor == len(pos)
    for s_idx, syl in enumerate(syllables):
        for i in syl["phones"]:
            info[i]["syl"] = s_idx

    # Phrases: maximal runs of non-SIL phones
    phrases, cur = [], []
    for i, p in enumerate(phones):
        if p == SIL:
            if cur:
                phrases.append(cur)
            cur = []
        else:
            cur.append(i)
    if cur:
        phrases.append(cur)
    phrase_of = {i: j for j, ph in enumerate(phrases) for i in ph}

    utt = {"n_syls": len(syllables), "n_words": len(words), "n_phrases": len(phrases)}
    return info, syllables, words, phrases, phrase_of, utt


def phone_features(phones, word_idx, syllabify):
    """(N, 239) float32 phoneme-level feature matrix, one row per phone."""
    info, syllables, words, phrases, phrase_of, utt = annotate(phones, word_idx, syllabify)
    col = {name: j for j, name in enumerate(feature_names())}
    x = np.zeros((len(phones), N_FEATURES), dtype=np.float32)

    # Words of each phrase, syllables of each word
    phrase_words = [sorted({word_idx[i] for i in ph}) for ph in phrases]
    word_syls = {}
    for s_idx, syl in enumerate(syllables):
        word_syls.setdefault(syl["word"], []).append(s_idx)

    for i, p in enumerate(phones):
        # Quin-phone identity; outside the utterance counts as SIL
        for q, off in zip(QUIN, (-2, -1, 0, 1, 2), strict=True):
            j = i + off
            ctx = base_phone(phones[j]) if 0 <= j < len(phones) else SIL
            x[i, col[f"{q}={ctx}"]] = 1
        # Utterance-level counts are set for every phone, including SIL
        x[i, col["n_syls_in_utt"]] = utt["n_syls"]
        x[i, col["n_words_in_utt"]] = utt["n_words"]
        x[i, col["n_phrases_in_utt"]] = utt["n_phrases"]

        # SIL: stress "none"; positions and the other counts stay 0
        if p == SIL:
            for w in ("prev", "cur", "next"):
                x[i, col[f"{w}_syl_stress=none"]] = 1
            continue

        s_idx, k, ph = info[i]["syl"], word_idx[i], phrase_of[i]
        syl = syllables[s_idx]
        # Stress of the previous / current / next syllable; a neighbour across a
        # pause (another phrase) or outside the utterance counts as "none"
        for w, t in (("prev", s_idx - 1), ("cur", s_idx), ("next", s_idx + 1)):
            ok = 0 <= t < len(syllables) and phrase_of[syllables[t]["phones"][0]] == ph
            x[i, col[f"{w}_syl_stress={syllables[t]['stress'] if ok else 'none'}"]] = 1
        x[i, col[f"phone_in_syl={info[i]['onc']}"]] = 1

        # Positions: phone in word, syllable in word, word in phrase, phrase in utterance
        wp, ws, pw = words[k], word_syls[k], phrase_words[ph]
        x[i, col[f"phone_in_word={pos4(wp.index(i), len(wp))}"]] = 1
        x[i, col[f"syl_in_word={pos4(ws.index(s_idx), len(ws))}"]] = 1
        x[i, col[f"word_in_phrase={pos4(pw.index(k), len(pw))}"]] = 1
        x[i, col[f"phrase_in_utt={pos4(ph, len(phrases))}"]] = 1

        # Counts of the current syllable, word and phrase
        x[i, col["n_phones_in_syl"]] = len(syl["phones"])
        x[i, col["n_syls_in_word"]] = len(ws)
        x[i, col["n_phones_in_word"]] = len(wp)
        x[i, col["n_words_in_phrase"]] = len(pw)
        x[i, col["n_syls_in_phrase"]] = sum(len(word_syls[w]) for w in pw)
    return x
