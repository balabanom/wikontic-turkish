"""
sentence_splitter.py

input_text'i cümlelere böler.
Deterministic: aynı input → aynı çıktı.

Kullanım:
    from .sentence_splitter import split_sentences

    sentences = split_sentences("Einstein was born in Ulm. He won the Nobel Prize.")
    # [
    #   {"id": 0, "text": "Einstein was born in Ulm.", "start": 0,  "end": 25},
    #   {"id": 1, "text": "He won the Nobel Prize.",  "start": 26, "end": 49},
    # ]
"""

import re
from typing import List


# Cümle sonlandırıcı + ardından boşluk + büyük harf/rakam
# Parantez içindeki kısaltmalar (U.S.A., Mr., Dr.) bölünmesin
_SPLIT_PATTERN = re.compile(
    r"(?<!\w\.\w.)"            # kısaltma değil: U.S.A.
    r"(?<![A-Z][a-z]\.)"       # kısaltma değil: Mr. Dr.
    r"(?<=\.|\!|\?)"           # nokta/ünlem/soru işaretinden sonra
    r"\s+"                     # en az bir boşluk
    r"(?=[A-ZÇĞİÖŞÜ0-9\"])"   # büyük harf, rakam veya tırnak ile devam
)


def split_sentences(text: str) -> List[dict]:
    """
    input_text'i cümlelere böler.

    Returns:
        List[{id: int, text: str, start: int, end: int}]
        start/end: orijinal text içindeki karakter indeksleri (end exclusive)
    """
    if not text or not text.strip():
        return []

    raw_parts = _SPLIT_PATTERN.split(text)

    sentences = []
    cursor = 0

    for raw in raw_parts:
        stripped = raw.strip()
        if not stripped:
            cursor += len(raw)
            continue

        start = text.find(stripped, cursor)
        if start == -1:
            cursor += len(raw)
            continue

        end = start + len(stripped)
        sentences.append({
            "id":    len(sentences),
            "text":  stripped,
            "start": start,
            "end":   end,
        })
        cursor = end

    # Hiç bölünemediyse tüm text tek cümle
    if not sentences and text.strip():
        sentences.append({
            "id":    0,
            "text":  text.strip(),
            "start": 0,
            "end":   len(text.strip()),
        })

    return sentences


def sentences_to_numbered_str(sentences: List[dict]) -> str:
    """
    LLM prompt'una eklenecek numaralı cümle listesi üretir.

    Örnek çıktı:
        [0] Einstein was born in Ulm.
        [1] He won the Nobel Prize in 1921.
    """
    return "\n".join(f"[{s['id']}] {s['text']}" for s in sentences)