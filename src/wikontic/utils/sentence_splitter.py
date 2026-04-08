"""
sentence_splitter.py

Splits input text into sentences. Deterministic: identical input → identical output.

Usage:
    from .sentence_splitter import split_sentences

    sentences = split_sentences("Einstein was born in Ulm. He won the Nobel Prize.")
    # [
    #   {"id": 0, "text": "Einstein was born in Ulm.", "start": 0,  "end": 25},
    #   {"id": 1, "text": "He won the Nobel Prize.",  "start": 26, "end": 49},
    # ]
"""

import re
from typing import List


# Splits on sentence-ending punctuation followed by whitespace + capital/digit.
# Negative lookbehinds prevent splitting on abbreviations like U.S.A., Mr., Dr.
_SPLIT_PATTERN = re.compile(
    r"(?<!\w\.\w.)"            # not an abbreviation: U.S.A.
    r"(?<![A-Z][a-z]\.)"       # not an abbreviation: Mr. Dr.
    r"(?<=\.|\!|\?)"           # after sentence-ending punctuation
    r"\s+"                     # at least one whitespace
    r"(?=[A-ZÇĞİÖŞÜ0-9\"])"   # followed by capital letter, digit, or quote
)


def split_sentences(text: str) -> List[dict]:
    """
    Split text into sentences with character-level provenance.

    Returns:
        List[{id: int, text: str, start: int, end: int}]
        start/end are character indices into the original text (end exclusive).
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

    # If no split occurred, treat the entire text as a single sentence.
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
    Format sentences for inclusion in an LLM prompt.

    Example output:
        [0] Einstein was born in Ulm.
        [1] He won the Nobel Prize in 1921.
    """
    return "\n".join(f"[{s['id']}] {s['text']}" for s in sentences)