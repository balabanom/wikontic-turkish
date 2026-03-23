"""
sentence_matcher.py

LLM sentence_id vermediğinde veya geçersiz verdiğinde çalışan fallback matcher.

Kullanım:
    from .sentence_matcher import assign_sentence_id

    sid = assign_sentence_id(triplet, sentences)
    # → int veya None (min_score altındaysa)
"""

import re
from typing import Optional, List


# Minimum eşik: en az bu kadar puan olmadan eşleştirme yapmayız
_MIN_SCORE = 2


def _normalize(text: str) -> str:
    """Küçük harf, noktalama temizle."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _word_overlap(a: str, b: str) -> int:
    """İki string arasındaki ortak kelime sayısı."""
    words_a = set(_normalize(a).split())
    words_b = set(_normalize(b).split())
    return len(words_a & words_b)


def _contains(needle: str, haystack: str) -> bool:
    """needle, haystack içinde substring olarak geçiyor mu? (case-insensitive)"""
    return _normalize(needle) in _normalize(haystack)


def assign_sentence_id(
    triplet: dict,
    sentences: List[dict],
    min_score: int = _MIN_SCORE,
) -> Optional[int]:
    """
    Triplet için en uygun sentence_id'yi döner.

    Puanlama:
        subject substring geçiyorsa   +2
        object  substring geçiyorsa   +2
        relation kelime overlap       +1 per ortak kelime (max 2)

    Eşitlikte: daha düşük id (ilk geçen cümle) kazanır.
    min_score altında: None döner (unmatched).

    Args:
        triplet:   {"subject": ..., "relation": ..., "object": ...}
        sentences: split_sentences() çıktısı
        min_score: bu puanın altındaki eşleşmeler None döner

    Returns:
        sentence_id (int) veya None
    """
    if not sentences:
        return None

    subject  = str(triplet.get("subject",  "") or "")
    relation = str(triplet.get("relation", "") or "")
    obj      = str(triplet.get("object",   "") or "")

    best_id    = None
    best_score = -1

    for sent in sentences:
        text  = sent["text"]
        score = 0

        if subject and _contains(subject, text):
            score += 2
        if obj and _contains(obj, text):
            score += 2

        # Relation kelime overlap (max 2 puan)
        if relation:
            overlap = min(_word_overlap(relation, text), 2)
            score += overlap

        if score > best_score:
            best_score = score
            best_id    = sent["id"]

    if best_score < min_score:
        return None

    return best_id


def enrich_triplets_with_sentence_ids(
    triplets: List[dict],
    sentences: List[dict],
) -> List[dict]:
    """
    Triplet listesindeki her triplet için:
    - sentence_id None veya geçersizse → fallback matcher çalıştır
    - Geçerliyse → olduğu gibi bırak

    Returns: güncellenmiş triplet listesi (in-place değil, kopya)
    """
    n = len(sentences)
    result = []

    for t in triplets:
        t = dict(t)  # kopya al
        sid = t.get("sentence_id")

        # Geçerli mi? (int, 0 <= sid < n)
        valid = (
            isinstance(sid, int)
            and 0 <= sid < n
        )

        if not valid:
            t["sentence_id"] = assign_sentence_id(t, sentences)

        result.append(t)

    return result