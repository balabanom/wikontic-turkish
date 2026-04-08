"""
sentence_matcher.py

Fallback matcher used when the LLM omits or provides an invalid sentence_id.

Usage:
    from .sentence_matcher import assign_sentence_id

    sid = assign_sentence_id(triplet, sentences)
    # → int or None (if best score is below min_score)
"""

import re
from typing import Optional, List


# Matches scoring below this threshold are treated as unmatched (return None).
_MIN_SCORE = 2


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _word_overlap(a: str, b: str) -> int:
    """Count shared words between two strings."""
    words_a = set(_normalize(a).split())
    words_b = set(_normalize(b).split())
    return len(words_a & words_b)


def _contains(needle: str, haystack: str) -> bool:
    """Case-insensitive substring check."""
    return _normalize(needle) in _normalize(haystack)


def assign_sentence_id(
    triplet: dict,
    sentences: List[dict],
    min_score: int = _MIN_SCORE,
) -> Optional[int]:
    """
    Return the best-matching sentence_id for a triplet using word-overlap scoring.

    Scoring:
        subject appears as substring  → +2
        object  appears as substring  → +2
        relation word overlap         → +1 per shared word (capped at 2)

    Ties are broken by lowest id (earliest sentence wins).
    Returns None if the best score is below min_score.

    Args:
        triplet:   {"subject": ..., "relation": ..., "object": ...}
        sentences: output of split_sentences()
        min_score: minimum score to accept a match
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

        # Relation word overlap capped at 2 points.
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
    Ensure every triplet has a valid sentence_id.
    Runs the fallback matcher for any triplet whose sentence_id is missing or out-of-range.
    Returns a new list of copies; the originals are not modified.
    """
    n = len(sentences)
    result = []

    for t in triplets:
        t = dict(t)
        sid = t.get("sentence_id")

        valid = (
            isinstance(sid, int)
            and 0 <= sid < n
        )

        if not valid:
            t["sentence_id"] = assign_sentence_id(t, sentences)

        result.append(t)

    return result