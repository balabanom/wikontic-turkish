"""
run_compare.py

Produces diffs between two extraction runs:
final triplets, entity merges, and filter reason breakdowns.
"""
from collections import Counter
from typing import Dict, List, Tuple

from .run_reader import get_artifact, get_run


# ── Normalization ─────────────────────────────────────────────────────────────

def _triplet_key(t: dict) -> Tuple[str, str, str]:
    return (
        str(t.get("subject", "")).strip().lower(),
        str(t.get("relation", "")).strip().lower(),
        str(t.get("object", "")).strip().lower(),
    )


def _merge_key(m: dict) -> Tuple[str, str]:
    return (
        str(m.get("from", "")).strip().lower(),
        str(m.get("to", "")).strip().lower(),
    )


# ── Artifact getters ──────────────────────────────────────────────────────────

def _get_final_triplets(run_id: str, db_name: str | None = None) -> List[dict]:
    art = get_artifact(run_id, "final_triplets", db_name=db_name)
    return (art or {}).get("triplets", [])


def _get_parsed_triplets(run_id: str, db_name: str | None = None) -> List[dict]:
    art = get_artifact(run_id, "parsed_triplets", db_name=db_name)
    return (art or {}).get("triplets", [])


def _get_entity_merges(run_id: str, db_name: str | None = None) -> List[dict]:
    art = get_artifact(run_id, "merge_map_entities", db_name=db_name)
    return (art or {}).get("merges", [])


def _get_filtered_triplets(run_id: str, db_name: str | None = None) -> List[dict]:
    art = get_artifact(run_id, "filtered_out", db_name=db_name)
    return (art or {}).get("triplets", [])


def _get_stats(run_id: str, db_name: str | None = None) -> dict:
    meta = get_run(run_id, db_name=db_name) or {}
    return meta.get("stats") or {}


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_runs(run_id_a: str, run_id_b: str, db_name: str | None = None) -> dict:
    """
    Produce a comprehensive diff report between two runs.

    Returns:
    {
        "run_id_a": ...,
        "run_id_b": ...,
        "summary": {
            "raw_count_a": int,   "raw_count_b": int,
            "final_count_a": int, "final_count_b": int,
            "filtered_count_a": int, "filtered_count_b": int,
            "merges_entity_count_a": int, "merges_entity_count_b": int,
        },
        "final_diff": {
            "added_edges": [{"subject":..., "relation":..., "object":...}, ...],
            "removed_edges": [...],
            "added_count": int,
            "removed_count": int,
        },
        "merge_diff": {
            "entity": {
                "count_a": int, "count_b": int,
                "added": [...],    # present in B but not in A
                "removed": [...],  # present in A but not in B
            }
        },
        "filter_reason_diff": {
            "reasons": [
                {"reason": ..., "count_a": int, "count_b": int, "delta": int},
                ...
            ]
        }
    }
    """
    # ── Final triplet diff ────────────────────────────────────────────────────
    final_a = _get_final_triplets(run_id_a, db_name=db_name)
    final_b = _get_final_triplets(run_id_b, db_name=db_name)

    keys_a = {_triplet_key(t): t for t in final_a}
    keys_b = {_triplet_key(t): t for t in final_b}

    added_keys   = set(keys_b.keys()) - set(keys_a.keys())
    removed_keys = set(keys_a.keys()) - set(keys_b.keys())

    added_edges   = [keys_b[k] for k in added_keys]
    removed_edges = [keys_a[k] for k in removed_keys]

    # ── Merge diff ────────────────────────────────────────────────────────────
    merges_a = _get_entity_merges(run_id_a, db_name=db_name)
    merges_b = _get_entity_merges(run_id_b, db_name=db_name)

    mkeys_a = {_merge_key(m): m for m in merges_a}
    mkeys_b = {_merge_key(m): m for m in merges_b}

    entity_merge_added   = [mkeys_b[k] for k in set(mkeys_b) - set(mkeys_a)]
    entity_merge_removed = [mkeys_a[k] for k in set(mkeys_a) - set(mkeys_b)]

    # ── Filter reason diff ────────────────────────────────────────────────────
    filtered_a = _get_filtered_triplets(run_id_a, db_name=db_name)
    filtered_b = _get_filtered_triplets(run_id_b, db_name=db_name)

    reason_count_a: Counter = Counter(
        t.get("reason_code", "UNKNOWN") for t in filtered_a
    )
    reason_count_b: Counter = Counter(
        t.get("reason_code", "UNKNOWN") for t in filtered_b
    )

    all_reasons = sorted(set(reason_count_a.keys()) | set(reason_count_b.keys()))
    reason_diff = [
        {
            "reason": r,
            "count_a": reason_count_a.get(r, 0),
            "count_b": reason_count_b.get(r, 0),
            "delta": reason_count_b.get(r, 0) - reason_count_a.get(r, 0),
        }
        for r in all_reasons
    ]

    # ── Summary ───────────────────────────────────────────────────────────────
    parsed_a = _get_parsed_triplets(run_id_a, db_name=db_name)
    parsed_b = _get_parsed_triplets(run_id_b, db_name=db_name)

    summary = {
        "raw_count_a":            len(parsed_a),
        "raw_count_b":            len(parsed_b),
        "final_count_a":          len(final_a),
        "final_count_b":          len(final_b),
        "filtered_count_a":       len(filtered_a),
        "filtered_count_b":       len(filtered_b),
        "merges_entity_count_a":  len(merges_a),
        "merges_entity_count_b":  len(merges_b),
    }

    return {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "summary": summary,
        "final_diff": {
            "added_edges":   added_edges,
            "removed_edges": removed_edges,
            "added_count":   len(added_edges),
            "removed_count": len(removed_edges),
        },
        "merge_diff": {
            "entity": {
                "count_a":  len(merges_a),
                "count_b":  len(merges_b),
                "added":    entity_merge_added,
                "removed":  entity_merge_removed,
            }
        },
        "filter_reason_diff": {
            "reasons": reason_diff,
        },
    }


# ── Telemetry diff ────────────────────────────────────────────────────────────

_TIMED_STAGES = [
    "llm_extract",
    "parse",
    "merge",
    "ontology_alignment",
    "filter",
    "db_write",
]


def compare_telemetry(run_id_a: str, run_id_b: str, db_name: str | None = None) -> dict:
    """
    Produce a per-stage timing diff between two runs.

    Returns:
    {
        "total_time_ms_a": float | None,
        "total_time_ms_b": float | None,
        "total_delta_ms": float | None,
        "stages": [
            {"stage": "llm_extract", "ms_a": ..., "ms_b": ..., "delta_ms": ...},
            ...
        ]
    }
    """
    meta_a = get_run(run_id_a, db_name=db_name) or {}
    meta_b = get_run(run_id_b, db_name=db_name) or {}

    stats_a = meta_a.get("stats") or {}
    stats_b = meta_b.get("stats") or {}

    total_a = stats_a.get("total_time_ms")
    total_b = stats_b.get("total_time_ms")

    total_delta = (
        round(total_b - total_a, 2)
        if total_a is not None and total_b is not None
        else None
    )

    stage_rows = []
    for stage in _TIMED_STAGES:
        key = f"{stage}_time_ms"
        ms_a = stats_a.get(key)
        ms_b = stats_b.get(key)
        delta = (
            round(ms_b - ms_a, 2)
            if ms_a is not None and ms_b is not None
            else None
        )
        stage_rows.append({
            "stage":    stage,
            "ms_a":     ms_a,
            "ms_b":     ms_b,
            "delta_ms": delta,
        })

    return {
        "total_time_ms_a": total_a,
        "total_time_ms_b": total_b,
        "total_delta_ms":  total_delta,
        "stages":          stage_rows,
    }
