import os
from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient, DESCENDING

_client = None
_db = None


def _get_db():
    global _client, _db
    if _db is None:
        mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        _client = MongoClient(mongo_uri)
        _db = _client["demo"]
    return _db


# ── Normalization ─────────────────────────────────────────────────────────────

def _normalize_raw_llm_output(payload: dict) -> dict:
    return {
        "text": str(payload.get("text", payload.get("raw", ""))),
        "format": payload.get("format", "string"),
        "type": payload.get("type", "unknown"),
    }


def _normalize_triplets_payload(payload: dict) -> dict:
    triplets = payload.get("triplets", [])
    if isinstance(payload, list):
        triplets = payload

    normalized = []
    for t in triplets:
        normalized.append(
            {
                "subject": t.get("subject", ""),
                "relation": t.get("relation", ""),
                "object": t.get("object", ""),
                "subject_type": t.get("subject_type", ""),
                "object_type":  t.get("object_type", ""),
                "sentence_id":  t.get("sentence_id"),
            }
        )

    # sentences lookup: sentence_id → text
    sentences    = payload.get("sentences", [])
    sid_to_text  = {s["id"]: s["text"] for s in sentences} if sentences else {}

    # Derive sentence_preview from the sentence lookup table.
    for t in normalized:
        sid = t.get("sentence_id")
        if sid is not None and sid in sid_to_text:
            full = sid_to_text[sid]
            t["sentence_preview"] = full[:80] + ("…" if len(full) > 80 else "")
            t["sentence_full"]    = full
        else:
            t["sentence_preview"] = None
            t["sentence_full"]    = None

    return {
        "triplets":                normalized,
        "count":                   payload.get("count", len(normalized)),
        "filtered_count":          payload.get("filtered_count", None),
        "ontology_filtered_count": payload.get("ontology_filtered_count", None),
        "sentences":               sentences,
    }


def _normalize_filtered_out(payload: dict) -> dict:
    triplets  = payload.get("triplets", [])
    sentences = payload.get("sentences", [])
    sid_to_text = {s["id"]: s["text"] for s in sentences} if sentences else {}

    normalized = []
    for t in triplets:
        sid  = t.get("sentence_id")
        full = sid_to_text.get(sid) if sid is not None else None
        normalized.append({
            "subject":         t.get("subject", ""),
            "relation":        t.get("relation", ""),
            "object":          t.get("object", ""),
            "reason_code":     t.get("reason_code", ""),
            "filter_stage":    t.get("filter_stage", ""),
            "exception_text":  t.get("exception_text", ""),
            "sentence_id":     sid,
            "sentence_preview": (full[:80] + "…" if full and len(full) > 80 else full),
            "sentence_full":   full,
        })
    return {
        "triplets":  normalized,
        "count": payload.get("count", len(normalized)),
        "pipeline_exception_count": payload.get("pipeline_exception_count", 0),
        "ontology_filtered_count": payload.get("ontology_filtered_count", 0),
    }


def _normalize_merge_map(payload: dict) -> dict:
    return {
        "merges": payload.get("merges", []),
        "count": payload.get("count", len(payload.get("merges", []))),
    }


_NORMALIZERS = {
    "raw_llm_output": _normalize_raw_llm_output,
    "parsed_triplets": _normalize_triplets_payload,
    "final_triplets": _normalize_triplets_payload,
    "filtered_out": _normalize_filtered_out,
    "merge_map_entities": _normalize_merge_map,
}


def _normalize(stage: str, payload: dict) -> dict:
    normalizer = _NORMALIZERS.get(stage)
    if normalizer:
        return normalizer(payload)
    return payload


def _fmt_datetime(dt) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt) if dt else ""


# ── Public API ─────────────────────────────────────────────────────────────────

def get_run(run_id: str) -> Optional[dict]:
    """Return run metadata for the given run_id, or None if not found."""
    try:
        db = _get_db()
        return db["extraction_runs"].find_one({"_id": run_id})
    except Exception:
        return None


def get_artifact(run_id: str, stage: str) -> Optional[dict]:
    """Return the normalized artifact payload for the given run_id + stage, or None."""
    try:
        db = _get_db()
        doc = db["extraction_artifacts"].find_one({"run_id": run_id, "stage": stage})
        if doc is None:
            return None
        return _normalize(stage, doc.get("payload", {}))
    except Exception:
        return None


def get_all_artifacts(run_id: str) -> dict:
    """
    Return all stage artifacts for a run as {"stage_name": payload_dict, ...}.
    Used by the ZIP export pipeline.
    """
    try:
        db = _get_db()
        cursor = db["extraction_artifacts"].find({"run_id": run_id})
        result = {}
        for doc in cursor:
            stage = doc.get("stage", "unknown")
            result[stage] = _normalize(stage, doc.get("payload", {}))
        return result
    except Exception:
        return {}


def list_recent_runs(
    limit: int = 50,
    sample_id: Optional[str] = None,
    status: Optional[str] = None,
    model: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> list:
    """
    Return a summary list of recent runs, sorted by creation time descending.

    Each entry contains: run_id, created_at, model, sample_id, status,
                         input_preview, stats, label.

    All filters are exact-match; date_from/date_to are inclusive bounds.
    """
    try:
        db = _get_db()
        query = {}

        if sample_id:
            query["sample_id"] = sample_id
        if status:
            query["status"] = status
        if model:
            query["model"] = model
        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter["$gte"] = date_from
            if date_to:
                date_filter["$lte"] = date_to
            query["created_at"] = date_filter

        cursor = (
            db["extraction_runs"]
            .find(
                query,
                {
                    "_id": 1, "created_at": 1, "model": 1,
                    "sample_id": 1, "status": 1,
                    "input_text": 1, "stats": 1,
                    "error": 1, "finished_at": 1,
                    "extra_config": 1,
                },
            )
            .sort("created_at", DESCENDING)
            .limit(limit)
        )

        results = []
        for doc in cursor:
            created_str = _fmt_datetime(doc.get("created_at"))
            raw_input = doc.get("input_text", "") or ""
            input_preview = raw_input[:150] + ("…" if len(raw_input) > 150 else "")
            model_name = doc.get("model", "unknown")
            status_val = doc.get("status", "")

            results.append(
                {
                    "run_id": doc["_id"],
                    "created_at": created_str,
                    "model": model_name,
                    "sample_id": doc.get("sample_id", ""),
                    "status": status_val,
                    "input_preview": input_preview,
                    "input_text": raw_input,
                    "stats": doc.get("stats") or {},
                    "error": doc.get("error"),
                    "finished_at": _fmt_datetime(doc.get("finished_at")),
                    "extra_config": doc.get("extra_config") or {},
                    "label": f"{created_str}  |  {model_name}  |  {status_val}",
                }
            )
        return results

    except Exception:
        return []


def get_distinct_models() -> list:
    """Return the list of distinct models seen across all runs."""
    try:
        db = _get_db()
        return db["extraction_runs"].distinct("model")
    except Exception:
        return []


def get_child_runs(parent_run_id: str) -> list:
    """Return all replay runs that reference the given parent_run_id."""
    try:
        db = _get_db()
        cursor = (
            db["extraction_runs"]
            .find(
                {"parent_run_id": parent_run_id},
                {"_id": 1, "created_at": 1, "model": 1, "status": 1},
            )
            .sort("created_at", DESCENDING)
        )
        results = []
        for doc in cursor:
            created_str = _fmt_datetime(doc.get("created_at"))
            results.append({
                "run_id": doc["_id"],
                "created_at": created_str,
                "model": doc.get("model", "unknown"),
                "status": doc.get("status", ""),
                "label": f"{created_str}  |  {doc.get('model', 'unknown')}  |  {doc.get('status', '')}",
            })
        return results
    except Exception:
        return []


def delete_run(run_id: str) -> dict:
    """
    Delete a run and all its artifacts.
    Artifacts are deleted first to avoid orphaned documents.

    Returns: {"runs_deleted": int, "artifacts_deleted": int, "ok": bool}
    """
    try:
        db = _get_db()
        art_res = db["extraction_artifacts"].delete_many({"run_id": run_id})
        run_res = db["extraction_runs"].delete_one({"_id": run_id})
        return {
            "runs_deleted":      run_res.deleted_count,
            "artifacts_deleted": art_res.deleted_count,
            "ok": True,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}