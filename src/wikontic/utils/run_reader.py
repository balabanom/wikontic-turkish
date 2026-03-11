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


# ── Normalizasyon ──────────────────────────────────────────────────────────────

def _normalize_raw_llm_output(payload: dict) -> dict:
    """raw_llm_output payload'ını standart formata getirir."""
    return {
        "text": str(payload.get("text", payload.get("raw", ""))),
        "format": payload.get("format", "string"),
        "type": payload.get("type", "unknown"),
    }


def _normalize_triplets_payload(payload: dict) -> dict:
    """parsed_triplets / final_triplets payload'ını standart formata getirir."""
    triplets = payload.get("triplets", [])

    # Eski format: doğrudan liste geldiyse
    if isinstance(payload, list):
        triplets = payload

    normalized = []
    for t in triplets:
        normalized.append(
            {
                "subject": t.get("subject", ""),
                "relation": t.get("relation", ""),
                "object": t.get("object", ""),
                # final_triplets'te varsa tip bilgisini de al
                "subject_type": t.get("subject_type", ""),
                "object_type": t.get("object_type", ""),
            }
        )

    return {
        "triplets": normalized,
        "count": payload.get("count", len(normalized)),
        "filtered_count": payload.get("filtered_count", None),
        "ontology_filtered_count": payload.get("ontology_filtered_count", None),
    }


_NORMALIZERS = {
    "raw_llm_output": _normalize_raw_llm_output,
    "parsed_triplets": _normalize_triplets_payload,
    "final_triplets": _normalize_triplets_payload,
}


def _normalize(stage: str, payload: dict) -> dict:
    normalizer = _NORMALIZERS.get(stage)
    if normalizer:
        return normalizer(payload)
    return payload


# ── Public API ─────────────────────────────────────────────────────────────────

def get_run(run_id: str) -> Optional[dict]:
    """
    Verilen run_id için metadata döner.
    Bulunamazsa None döner.
    """
    try:
        db = _get_db()
        doc = db["extraction_runs"].find_one({"_id": run_id})
        return doc
    except Exception:
        return None


def get_artifact(run_id: str, stage: str) -> Optional[dict]:
    """
    Verilen run_id + stage için normalize edilmiş payload döner.
    Bulunamazsa None döner.
    """
    try:
        db = _get_db()
        doc = db["extraction_artifacts"].find_one(
            {"run_id": run_id, "stage": stage}
        )
        if doc is None:
            return None
        return _normalize(stage, doc.get("payload", {}))
    except Exception:
        return None


def list_recent_runs(limit: int = 20, sample_id: Optional[str] = None) -> list:
    """
    Son run'ların özetini döner (dropdown için).
    Her eleman: {run_id, created_at, model, sample_id, status, preview}

    preview: run'ın sample_id'si veya created_at string'i — dropdown'da göstermek için.
    """
    try:
        db = _get_db()
        query = {}
        if sample_id:
            query["sample_id"] = sample_id

        cursor = (
            db["extraction_runs"]
            .find(query, {"_id": 1, "created_at": 1, "model": 1, "sample_id": 1, "status": 1})
            .sort("created_at", DESCENDING)
            .limit(limit)
        )

        results = []
        for doc in cursor:
            created_at = doc.get("created_at")
            if isinstance(created_at, datetime):
                created_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
            else:
                created_str = str(created_at)

            results.append(
                {
                    "run_id": doc["_id"],
                    "created_at": created_str,
                    "model": doc.get("model", "unknown"),
                    "sample_id": doc.get("sample_id", ""),
                    "status": doc.get("status", ""),
                    # Dropdown'da gösterilecek label
                    "label": f"{created_str}  |  {doc.get('model', 'unknown')}  |  {doc.get('status', '')}",
                }
            )
        return results
    except Exception:
        return []