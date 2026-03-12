import os
import uuid
from datetime import datetime, timezone

from pymongo import MongoClient, DESCENDING, ASCENDING

_client = None
_db = None


def _get_db():
    global _client, _db
    if _db is None:
        mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        _client = MongoClient(mongo_uri)
        _db = _client["demo"]
        _ensure_indexes(_db)
    return _db


def _ensure_indexes(db):
    runs = db["extraction_runs"]
    runs.create_index([("created_at", DESCENDING)], background=True)
    runs.create_index([("sample_id", ASCENDING)], background=True)
    runs.create_index([("status", ASCENDING)], background=True)
    runs.create_index([("model", ASCENDING)], background=True)
    runs.create_index([("parent_run_id", ASCENDING)], background=True)

    artifacts = db["extraction_artifacts"]
    artifacts.create_index([("run_id", ASCENDING)], background=True)
    artifacts.create_index(
        [("run_id", ASCENDING), ("stage", ASCENDING)],
        unique=True,
        background=True,
    )


def start_run(
    sample_id: str,
    model: str,
    input_text: str | None = None,
    extra_config: dict | None = None,
    parent_run_id: str | None = None,
) -> str:
    """
    Yeni bir extraction run başlatır.

    Args:
        sample_id:      Streamlit user/session ID
        model:          Kullanılan model adı
        input_text:     Ham input metni (export/replay için şart)
        extra_config:   Ek config parametreleri
        parent_run_id:  Replay ise orijinal run'ın ID'si
    """
    db = _get_db()
    run_id = str(uuid.uuid4())

    doc = {
        "_id": run_id,
        "created_at": datetime.now(timezone.utc),
        "sample_id": sample_id,
        "model": model,
        "input_text": input_text or "",
        "status": "STARTED",
        "error": None,
        "stats": None,
        "extra_config": extra_config or {},
        "parent_run_id": parent_run_id,   # None ise normal run, dolu ise replay
    }

    db["extraction_runs"].insert_one(doc)
    return run_id


def log_artifact(run_id: str, stage: str, payload: dict) -> None:
    """
    Bir pipeline stage'inin çıktısını kaydeder.

    stage örnekleri:
        "raw_llm_output"
        "parsed_triplets"
        "merge_map_entities"
        "filtered_out"
        "final_triplets"
    """
    db = _get_db()

    doc = {
        "run_id": run_id,
        "stage": stage,
        "payload": payload,
        "created_at": datetime.now(timezone.utc),
    }

    db["extraction_artifacts"].update_one(
        {"run_id": run_id, "stage": stage},
        {"$set": doc},
        upsert=True,
    )


def finish_run(
    run_id: str,
    status: str = "DONE",
    error: str | None = None,
    stats: dict | None = None,
) -> None:
    """
    Run'ı tamamlandı (DONE) ya da hatalı (FAILED) olarak işaretler.
    """
    db = _get_db()

    update = {
        "$set": {
            "status": status,
            "error": error,
            "stats": stats or {},
            "finished_at": datetime.now(timezone.utc),
        }
    }

    db["extraction_runs"].update_one({"_id": run_id}, update)