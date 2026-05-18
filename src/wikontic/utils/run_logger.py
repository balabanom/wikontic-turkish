import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient, DESCENDING, ASCENDING

from ..profiles.runtime_profile import DEFAULT_RUNTIME_PROFILE, RuntimeProfile

_client: Optional[MongoClient] = None
_db_cache: dict = {}

_DEFAULT_DB_NAME = DEFAULT_RUNTIME_PROFILE.triplets_db_name  # "triplets"


def _get_db(db_name: str):
    global _client, _db_cache
    if _client is None:
        mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        _client = MongoClient(mongo_uri)
    if db_name not in _db_cache:
        db = _client[db_name]
        _ensure_indexes(db)
        _db_cache[db_name] = db
    return _db_cache[db_name]


def _ensure_indexes(db):
    runs = db["extraction_runs"]
    runs.create_index([("created_at", DESCENDING)], background=True)
    runs.create_index([("sample_id", ASCENDING)], background=True)
    runs.create_index([("status", ASCENDING)], background=True)
    runs.create_index([("model", ASCENDING)], background=True)
    runs.create_index([("parent_run_id", ASCENDING)], background=True)
    # Profile-level indexes for future audit filtering
    runs.create_index([("profile_id", ASCENDING)], background=True)
    runs.create_index([("ontology_language", ASCENDING)], background=True)
    runs.create_index([("embedding_model_name", ASCENDING)], background=True)
    runs.create_index([("extra_config.batch_id", ASCENDING)], background=True)

    artifacts = db["extraction_artifacts"]
    artifacts.create_index([("run_id", ASCENDING)], background=True)
    artifacts.create_index(
        [("run_id", ASCENDING), ("stage", ASCENDING)],
        unique=True,
        background=True,
    )
    artifacts.create_index([("profile_id", ASCENDING)], background=True)
    artifacts.create_index([("embedding_model_name", ASCENDING)], background=True)
    artifacts.create_index([("embedding_profile_id", ASCENDING)], background=True)
    artifacts.create_index([("ontology_profile_id", ASCENDING)], background=True)
    artifacts.create_index([("payload.batch_info.batch_id", ASCENDING)], background=True)


def start_run(
    sample_id: str,
    model: str,
    input_text: Optional[str] = None,
    extra_config: Optional[dict] = None,
    parent_run_id: Optional[str] = None,
    runtime_profile: Optional[RuntimeProfile] = None,
    db_name: Optional[str] = None,
) -> str:
    """
    Create a new extraction run document and return its run_id.

    Args:
        sample_id:       Streamlit user/session identifier.
        model:           LLM model name used for this run.
        input_text:      Raw input text (required for export and replay).
        extra_config:    Arbitrary pipeline configuration metadata.
        parent_run_id:   Original run_id if this is a replay; None otherwise.
        runtime_profile: Active RuntimeProfile. Profile metadata is stored
                         as top-level fields for audit filtering.
        db_name:         Triplets DB to write to. Defaults to active profile's
                         triplets_db_name, or the global default.
    """
    resolved_db = db_name or (
        runtime_profile.triplets_db_name if runtime_profile else _DEFAULT_DB_NAME
    )
    db = _get_db(resolved_db)
    run_id = str(uuid.uuid4())

    profile_meta = runtime_profile.to_metadata_dict() if runtime_profile else {}

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
        "parent_run_id": parent_run_id,
        # Top-level profile fields (filter-friendly for future audit page)
        "profile_id": profile_meta.get("profile_id", ""),
        "ontology_profile_id": profile_meta.get("ontology_profile_id", ""),
        "embedding_profile_id": profile_meta.get("embedding_profile_id", ""),
        "ontology_db_name": profile_meta.get("ontology_db_name", ""),
        "triplets_db_name": profile_meta.get("triplets_db_name", resolved_db),
        "ontology_language": profile_meta.get("ontology_language", ""),
        "embedding_model_name": profile_meta.get("embedding_model_name", ""),
        "embedding_dimension": profile_meta.get("embedding_dimension", None),
    }

    db["extraction_runs"].insert_one(doc)
    return run_id


def log_artifact(
    run_id: str,
    stage: str,
    payload: dict,
    db_name: Optional[str] = None,
    profile_id: Optional[str] = None,
    runtime_profile: Optional[RuntimeProfile] = None,
) -> None:
    """
    Persist the output of a pipeline stage.

    Known stage names: raw_llm_output, parsed_triplets, merge_map_entities,
                       filtered_out, final_triplets.

    Args:
        run_id:           Run identifier.
        stage:            Pipeline stage name.
        payload:          Stage output data.
        db_name:          DB to write to. Defaults to global default.
        profile_id:       Optional explicit profile_id override.
        runtime_profile:  Optional RuntimeProfile for top-level filter fields.
    """
    db = _get_db(db_name or _DEFAULT_DB_NAME)
    profile_meta = runtime_profile.to_metadata_dict() if runtime_profile else {}

    doc = {
        "run_id": run_id,
        "stage": stage,
        "payload": payload,
        "created_at": datetime.now(timezone.utc),
        "profile_id": profile_id or profile_meta.get("profile_id", ""),
        "ontology_profile_id": profile_meta.get("ontology_profile_id", ""),
        "embedding_profile_id": profile_meta.get("embedding_profile_id", ""),
        "ontology_db_name": profile_meta.get("ontology_db_name", ""),
        "triplets_db_name": profile_meta.get("triplets_db_name", db_name or _DEFAULT_DB_NAME),
        "ontology_language": profile_meta.get("ontology_language", ""),
        "embedding_model_name": profile_meta.get("embedding_model_name", ""),
        "embedding_dimension": profile_meta.get("embedding_dimension", None),
    }

    db["extraction_artifacts"].update_one(
        {"run_id": run_id, "stage": stage},
        {"$set": doc},
        upsert=True,
    )


def finish_run(
    run_id: str,
    status: str = "DONE",
    error: Optional[str] = None,
    stats: Optional[dict] = None,
    db_name: Optional[str] = None,
) -> None:
    """Mark a run as DONE or FAILED and store final timing/token stats."""
    db = _get_db(db_name or _DEFAULT_DB_NAME)

    update = {
        "$set": {
            "status": status,
            "error": error,
            "stats": stats or {},
            "finished_at": datetime.now(timezone.utc),
        }
    }

    db["extraction_runs"].update_one({"_id": run_id}, update)
