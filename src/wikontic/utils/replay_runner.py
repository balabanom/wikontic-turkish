"""
replay_runner.py

Re-executes the extraction pipeline for a given run_id using its stored input and config.
Creates a new run_id linked to the original via parent_run_id.

The replay inherits the original run's ontology profile and embedding profile.
Changing the model alone is acceptable.
Changing the ontology/embedding profile must be an explicit, deliberate action.

Usage:
    new_run_id = replay_run("some-uuid", overrides={"model": "gpt-4.1"})
"""
import os
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient

from .run_reader import get_run

_ = load_dotenv(find_dotenv())


def _restore_profile_from_run(run_meta: dict):
    """
    Reconstruct a RuntimeProfile from the metadata stored in a run document.

    Tries top-level profile fields first (new format), then falls back to
    extra_config (legacy format), then falls back to default profile.
    """
    from ..profiles import resolve_runtime_profile, DEFAULT_RUNTIME_PROFILE

    ontology_profile_id = run_meta.get("ontology_profile_id") or (
        (run_meta.get("extra_config") or {}).get("ontology_profile_id")
    )
    embedding_profile_id = run_meta.get("embedding_profile_id") or (
        (run_meta.get("extra_config") or {}).get("embedding_profile_id")
    )

    if ontology_profile_id and embedding_profile_id:
        try:
            return resolve_runtime_profile(ontology_profile_id, embedding_profile_id)
        except ValueError:
            pass

    return DEFAULT_RUNTIME_PROFILE


def replay_run(
    run_id: str,
    overrides: Optional[dict] = None,
    db_name: Optional[str] = None,
) -> str:
    """
    Re-run the extraction pipeline using the input_text and config of an existing run.

    The ontology and embedding profile are inherited from the original run.
    Only the LLM model and sample_id may be overridden without a profile switch.

    Args:
        run_id:    ID of the original run to replay.
        overrides: Optional overrides, e.g. {"model": "gpt-4.1", "sample_id": "..."}.
        db_name:   DB containing the original run. Defaults to the original run's
                   triplets_db_name (read from run metadata), then global default.

    Returns:
        new_run_id (str)

    Raises:
        ValueError: if the run is not found or has no input_text.
        Exception:  on pipeline failure (run is marked FAILED).
    """
    from ..profiles.runtime_profile import DEFAULT_RUNTIME_PROFILE

    # Resolve which DB to look in for the original run
    resolved_db = db_name or DEFAULT_RUNTIME_PROFILE.triplets_db_name
    run_meta = get_run(run_id, db_name=resolved_db)

    # Try other known profile DBs if not found in the resolved DB
    if run_meta is None:
        from ..profiles import ONTOLOGY_PROFILES, EMBEDDING_PROFILES, resolve_runtime_profile
        for op_id in ONTOLOGY_PROFILES:
            for ep_id in EMBEDDING_PROFILES:
                try:
                    candidate_profile = resolve_runtime_profile(op_id, ep_id)
                    run_meta = get_run(run_id, db_name=candidate_profile.triplets_db_name)
                    if run_meta is not None:
                        resolved_db = candidate_profile.triplets_db_name
                        break
                except ValueError:
                    continue
            if run_meta is not None:
                break

    if run_meta is None:
        raise ValueError(f"Run not found: {run_id}")

    input_text = run_meta.get("input_text", "")
    if not input_text:
        raise ValueError(
            f"Run '{run_id}' has no input_text; replay requires a stored input."
        )

    # Restore original profile — do NOT silently switch profiles
    original_profile = _restore_profile_from_run(run_meta)

    overrides = overrides or {}
    override_ontology_profile_id = overrides.get("ontology_profile_id")
    override_embedding_profile_id = overrides.get("embedding_profile_id")

    if (override_ontology_profile_id is None) ^ (override_embedding_profile_id is None):
        raise ValueError(
            "Explicit profile override requires both 'ontology_profile_id' and "
            "'embedding_profile_id'."
        )

    if override_ontology_profile_id and override_embedding_profile_id:
        from ..profiles import resolve_runtime_profile
        original_profile = resolve_runtime_profile(
            override_ontology_profile_id,
            override_embedding_profile_id,
        )

    model     = overrides.get("model") or run_meta.get("model", "google/gemini-2.5-flash-lite")
    sample_id = overrides.get("sample_id") or run_meta.get("sample_id", "replay")

    extra_config = dict(run_meta.get("extra_config") or {})
    extra_config.update({
        k: v
        for k, v in overrides.items()
        if k not in ("model", "sample_id", "ontology_profile_id", "embedding_profile_id")
    })
    source_text_id = extra_config.get("source_text_id")

    # Lazy imports to avoid circular dependencies and defer heavy model loading.
    from pymongo import MongoClient
    from .openai_utils import LLMTripletExtractor
    from .structured_aligner import Aligner
    from .structured_inference_with_db import StructuredInferenceWithDB

    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    api_key   = os.environ.get("KEY")
    proxy_url = os.environ.get("PROXY_URL")

    mongo_client = MongoClient(mongo_uri)
    ontology_db  = mongo_client.get_database(original_profile.ontology_db_name)
    triplets_db  = mongo_client.get_database(original_profile.triplets_db_name)

    extractor = LLMTripletExtractor(model=model, api_key=api_key, proxy=proxy_url)
    aligner   = Aligner(
        ontology_db=ontology_db,
        triplets_db=triplets_db,
        embedding_model_name=original_profile.embedding_model_name,
    )
    inference = StructuredInferenceWithDB(
        extractor=extractor,
        aligner=aligner,
        triplets_db=triplets_db,
        runtime_profile=original_profile,
    )

    (
        _initial,
        _final,
        _filtered,
        _ontology_filtered,
        new_run_id,
    ) = inference.extract_triplets_with_ontology_filtering_and_add_to_db(
        text=input_text,
        sample_id=sample_id,
        source_text_id=source_text_id,
    )

    try:
        triplets_db["extraction_runs"].update_one(
            {"_id": new_run_id},
            {"$set": {"parent_run_id": run_id}},
        )
    except Exception:
        pass  # failure here is non-fatal; the run itself completed successfully

    return new_run_id
