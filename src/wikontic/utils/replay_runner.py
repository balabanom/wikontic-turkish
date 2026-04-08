"""
replay_runner.py

Re-executes the extraction pipeline for a given run_id using its stored input and config.
Creates a new run_id linked to the original via parent_run_id.

Usage:
    new_run_id = replay_run("some-uuid", overrides={"model": "gpt-4.1"})
"""
import os
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient

from .run_reader import get_run

_ = load_dotenv(find_dotenv())


def replay_run(
    run_id: str,
    overrides: Optional[dict] = None,
) -> str:
    """
    Re-run the extraction pipeline using the input_text and config of an existing run.

    Args:
        run_id:    ID of the original run to replay.
        overrides: Optional overrides, e.g. {"model": "gpt-4.1", "sample_id": "..."}.

    Returns:
        new_run_id (str)

    Raises:
        ValueError: if the run is not found or has no input_text.
        Exception:  on pipeline failure (run is marked FAILED).
    """
    run_meta = get_run(run_id)
    if run_meta is None:
        raise ValueError(f"Run not found: {run_id}")

    input_text = run_meta.get("input_text", "")
    if not input_text:
        raise ValueError(
            f"Run '{run_id}' has no input_text; replay requires a stored input."
        )

    overrides = overrides or {}
    model = overrides.get("model") or run_meta.get("model", "google/gemini-2.5-flash-lite")
    sample_id = overrides.get("sample_id") or run_meta.get("sample_id", "replay")

    extra_config = dict(run_meta.get("extra_config") or {})
    extra_config.update({k: v for k, v in overrides.items() if k not in ("model", "sample_id")})
    source_text_id = extra_config.get("source_text_id")

    # Lazy imports to avoid circular dependencies and defer heavy model loading.
    from pymongo import MongoClient
    from .openai_utils import LLMTripletExtractor
    from .structured_aligner import Aligner
    from .structured_inference_with_db import StructuredInferenceWithDB

    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    api_key = os.environ.get("KEY")
    proxy_url = os.environ.get("PROXY_URL")

    mongo_client = MongoClient(mongo_uri)
    ontology_db = mongo_client.get_database("wikidata_ontology")
    triplets_db = mongo_client.get_database("demo")

    extractor = LLMTripletExtractor(
        model=model,
        api_key=api_key,
        proxy=proxy_url,
    )
    aligner = Aligner(ontology_db=ontology_db, triplets_db=triplets_db)
    inference = StructuredInferenceWithDB(
        extractor=extractor,
        aligner=aligner,
        triplets_db=triplets_db,
    )

    # parent_run_id is patched into the run document after creation because
    # extract_triplets_with_ontology_filtering_and_add_to_db calls start_run
    # internally with no parent_run_id hook. Updating post-creation is the
    # least invasive approach without modifying the pipeline signature.
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