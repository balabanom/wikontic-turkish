"""
api.py — Wikontic FastAPI

Extracts knowledge graph triplets from text using the full ontology pipeline
but WITHOUT writing anything to the database.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

POST /extract
    Body: { "text": "...", "embedding_model": "contriever", "llm_model": "gpt-4o-mini" }
    Returns: { "triplets": [...], "count": N }
"""

import os
import logging
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient

load_dotenv(find_dotenv())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="Wikontic Extraction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_API_KEY   = os.getenv("KEY")
_PROXY_URL = os.getenv("PROXY_URL")

_mongo_client: Optional[MongoClient] = None


def _get_mongo() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(_MONGO_URI, serverSelectionTimeoutMS=3000)
    return _mongo_client


def _check_mongo():
    """Raise HTTPException 503 if MongoDB is unreachable."""
    try:
        _get_mongo().admin.command("ping")
    except Exception as e:
        logging.error("MongoDB unreachable: %s", e)
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not running. Start it (e.g. docker-compose up -d) and retry.",
        )


# ── Profile resolution ────────────────────────────────────────────────────────

def _resolve_profile(embedding_key: str):
    """
    Resolve RuntimeProfile for English ontology + given embedding key.
    Raises ValueError for unknown embedding_key.
    """
    from src.wikontic.profiles import EMBEDDING_PROFILES, resolve_runtime_profile

    matched_ep = next(
        (ep for ep in EMBEDDING_PROFILES.values() if ep.embedding_key == embedding_key),
        None,
    )
    if matched_ep is None:
        known = [ep.embedding_key for ep in EMBEDDING_PROFILES.values()]
        raise ValueError(
            f"Unknown embedding_model '{embedding_key}'. Known keys: {known}"
        )

    return resolve_runtime_profile("ontology_en_v1", matched_ep.profile_id)


# ── Aligner cache (one per embedding model — model loading is expensive) ──────

@lru_cache(maxsize=8)
def _get_aligner(embedding_key: str):
    from src.wikontic.utils.structured_aligner import Aligner

    profile = _resolve_profile(embedding_key)
    client  = _get_mongo()
    return Aligner(
        ontology_db=client.get_database(profile.ontology_db_name),
        triplets_db=client.get_database(profile.triplets_db_name),
        embedding_model_name=profile.embedding_model_name,
        runtime_profile=profile,
    )


# ── Request / Response schemas ────────────────────────────────────────────────

class ExtractionRequest(BaseModel):
    text: str
    embedding_model: str = "contriever"   # embedding_key
    llm_model: str = "gpt-4o-mini"


class Triplet(BaseModel):
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str


class ExtractionResponse(BaseModel):
    triplets: list[Triplet]
    count: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def healthcheck():
    logging.info("GET /  →  healthcheck")
    try:
        _get_mongo().admin.command("ping")
        db_status = "ok"
    except Exception:
        db_status = "unreachable"
    return {"status": "ok", "service": "Wikontic Extraction API", "mongodb": db_status}


@app.post("/extract", response_model=ExtractionResponse)
def extract(req: ExtractionRequest):
    logging.info(
        "POST /extract  →  embedding_model=%s  llm_model=%s  text_len=%d  text_preview=%r",
        req.embedding_model,
        req.llm_model,
        len(req.text),
        req.text[:120],
    )
    """
    Extract knowledge graph triplets from text through the full ontology
    pipeline (entity type refinement, relation refinement, validation)
    WITHOUT writing anything to the database.

    Parameters
    ----------
    text            : Input paragraph to extract from.
    embedding_model : Embedding model key (e.g. "contriever", "bge_m3",
                      "turkish_e5_large", "mft_random").
    llm_model       : LLM model name (e.g. "gpt-4o-mini",
                      "google/gemini-2.5-flash-lite").

    Returns
    -------
    triplets : List of { subject, subject_type, relation, object, object_type }
    count    : Number of final triplets.
    """
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    _check_mongo()

    # Resolve profile
    try:
        profile = _resolve_profile(req.embedding_model)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Build extractor
    from src.wikontic.utils.openai_utils import LLMTripletExtractor
    extractor = LLMTripletExtractor(
        model=req.llm_model,
        api_key=_API_KEY,
        proxy=_PROXY_URL,
    )

    # Get cached aligner for this embedding model
    try:
        aligner = _get_aligner(req.embedding_model)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load aligner for '{req.embedding_model}': {e}",
        )

    # Run pipeline — no DB writes
    from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
    pipeline = StructuredInferenceWithDB(
        extractor=extractor,
        aligner=aligner,
        triplets_db=None,           # no DB needed — we call the no-write method
        runtime_profile=profile,
    )

    try:
        _, final_triplets, _, _ = pipeline.extract_triplets_with_ontology_filtering(
            text=req.text,
            sample_id="api_preview",
            source_text_id=None,
            run_id=None,            # no run logging
            timer=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    result = [
        Triplet(
            subject=str(t.get("subject", "")),
            subject_type=str(t.get("subject_type", "")),
            relation=str(t.get("relation", "")),
            object=str(t.get("object", "")),
            object_type=str(t.get("object_type", "")),
        )
        for t in final_triplets
    ]

    logging.info("POST /extract  ←  returned %d triplets", len(result))
    return ExtractionResponse(triplets=result, count=len(result))
