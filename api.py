"""
api.py — Wikontic FastAPI

Extracts knowledge graph triplets from text using the full ontology pipeline
but WITHOUT writing anything to the database.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

POST /extract
    Body: { "text": "...", "embedding_model": "contriever", "llm_model": "google/gemini-2.5-flash-lite" }
    Returns: { "triplets": [...], "count": N }
"""

import os
import logging
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pymongo import MongoClient

from src.wikontic.llm_models import DEFAULT_LLM_MODEL

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

_LANG_TO_ONTOLOGY_PROFILE = {
    "en": "ontology_en_v1",
    "tr": "ontology_tr_v1",
}


def _resolve_profile(embedding_key: str, ontology_language: str = "en"):
    """
    Resolve RuntimeProfile for given ontology language + embedding key.
    Raises ValueError for unknown embedding_key or ontology_language.
    """
    from src.wikontic.profiles import EMBEDDING_PROFILES, resolve_runtime_profile

    ontology_profile_id = _LANG_TO_ONTOLOGY_PROFILE.get(ontology_language)
    if ontology_profile_id is None:
        raise ValueError(
            f"Unknown ontology_language '{ontology_language}'. "
            f"Known: {list(_LANG_TO_ONTOLOGY_PROFILE)}"
        )

    matched_ep = next(
        (ep for ep in EMBEDDING_PROFILES.values() if ep.embedding_key == embedding_key),
        None,
    )
    if matched_ep is None:
        known = [ep.embedding_key for ep in EMBEDDING_PROFILES.values()]
        raise ValueError(
            f"Unknown embedding_model '{embedding_key}'. Known keys: {known}"
        )

    return resolve_runtime_profile(ontology_profile_id, matched_ep.profile_id)


# ── Aligner cache (one per (embedding, language) — model loading is expensive) ─

@lru_cache(maxsize=8)
def _get_aligner(embedding_key: str, ontology_language: str = "en"):
    from src.wikontic.utils.structured_aligner import Aligner

    profile = _resolve_profile(embedding_key, ontology_language)
    client  = _get_mongo()
    return Aligner(
        ontology_db=client.get_database(profile.ontology_db_name),
        triplets_db=client.get_database(profile.triplets_db_name),
        embedding_model_name=profile.embedding_model_name,
        runtime_profile=profile,
    )


# ── Request / Response schemas ────────────────────────────────────────────────

class ExtractionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    text: str
    embedding_model: str = "contriever"   # embedding_key
    llm_model: str = Field(
        default=DEFAULT_LLM_MODEL,
        validation_alias=AliasChoices("llm_model", "model"),
    )
    ontology_language: str = "en"         # "en" or "tr"
    prompt_type: str = "temel"            # "temel" | "ape" | "dspy" | "textgrad"


class Qualifier(BaseModel):
    relation: str = ""
    object: str = ""


class Triplet(BaseModel):
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    qualifiers: list[Qualifier] = []
    kaynak_cumle: str = ""


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
        "POST /extract  →  embedding_model=%s  llm_model=%s  ontology_language=%s  prompt_type=%s  text_len=%d  text_preview=%r",
        req.embedding_model,
        req.llm_model,
        req.ontology_language,
        req.prompt_type,
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
    llm_model       : LLM model name (e.g. "openai/gpt-4o-mini",
                      "google/gemini-2.5-flash-lite").

    Returns
    -------
    triplets : List of { subject, subject_type, relation, object, object_type }
    count    : Number of final triplets.
    """
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    from prompts.dispatcher import is_valid_prompt_type
    if not is_valid_prompt_type(req.prompt_type):
        raise HTTPException(
            status_code=422,
            detail=f"Unknown prompt_type '{req.prompt_type}'. Known: temel, ape, dspy, textgrad",
        )

    _check_mongo()

    # Resolve profile
    try:
        profile = _resolve_profile(req.embedding_model, req.ontology_language)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Build extractor
    from src.wikontic.utils.openai_utils import LLMTripletExtractor
    extractor = LLMTripletExtractor(
        model=req.llm_model,
        api_key=_API_KEY,
        proxy=_PROXY_URL,
        prompt_type=req.prompt_type,
    )

    # Get cached aligner for this (embedding, language) combo
    try:
        aligner = _get_aligner(req.embedding_model, req.ontology_language)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load aligner for '{req.embedding_model}' / '{req.ontology_language}': {e}",
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

    def _normalise_qualifiers(raw) -> list[Qualifier]:
        if not isinstance(raw, list):
            return []
        out: list[Qualifier] = []
        for q in raw:
            if isinstance(q, dict):
                out.append(Qualifier(
                    relation=str(q.get("relation", "")),
                    object=str(q.get("object", "")),
                ))
        return out

    result = [
        Triplet(
            subject=str(t.get("subject", "")),
            subject_type=str(t.get("subject_type", "")),
            relation=str(t.get("relation", "")),
            object=str(t.get("object", "")),
            object_type=str(t.get("object_type", "")),
            qualifiers=_normalise_qualifiers(t.get("qualifiers")),
            kaynak_cumle=str(t.get("kaynak_cumle", "")),
        )
        for t in final_triplets
    ]

    logging.info("POST /extract  ←  returned %d triplets", len(result))
    return ExtractionResponse(triplets=result, count=len(result))
