"""
profile_readiness.py

Validates that a runtime profile's databases and indexes are correctly initialized
before extraction is allowed to start.

No silent fallback is performed. If a profile is not ready, the caller receives
a ReadinessResult with ready=False and a list of specific issues.
"""
from dataclasses import dataclass, field, replace as dc_replace

from pymongo import MongoClient

from .profiles import RuntimeProfile


@dataclass
class ReadinessResult:
    ready: bool
    profile_id: str
    issues: list = field(default_factory=list)

    @property
    def error_message(self) -> str:
        return "\n".join(self.issues) if self.issues else ""


def _append_profile_metadata_issues(
    *,
    issues: list[str],
    profile: RuntimeProfile,
    meta: dict | None,
    db_name: str,
    source: str,
) -> None:
    if meta is None:
        issues.append(
            f"{source}.system_profile_metadata has no document for "
            f"profile_id='{profile.profile_id}' in DB '{db_name}'"
        )
        return
    if meta.get("embedding_dimension") != profile.embedding_dimension:
        issues.append(
            f"Embedding dimension mismatch in {source} DB '{db_name}': "
            f"stored={meta.get('embedding_dimension')}, expected={profile.embedding_dimension}"
        )
    if meta.get("embedding_model_name") != profile.embedding_model_name:
        issues.append(
            f"Embedding model mismatch in {source} DB '{db_name}': "
            f"stored={meta.get('embedding_model_name')}, expected={profile.embedding_model_name}"
        )
    if meta.get("embedding_profile_id") != profile.embedding_profile_id:
        issues.append(
            f"Embedding profile mismatch in {source} DB '{db_name}': "
            f"stored={meta.get('embedding_profile_id')}, expected={profile.embedding_profile_id}"
        )


def _validate_vector_index_exists(
    *,
    issues: list[str],
    db_name: str,
    collection,
    collection_name: str,
    index_name: str,
) -> None:
    try:
        names = {idx.get("name") for idx in collection.list_search_indexes()}
    except Exception as e:
        issues.append(
            f"Cannot validate vector index '{index_name}' on "
            f"'{db_name}.{collection_name}': {e}"
        )
        return
    if index_name not in names:
        issues.append(
            f"Missing vector index '{index_name}' on '{db_name}.{collection_name}'"
        )


def _extract_vector_dimensions(index_doc: dict) -> list[int]:
    dims: list[int] = []

    def _walk(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "knnVector" and isinstance(obj.get("dimensions"), int):
                dims.append(obj["dimensions"])
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(index_doc)
    return dims


def _validate_vector_index_dimension(
    *,
    issues: list[str],
    db_name: str,
    collection,
    collection_name: str,
    index_name: str,
    expected_dimension: int,
) -> None:
    try:
        index_docs = list(collection.list_search_indexes())
    except Exception:
        return

    target = next((d for d in index_docs if d.get("name") == index_name), None)
    if target is None:
        return
    dims = _extract_vector_dimensions(target)
    if not dims:
        issues.append(
            f"Cannot verify vector dimensions for index '{index_name}' on "
            f"'{db_name}.{collection_name}'"
        )
        return
    if expected_dimension not in dims:
        issues.append(
            f"Vector dimension mismatch for index '{index_name}' on "
            f"'{db_name}.{collection_name}': stored={sorted(set(dims))}, "
            f"expected={expected_dimension}"
        )


def check_profile_readiness(
    profile: RuntimeProfile,
    mongo_client: MongoClient,
    relax_ontology_metadata: bool = False,
    relax_triplets_metadata: bool = False,
) -> ReadinessResult:
    """
    Validate that the given runtime profile has its DBs correctly initialized.

    Checks:
    1. Ontology DB exists.
    2. Required ontology collections exist.
    3. If required by profile, system_profile_metadata exists and matches embedding_dimension.
    4. Triplets DB exists.
    5. Required triplets collections exist.

    Returns ReadinessResult. Never raises; all errors are captured as issues.
    """
    issues: list[str] = []
    required_triplets_cols = {
        "entity_aliases",
        "triplets",
        "initial_triplets",
        "filtered_triplets",
        "ontology_filtered_triplets",
    }

    try:
        db_names = mongo_client.list_database_names()
    except Exception as e:
        return ReadinessResult(
            ready=False,
            profile_id=profile.profile_id,
            issues=[f"Cannot connect to MongoDB: {e}"],
        )

    # Work on local copies — never mutate the caller's profile object.
    ontology_db_name = profile.ontology_db_name
    triplets_db_name = profile.triplets_db_name

    # ── Ontology DB ───────────────────────────────────────────────────────────
    if profile.profile_id == "en_legacy__contriever" and ontology_db_name not in db_names:
        # Legacy deployments may use different DB names.
        for candidate in ("wikidata_ontology", "wikontic_ontology"):
            if candidate in db_names:
                ontology_db_name = candidate
                break

    if ontology_db_name not in db_names:
        issues.append(
            f"Ontology DB not found: '{ontology_db_name}'. "
            f"Initialize with: python init_dbs.py --profile {profile.profile_id}"
        )
        return ReadinessResult(ready=False, profile_id=profile.profile_id, issues=issues)

    ontology_db = mongo_client[ontology_db_name]
    existing_ontology_cols = set(ontology_db.list_collection_names())

    required_ontology_cols = {
        "entity_types",
        "entity_type_aliases",
        "properties",
        "property_aliases",
    }
    if profile.requires_system_profile_metadata and not relax_ontology_metadata:
        required_ontology_cols.add("system_profile_metadata")
    missing_ontology = required_ontology_cols - existing_ontology_cols
    if missing_ontology:
        issues.append(
            f"Missing collections in ontology DB '{ontology_db_name}': "
            f"{sorted(missing_ontology)}"
        )

    # Validate ontology metadata integrity for embedding compatibility
    if (
        profile.requires_system_profile_metadata
        and not relax_ontology_metadata
        and "system_profile_metadata" in existing_ontology_cols
    ):
        meta = ontology_db["system_profile_metadata"].find_one(
            {"profile_id": profile.profile_id}
        )
        _append_profile_metadata_issues(
            issues=issues,
            profile=profile,
            meta=meta,
            db_name=ontology_db_name,
            source="ontology",
        )

    # ── Triplets DB ───────────────────────────────────────────────────────────
    if profile.profile_id == "en_legacy__contriever":
        # Legacy mode: pick the most likely historical workspace DB automatically.
        # Priority: user-selected/default name -> known legacy names -> ontology DB.
        candidate_dbs: list[str] = [
            triplets_db_name,
            "demo",
            "wikontic_ontology",
            "triplets__en__contriever",
        ]
        seen: set[str] = set()
        candidate_dbs = [d for d in candidate_dbs if d and not (d in seen or seen.add(d))]

        preferred = None
        fallback = None
        for candidate in candidate_dbs:
            if candidate not in db_names:
                continue
            cols = set(mongo_client[candidate].list_collection_names())
            if "triplets" in cols and "extraction_runs" in cols:
                preferred = candidate
                break
            if "triplets" in cols and fallback is None:
                fallback = candidate

        chosen = preferred or fallback
        if chosen:
            triplets_db_name = chosen

    if triplets_db_name not in db_names:
        issues.append(
            f"Triplets DB not found: '{triplets_db_name}'. "
            f"Initialize with: python init_dbs.py --profile {profile.profile_id}"
        )
        return ReadinessResult(ready=False, profile_id=profile.profile_id, issues=issues)

    if triplets_db_name == ontology_db_name:
        issues.append(
            "Triplets DB and Ontology DB must be different. "
            f"Both are '{triplets_db_name}'."
        )
        return ReadinessResult(ready=False, profile_id=profile.profile_id, issues=issues)

    triplets_db = mongo_client[triplets_db_name]
    existing_triplets_cols = set(triplets_db.list_collection_names())

    missing_triplets = required_triplets_cols - existing_triplets_cols
    if missing_triplets:
        issues.append(
            f"Missing collections in triplets DB '{triplets_db_name}': "
            f"{sorted(missing_triplets)}"
        )

    # Validate triplets metadata integrity for embedding compatibility
    if profile.requires_system_profile_metadata and not relax_triplets_metadata:
        if "system_profile_metadata" not in existing_triplets_cols:
            issues.append(
                f"Missing collection in triplets DB '{triplets_db_name}': "
                "['system_profile_metadata']"
            )
        else:
            triplets_meta = triplets_db["system_profile_metadata"].find_one(
                {"profile_id": profile.profile_id}
            )
            _append_profile_metadata_issues(
                issues=issues,
                profile=profile,
                meta=triplets_meta,
                db_name=triplets_db_name,
                source="triplets",
            )

    # Validate required vector indexes for embedding-based retrieval.
    if profile.requires_system_profile_metadata:
        _validate_vector_index_exists(
            issues=issues,
            db_name=ontology_db_name,
            collection=ontology_db["entity_type_aliases"],
            collection_name="entity_type_aliases",
            index_name=profile.entity_type_vector_index_name,
        )
        _validate_vector_index_dimension(
            issues=issues,
            db_name=ontology_db_name,
            collection=ontology_db["entity_type_aliases"],
            collection_name="entity_type_aliases",
            index_name=profile.entity_type_vector_index_name,
            expected_dimension=profile.embedding_dimension,
        )
        _validate_vector_index_exists(
            issues=issues,
            db_name=ontology_db_name,
            collection=ontology_db["property_aliases"],
            collection_name="property_aliases",
            index_name=profile.property_vector_index_name,
        )
        _validate_vector_index_dimension(
            issues=issues,
            db_name=ontology_db_name,
            collection=ontology_db["property_aliases"],
            collection_name="property_aliases",
            index_name=profile.property_vector_index_name,
            expected_dimension=profile.embedding_dimension,
        )
        _validate_vector_index_exists(
            issues=issues,
            db_name=triplets_db_name,
            collection=triplets_db["entity_aliases"],
            collection_name="entity_aliases",
            index_name=profile.entity_aliases_vector_index_name,
        )
        _validate_vector_index_dimension(
            issues=issues,
            db_name=triplets_db_name,
            collection=triplets_db["entity_aliases"],
            collection_name="entity_aliases",
            index_name=profile.entity_aliases_vector_index_name,
            expected_dimension=profile.embedding_dimension,
        )

    return ReadinessResult(
        ready=len(issues) == 0,
        profile_id=profile.profile_id,
        issues=issues,
    )
