"""
profile_readiness.py

Validates that a runtime profile's databases and indexes are correctly initialized
before extraction is allowed to start.

No silent fallback is performed. If a profile is not ready, the caller receives
a ReadinessResult with ready=False and a list of specific issues.
"""
from dataclasses import dataclass, field

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


def check_profile_readiness(
    profile: RuntimeProfile,
    mongo_client: MongoClient,
) -> ReadinessResult:
    """
    Validate that the given runtime profile has its DBs correctly initialized.

    Checks:
    1. Ontology DB exists.
    2. Required ontology collections exist.
    3. system_profile_metadata document exists and embedding_dimension matches.
    4. Triplets DB exists.
    5. Required triplets collections exist.

    Returns ReadinessResult. Never raises; all errors are captured as issues.
    """
    issues: list[str] = []

    try:
        db_names = mongo_client.list_database_names()
    except Exception as e:
        return ReadinessResult(
            ready=False,
            profile_id=profile.profile_id,
            issues=[f"Cannot connect to MongoDB: {e}"],
        )

    # ── Ontology DB ───────────────────────────────────────────────────────────
    if profile.ontology_db_name not in db_names:
        issues.append(
            f"Ontology DB not found: '{profile.ontology_db_name}'. "
            f"Initialize with: python init_dbs.py --profile {profile.profile_id}"
        )
        return ReadinessResult(ready=False, profile_id=profile.profile_id, issues=issues)

    ontology_db = mongo_client[profile.ontology_db_name]
    existing_ontology_cols = set(ontology_db.list_collection_names())

    required_ontology_cols = {
        "entity_types",
        "entity_type_aliases",
        "properties",
        "property_aliases",
        "system_profile_metadata",
    }
    missing_ontology = required_ontology_cols - existing_ontology_cols
    if missing_ontology:
        issues.append(
            f"Missing collections in ontology DB '{profile.ontology_db_name}': "
            f"{sorted(missing_ontology)}"
        )

    # Validate embedding dimension in stored metadata
    if "system_profile_metadata" in existing_ontology_cols:
        meta = ontology_db["system_profile_metadata"].find_one(
            {"profile_id": profile.profile_id}
        )
        if meta is None:
            issues.append(
                f"system_profile_metadata has no document for "
                f"profile_id='{profile.profile_id}'"
            )
        elif meta.get("embedding_dimension") != profile.embedding_dimension:
            issues.append(
                f"Embedding dimension mismatch in '{profile.ontology_db_name}': "
                f"stored={meta.get('embedding_dimension')}, "
                f"expected={profile.embedding_dimension}"
            )

    # ── Triplets DB ───────────────────────────────────────────────────────────
    if profile.triplets_db_name not in db_names:
        issues.append(
            f"Triplets DB not found: '{profile.triplets_db_name}'. "
            f"Initialize with: python init_dbs.py --profile {profile.profile_id}"
        )
        return ReadinessResult(ready=False, profile_id=profile.profile_id, issues=issues)

    triplets_db = mongo_client[profile.triplets_db_name]
    existing_triplets_cols = set(triplets_db.list_collection_names())

    required_triplets_cols = {
        "entity_aliases",
        "property_aliases",
        "triplets",
        "initial_triplets",
        "filtered_triplets",
        "ontology_filtered_triplets",
    }
    missing_triplets = required_triplets_cols - existing_triplets_cols
    if missing_triplets:
        issues.append(
            f"Missing collections in triplets DB '{profile.triplets_db_name}': "
            f"{sorted(missing_triplets)}"
        )

    return ReadinessResult(
        ready=len(issues) == 0,
        profile_id=profile.profile_id,
        issues=issues,
    )
