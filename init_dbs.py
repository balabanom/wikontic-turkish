"""
init_dbs.py

Initializes the ontology DB and triplets DB for a given runtime profile.

Usage:
    # Default: English + Contriever
    python init_dbs.py

    # Explicit profile
    python init_dbs.py --profile en__contriever

    # When Turkish data is ready
    python init_dbs.py --profile tr__turkish_e5_large

The --profile argument must match a profile_id resolvable from the registry.
No silent fallback: if the profile is unknown, the script exits with an error.
"""

import argparse
import os
import sys

from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid

load_dotenv(find_dotenv())

mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27018/?directConnection=true")


def _resolve_profile(profile_id: str):
    """Resolve a RuntimeProfile from a profile_id string like 'en__contriever'."""
    from src.wikontic.profiles import (
        ONTOLOGY_PROFILES,
        EMBEDDING_PROFILES,
        resolve_runtime_profile,
    )

    # profile_id is "{runtime_key}__{embedding_key}" — derive profile IDs from registry
    for op_id, op in ONTOLOGY_PROFILES.items():
        for ep_id, ep in EMBEDDING_PROFILES.items():
            candidate_id = f"{op.runtime_key}__{ep.embedding_key}"
            if candidate_id == profile_id:
                return resolve_runtime_profile(op_id, ep_id)

    known = []
    for op in ONTOLOGY_PROFILES.values():
        for ep in EMBEDDING_PROFILES.values():
            known.append(f"{op.runtime_key}__{ep.embedding_key}")

    print(f"ERROR: Unknown profile_id '{profile_id}'.")
    print(f"Known profiles: {', '.join(known)}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Initialize ontology + triplets DBs for a runtime profile"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="en__contriever",
        help="Runtime profile ID (default: en__contriever)",
    )
    parser.add_argument(
        "--drop_triplets",
        action="store_true",
        default=False,
        help="Drop and recreate the triplets DB (WARNING: deletes user KG data)",
    )
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)

    print(f"\n{'='*60}")
    print(f"  Runtime Profile: {profile.profile_id}")
    print(f"  Display Name:    {profile.display_name}")
    print(f"  Ontology DB:     {profile.ontology_db_name}")
    print(f"  Triplets DB:     {profile.triplets_db_name}")
    print(f"  Embedding Model: {profile.embedding_model_name}")
    print(f"  Dimension:       {profile.embedding_dimension}")
    print(f"{'='*60}\n")

    from src.wikontic.create_wikidata_ontology_db import create_wikidata_ontology_database
    from src.wikontic.create_ontological_triplets_db import create_ontological_triplets_database

    client = MongoClient(mongo_uri)

    # ── 1) Ontology DB (rebuild from scratch) ────────────────────────────────
    print(f"[1/2] Building ontology DB: {profile.ontology_db_name} ...")

    profile_metadata = {
        "profile_id":           profile.profile_id,
        "ontology_profile_id":  profile.ontology_profile_id,
        "embedding_profile_id": profile.embedding_profile_id,
        "ontology_db_name":     profile.ontology_db_name,
        "triplets_db_name":     profile.triplets_db_name,
        "ontology_language":    profile.ontology_language,
        "embedding_model_name": profile.embedding_model_name,
        "embedding_dimension":  profile.embedding_dimension,
        "build_version":        "v1",
    }

    create_wikidata_ontology_database(
        mongo_uri=mongo_uri,
        database=profile.ontology_db_name,
        drop_collections=True,
        model_name=profile.embedding_model_name,
        embedding_dimension=profile.embedding_dimension,
        profile_metadata=profile_metadata,
    )
    print(f"✅ Ontology DB '{profile.ontology_db_name}' ready.\n")

    # ── 2) Triplets DB (preserve if exists, unless --drop_triplets) ───────────
    print(f"[2/2] Setting up triplets DB: {profile.triplets_db_name} ...")

    triplets_db = client[profile.triplets_db_name]
    required_collections = {
        "triplets",
        "entity_aliases",
        "property_aliases",
        "entity_types",
        "properties",
        "initial_triplets",
        "filtered_triplets",
        "ontology_filtered_triplets",
    }
    existing = set(triplets_db.list_collection_names())

    if not args.drop_triplets and required_collections.issubset(existing):
        print(
            f"✅ Triplets DB '{profile.triplets_db_name}' already has all required "
            f"collections. Skipping init (user KG data preserved).\n"
        )
    else:
        try:
            create_ontological_triplets_database(
                mongo_uri=mongo_uri,
                db_name=profile.triplets_db_name,
                drop_collections=args.drop_triplets,
                embedding_dimension=profile.embedding_dimension,
            )
            print(f"✅ Triplets DB '{profile.triplets_db_name}' ready.\n")
        except CollectionInvalid as e:
            print(
                f"⚠️  Some triplets collections already exist: {e}\n"
                f"✅ Triplets DB init skipped safely (data preserved).\n"
            )

    print(f"✅ Profile '{profile.profile_id}' fully initialized.")


if __name__ == "__main__":
    main()
