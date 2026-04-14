# init_triplets_only.py
import argparse
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid

from src.wikontic.profiles import ONTOLOGY_PROFILES, EMBEDDING_PROFILES, resolve_runtime_profile
from src.wikontic.create_ontological_triplets_db import create_ontological_triplets_database

load_dotenv(find_dotenv())
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27018/?directConnection=true")


def resolve_profile_from_runtime_id(profile_id: str):
    # profile_id format: "{runtime_key}__{embedding_key}"  e.g. en__mft_random
    for op_id, op in ONTOLOGY_PROFILES.items():
        for ep_id, ep in EMBEDDING_PROFILES.items():
            candidate_id = f"{op.runtime_key}__{ep.embedding_key}"
            if candidate_id == profile_id:
                return resolve_runtime_profile(op_id, ep_id)

    known = [f"{op.runtime_key}__{ep.embedding_key}" for op in ONTOLOGY_PROFILES.values() for ep in EMBEDDING_PROFILES.values()]
    print(f"ERROR: Unknown profile_id '{profile_id}'.")
    print("Known profiles:", ", ".join(sorted(known)))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Initialize ONLY triplets DB for a runtime profile")
    parser.add_argument("--profile", type=str, default="en__mft_random")
    parser.add_argument("--drop_triplets", action="store_true", help="Drop and recreate triplets DB")
    args = parser.parse_args()

    profile = resolve_profile_from_runtime_id(args.profile)
    client = MongoClient(mongo_uri)
    triplets_db = client[profile.triplets_db_name]

    required_collections = {
        "triplets",
        profile.entity_aliases_collection_name,
        "initial_triplets",
        "filtered_triplets",
        "ontology_filtered_triplets",
    }

    existing = set(triplets_db.list_collection_names())

    print(f"Profile: {profile.profile_id}")
    print(f"Triplets DB: {profile.triplets_db_name}")
    print(f"Entity aliases collection: {profile.entity_aliases_collection_name}")
    print(f"Embedding model: {profile.embedding_model_name}")
    print(f"Embedding dimension: {profile.embedding_dimension}")

    if not args.drop_triplets and required_collections.issubset(existing):
        print("Triplets DB already initialized for this profile. Skipping collection creation.")
    else:
        try:
            create_ontological_triplets_database(
                mongo_uri=mongo_uri,
                db_name=profile.triplets_db_name,
                entity_aliases_collection=profile.entity_aliases_collection_name,
                entity_aliases_index=profile.entity_aliases_vector_index_name,
                drop_collections=args.drop_triplets,
                embedding_dimension=profile.embedding_dimension,
            )
            print("Triplets collections + indexes created.")
        except CollectionInvalid as e:
            print(f"Collection already exists, skipped safely: {e}")

    # profile metadata (strict readiness checks için önemli)
    profile_metadata = {
        "profile_id": profile.profile_id,
        "ontology_profile_id": profile.ontology_profile_id,
        "embedding_profile_id": profile.embedding_profile_id,
        "ontology_db_name": profile.ontology_db_name,
        "triplets_db_name": profile.triplets_db_name,
        "ontology_language": profile.ontology_language,
        "embedding_model_name": profile.embedding_model_name,
        "embedding_dimension": profile.embedding_dimension,
        "build_version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    triplets_db["system_profile_metadata"].replace_one(
        {"profile_id": profile.profile_id},
        profile_metadata,
        upsert=True,
    )
    print("system_profile_metadata updated.")
    print("Done.")


if __name__ == "__main__":
    main()
