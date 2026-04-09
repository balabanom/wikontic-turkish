from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
import pymongo

from typing import List, Callable, Optional
from pydantic import BaseModel, ValidationError
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import time
import argparse
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
import torch
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_embedding_model(model_name: str, device: torch.device):
    """Load tokenizer and model by name. Model is moved to device."""
    logger.info(f"Loading embedding model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device)
    logger.info(f"Embedding model loaded on {device}")
    return tokenizer, model


def _make_get_embedding(tokenizer, model, device) -> Callable[[str], Optional[List[float]]]:
    """Return a closure that embeds a single string using mean pooling."""

    def get_embedding(text: str) -> Optional[List[float]]:
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
            return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

        if not text or not isinstance(text, str):
            return None
        try:
            inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs.to(device))
            embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
            return embeddings.detach().cpu().tolist()[0]
        except Exception as e:
            logger.error(f"Error in get_embedding: {e}")
            return None

    return get_embedding


class EntityType(BaseModel):
    _id: int
    entity_type_id: str
    label: str
    parent_type_ids: List[str]
    valid_subject_property_ids: List[str]
    valid_object_property_ids: List[str]


class Property(BaseModel):
    _id: int
    property_id: str
    label: str
    valid_subject_type_ids: List[str]
    valid_object_type_ids: List[str]


class EntityTypeAlias(BaseModel):
    _id: int
    entity_type_id: str
    alias_label: str
    alias_text_embedding: List[float]


class PropertyAlias(BaseModel):
    _id: int
    relation_id: str
    alias_label: str
    alias_text_embedding: List[float]


def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    logger.info("Connection to MongoDB successful")
    return client


def populate_entity_types(
    ENTITY_2_LABEL,
    ENTITY_2_HIERARCHY,
    SUBJ_2_PROP_CONSTRAINTS,
    OBJ_2_PROP_CONSTRAINTS,
    db,
    collection_name="entity_types",
):
    logger.info(f"Starting to populate {collection_name} collection")
    entity_metadata_list = []

    for i, entity in enumerate(ENTITY_2_LABEL.keys()):
        label = ENTITY_2_LABEL[entity]
        parents = ENTITY_2_HIERARCHY[entity]

        valid_subject_property_ids = (
            SUBJ_2_PROP_CONSTRAINTS[entity] if entity in SUBJ_2_PROP_CONSTRAINTS else []
        )
        valid_object_property_ids = (
            OBJ_2_PROP_CONSTRAINTS[entity] if entity in OBJ_2_PROP_CONSTRAINTS else []
        )

        entity_metadata_list.append(
            {
                "_id": i,
                "entity_type_id": entity,
                "label": label,
                "parent_type_ids": parents,
                "valid_subject_property_ids": valid_subject_property_ids,
                "valid_object_property_ids": valid_object_property_ids,
            }
        )

    entity_metadata_list.append(
        {
            "_id": i + 1,
            "entity_type_id": "ANY",
            "label": "ANY",
            "parent_type_ids": [],
            "valid_subject_property_ids": SUBJ_2_PROP_CONSTRAINTS["<ANY SUBJECT>"],
            "valid_object_property_ids": OBJ_2_PROP_CONSTRAINTS["<ANY OBJECT>"],
        }
    )

    try:
        records = [EntityType(**record).model_dump() for record in entity_metadata_list]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_entity_type_aliases(
    ENTITY_2_LABEL,
    ENTITY_2_ALIASES,
    db,
    get_embedding: Callable,
    collection_name="entity_type_aliases",
):
    logger.info(f"Starting to populate {collection_name} collection")
    entity_types_list = []
    id_count = 0

    for e, aliases in tqdm(ENTITY_2_ALIASES.items()):
        alias_embedding = get_embedding(ENTITY_2_LABEL[e])
        entity_types_list.append(
            {
                "_id": id_count,
                "entity_type_id": e,
                "alias_label": ENTITY_2_LABEL[e],
                "alias_text_embedding": alias_embedding,
            }
        )
        id_count += 1

        for alias in aliases:
            alias_embedding = get_embedding(alias)
            entity_types_list.append(
                {
                    "_id": id_count,
                    "entity_type_id": e,
                    "alias_label": alias,
                    "alias_text_embedding": alias_embedding,
                }
            )
            id_count += 1
    try:
        records = [
            EntityTypeAlias(**record).model_dump() for record in entity_types_list
        ]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_properties(
    PROP_2_LABEL, PROP_2_CONSTRAINT, db, collection_name="properties"
):
    logger.info(f"Starting to populate {collection_name} collection")
    property_list = []

    for i, prop_id in enumerate(PROP_2_LABEL.keys()):
        property_list.append(
            {
                "_id": i,
                "property_id": prop_id,
                "label": PROP_2_LABEL[prop_id],
                "valid_subject_type_ids": PROP_2_CONSTRAINT[prop_id][
                    "Subject type constraint"
                ],
                "valid_object_type_ids": PROP_2_CONSTRAINT[prop_id][
                    "Value-type constraint"
                ],
            }
        )

    try:
        records = [Property(**record).model_dump() for record in property_list]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_property_aliases(
    PROP_2_LABEL,
    PROP_2_ALIASES,
    db,
    get_embedding: Callable,
    collection_name="property_aliases",
):
    logger.info(f"Starting to populate {collection_name} collection")
    relation_alias_id_pairs = []
    id_count = 0

    for r, aliases in tqdm(PROP_2_ALIASES.items()):
        alias_embedding = get_embedding(PROP_2_LABEL[r])
        relation_alias_id_pairs.append(
            {
                "_id": id_count,
                "relation_id": r,
                "alias_label": PROP_2_LABEL[r],
                "alias_text_embedding": alias_embedding,
            }
        )
        id_count += 1

        for alias in aliases:
            alias_embedding = get_embedding(alias)
            relation_alias_id_pairs.append(
                {
                    "_id": id_count,
                    "relation_id": r,
                    "alias_label": alias,
                    "alias_text_embedding": alias_embedding,
                }
            )
            id_count += 1
    try:
        records = [
            PropertyAlias(**record).model_dump() for record in relation_alias_id_pairs
        ]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def create_search_index_for_entity_types(
    db,
    collection_name="entity_type_aliases",
    embedding_field_name="alias_text_embedding",
    index_name="entity_type_aliases",
    dimensions=768,
):
    logger.info(f"Starting to create index {index_name} for {collection_name}")
    collection = db.get_collection(collection_name)
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    embedding_field_name: {
                        "dimensions": dimensions,
                        "similarity": "cosine",
                        "type": "knnVector",
                    }
                },
            }
        },
        name=index_name,
    )

    try:
        result = collection.create_search_index(model=vector_search_index_model)
        logger.info("Creating index...")
        time.sleep(20)
        logger.info(f"New index {index_name} created successfully: {result}")
    except Exception as e:
        logger.error(f"Error creating new vector search index {index_name}: {str(e)}")


def create_search_index_for_properties(
    db,
    collection_name="property_aliases",
    embedding_field_name="alias_text_embedding",
    prop_id_field_name="relation_id",
    index_name="property_aliases_ids",
    dimensions=768,
):
    logger.info(f"Starting to create index {index_name} for {collection_name}")
    collection = db.get_collection(collection_name)
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    embedding_field_name: {
                        "dimensions": dimensions,
                        "similarity": "cosine",
                        "type": "knnVector",
                    },
                    prop_id_field_name: {"type": "token"},
                },
            }
        },
        name=index_name,
    )

    try:
        result = collection.create_search_index(model=vector_search_index_model)
        logger.info("Creating index...")
        time.sleep(20)
        logger.info(f"New index {index_name} created successfully: {result}")
    except Exception as e:
        logger.error(f"Error creating new vector search index {index_name}: {str(e)}")


def create_indexes(db):
    logger.info("Creating indexes for entity_types collection...")
    db.entity_types.create_index([("entity_type_id", 1)])
    db.entity_types.create_index([("label", 1)])

    logger.info("Creating indexes for entity_type_aliases collection...")
    db.entity_type_aliases.create_index([("entity_type_id", 1)])
    db.entity_type_aliases.create_index([("alias_label", 1)])

    logger.info("Creating indexes for properties collection...")
    db.properties.create_index([("property_id", 1)])

    logger.info("All ontology indexes created successfully")


def create_wikidata_ontology_database(
    mongo_uri: str = "mongodb://localhost:27018/?directConnection=true",
    database: str = None,
    mappings_dir: str = None,
    entity_types_collection: str = "entity_types",
    entity_type_aliases_collection: str = "entity_type_aliases",
    properties_collection: str = "properties",
    property_aliases_collection: str = "property_aliases",
    entity_types_index: str = "entity_type_aliases",
    property_aliases_index: str = "property_aliases",
    drop_collections: bool = True,
    model_name: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    profile_metadata: Optional[dict] = None,
):
    """
    Populate MongoDB with Wikidata ontology data.

    Args:
        mongo_uri:                MongoDB connection URI
        database:                 MongoDB database name (resolved from runtime profile by caller)
        mappings_dir:             Directory containing ontology mapping files.
        entity_types_collection:  Collection name for entity types
        entity_type_aliases_collection: Collection name for entity type aliases
        properties_collection:    Collection name for properties
        property_aliases_collection: Collection name for property aliases
        entity_types_index:       Vector index name for entity types
        property_aliases_index:   Vector index name for property aliases
        drop_collections:         Whether to drop existing collections before creating new ones
        model_name:               HuggingFace model name for building alias embeddings
        embedding_dimension:      Dimension of the embedding vectors (must match model output)
        profile_metadata:         Dict to store in system_profile_metadata collection.
                                  Should contain at minimum: profile_id, ontology_profile_id,
                                  embedding_profile_id, ontology_language, embedding_model_name,
                                  embedding_dimension, build_version.

    Returns:
        Database object
    """
    if database is None:
        raise ValueError(
            "database name must be provided. Use resolve_runtime_profile() to derive it."
        )
    if not model_name:
        raise ValueError(
            "model_name must be provided by runtime profile; hardcoded defaults are disabled."
        )
    if embedding_dimension is None:
        raise ValueError(
            "embedding_dimension must be provided by runtime profile; hardcoded defaults are disabled."
        )

    # Default mappings directory
    if mappings_dir is None:
        current_file = Path(__file__).parent
        mappings_dir = str(current_file / "utils" / "ontology_mappings" / "")
        if not os.path.exists(mappings_dir):
            mappings_dir = "utils/ontology_mappings/"

    logger.info("Starting database population process")
    logger.info(f"Using database: {database}")
    logger.info(f"Using embedding model: {model_name} (dim={embedding_dimension})")
    logger.info(f"Loading mapping files from: {mappings_dir}")

    # Load mapping files
    with open(os.path.join(mappings_dir, "subj_constraint2prop.json"), "r") as f:
        subj2prop_constraints = json.load(f)
    with open(os.path.join(mappings_dir, "obj_constraint2prop.json"), "r") as f:
        obj2prop_constraints = json.load(f)
    with open(os.path.join(mappings_dir, "entity_type2label.json"), "r") as f:
        ENTITY_2_LABEL = json.load(f)
    with open(os.path.join(mappings_dir, "entity_type2hierarchy.json"), "r") as f:
        ENTITY_2_HIERARCHY = json.load(f)
    with open(os.path.join(mappings_dir, "entity_type2aliases.json"), "r") as f:
        ENTITY_2_ALIASES = json.load(f)
    with open(os.path.join(mappings_dir, "prop2constraints.json"), "r") as f:
        PROP_2_CONSTRAINT = json.load(f)
    with open(os.path.join(mappings_dir, "prop2label.json"), "r") as f:
        PROP_2_LABEL = json.load(f)
    with open(os.path.join(mappings_dir, "prop2aliases.json"), "r") as f:
        PROP_2_ALIASES = json.load(f)

    logger.info("Successfully loaded all mapping files")

    # Load embedding model inside the function — no module-level side effects
    device = _resolve_device()
    tokenizer, emb_model = _load_embedding_model(model_name, device)
    get_embedding = _make_get_embedding(tokenizer, emb_model, device)

    # Connect to MongoDB
    mongo_client = get_mongo_client(mongo_uri)
    db = mongo_client.get_database(database)

    # Drop all existing collections
    if drop_collections:
        logger.info("Dropping existing collections...")
        for collection_name in db.list_collection_names():
            logger.info(f"Dropping collection: {collection_name}")
            db.drop_collection(collection_name)
        logger.info("Successfully dropped all existing collections")

    # Populate collections
    populate_entity_types(
        ENTITY_2_LABEL,
        ENTITY_2_HIERARCHY,
        subj2prop_constraints,
        obj2prop_constraints,
        db,
        collection_name=entity_types_collection,
    )

    populate_entity_type_aliases(
        ENTITY_2_LABEL,
        ENTITY_2_ALIASES,
        db,
        get_embedding=get_embedding,
        collection_name=entity_type_aliases_collection,
    )

    populate_properties(
        PROP_2_LABEL, PROP_2_CONSTRAINT, db, collection_name=properties_collection
    )

    populate_property_aliases(
        PROP_2_LABEL,
        PROP_2_ALIASES,
        db,
        get_embedding=get_embedding,
        collection_name=property_aliases_collection,
    )

    # Create vector search indexes with correct dimension
    create_search_index_for_entity_types(
        db,
        collection_name=entity_type_aliases_collection,
        index_name=entity_types_index,
        dimensions=embedding_dimension,
    )
    create_search_index_for_properties(
        db,
        collection_name=property_aliases_collection,
        index_name=property_aliases_index,
        dimensions=embedding_dimension,
    )

    # Create standard indexes
    create_indexes(db)

    # Write system_profile_metadata so readiness checks can validate this DB
    if profile_metadata:
        meta_doc = {
            **profile_metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model_name": model_name,
            "embedding_dimension": embedding_dimension,
            "ontology_db_name": database,
        }
        db["system_profile_metadata"].replace_one(
            {"profile_id": profile_metadata.get("profile_id", "")},
            meta_doc,
            upsert=True,
        )
        logger.info(
            f"system_profile_metadata written for profile_id='{profile_metadata.get('profile_id')}'"
        )

    logger.info("Database population process completed")
    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate MongoDB with Wikidata ontology data"
    )

    parser.add_argument(
        "--mappings_dir",
        type=str,
        default=None,
        help="Directory containing ontology mapping files",
    )
    parser.add_argument(
        "--mongo_uri",
        type=str,
        default="mongodb://localhost:27018/?directConnection=true",
        help="MongoDB connection URI",
    )
    parser.add_argument(
        "--database",
        type=str,
        required=True,
        help="MongoDB database name (e.g. ontology__en__contriever)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace embedding model name",
    )
    parser.add_argument(
        "--embedding_dimension",
        type=int,
        required=True,
        help="Embedding vector dimension",
    )
    parser.add_argument(
        "--entity_types_collection",
        type=str,
        default="entity_types",
    )
    parser.add_argument(
        "--entity_type_aliases_collection",
        type=str,
        default="entity_type_aliases",
    )
    parser.add_argument(
        "--properties_collection",
        type=str,
        default="properties",
    )
    parser.add_argument(
        "--property_aliases_collection",
        type=str,
        default="property_aliases",
    )
    parser.add_argument(
        "--entity_types_index",
        type=str,
        default="entity_type_aliases",
    )
    parser.add_argument(
        "--property_aliases_index",
        type=str,
        default="property_aliases",
    )

    args = parser.parse_args()
    create_wikidata_ontology_database(
        mongo_uri=args.mongo_uri,
        database=args.database,
        mappings_dir=args.mappings_dir,
        model_name=args.model_name,
        embedding_dimension=args.embedding_dimension,
        entity_types_collection=args.entity_types_collection,
        entity_type_aliases_collection=args.entity_type_aliases_collection,
        properties_collection=args.properties_collection,
        property_aliases_collection=args.property_aliases_collection,
        entity_types_index=args.entity_types_index,
        property_aliases_index=args.property_aliases_index,
    )
