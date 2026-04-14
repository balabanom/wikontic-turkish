from dataclasses import dataclass

from .ontology_profiles import ONTOLOGY_PROFILES
from .embedding_profiles import EMBEDDING_PROFILES


@dataclass
class RuntimeProfile:
    """
    Fully resolved execution context for a given ontology + embedding combination.

    Ontology DBs are shared per language; triplets DB is a single shared DB.
    Embedding-model-specific collections are namespaced by embedding_key so that
    different models never share the same vector workspace within a DB.

    DB naming convention:
        ontology_db_name = "ontology__{language}"          (one per language)
        triplets_db_name = "triplets"                      (single shared DB)

    Collection naming convention (model-namespaced):
        entity_type_aliases__{embedding_key}
        property_aliases__{embedding_key}
        entity_aliases__{embedding_key}

    Legacy override (en_legacy + contriever only):
        ontology_db_name = "wikidata_ontology"
        triplets_db_name = "demo"
        collections keep their original un-suffixed names
    """

    profile_id: str
    ontology_profile_id: str
    embedding_profile_id: str
    ontology_db_name: str
    triplets_db_name: str
    ontology_language: str
    embedding_model_name: str
    embedding_dimension: int
    requires_system_profile_metadata: bool
    # Model-namespaced collection names (differ per embedding model in shared DBs)
    entity_type_aliases_collection_name: str
    property_aliases_collection_name: str
    entity_aliases_collection_name: str
    # Vector index names (by convention equal to their collection names)
    entity_type_vector_index_name: str
    property_vector_index_name: str
    entity_aliases_vector_index_name: str
    display_name: str

    def to_metadata_dict(self) -> dict:
        """
        Flat dict of profile fields for storing in run / artifact / export documents.
        All fields are top-level so future audit filters can query them directly.
        """
        return {
            "profile_id": self.profile_id,
            "ontology_profile_id": self.ontology_profile_id,
            "embedding_profile_id": self.embedding_profile_id,
            "ontology_db_name": self.ontology_db_name,
            "triplets_db_name": self.triplets_db_name,
            "ontology_language": self.ontology_language,
            "embedding_model_name": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "requires_system_profile_metadata": self.requires_system_profile_metadata,
        }


def resolve_runtime_profile(
    ontology_profile_id: str = "ontology_en_v1",
    embedding_profile_id: str = "contriever_v1",
) -> RuntimeProfile:
    """
    Resolve a RuntimeProfile from registry IDs.

    Raises ValueError for unknown profile IDs — no silent fallback.
    """
    ontology = ONTOLOGY_PROFILES.get(ontology_profile_id)
    if ontology is None:
        raise ValueError(f"Unknown ontology profile: {ontology_profile_id!r}")

    embedding = EMBEDDING_PROFILES.get(embedding_profile_id)
    if embedding is None:
        raise ValueError(f"Unknown embedding profile: {embedding_profile_id!r}")

    lang = ontology.language
    runtime_key = ontology.runtime_key
    emb_key = embedding.embedding_key
    profile_id = f"{runtime_key}__{emb_key}"
    is_legacy_contriever = (
        ontology.profile_id == "ontology_en_legacy_v1" and emb_key == "contriever"
    )

    if is_legacy_contriever:
        # Legacy deployments keep their original hardcoded DB and collection names.
        ontology_db_name = ontology.ontology_db_name_override or f"ontology__{lang}"
        triplets_db_name = ontology.triplets_db_name_override or "triplets"
        requires_system_profile_metadata = ontology.requires_system_profile_metadata
        entity_type_aliases_col = "entity_type_aliases"
        property_aliases_col = "property_aliases"
        entity_aliases_col = "entity_aliases"
    else:
        # Standard: one ontology DB per language, one shared triplets DB.
        # Alias collections are namespaced by embedding key within each shared DB.
        ontology_db_name = f"ontology__{lang}"
        triplets_db_name = "triplets"
        requires_system_profile_metadata = True
        entity_type_aliases_col = f"entity_type_aliases__{emb_key}"
        property_aliases_col = f"property_aliases__{emb_key}"
        entity_aliases_col = f"entity_aliases__{emb_key}"

    return RuntimeProfile(
        profile_id=profile_id,
        ontology_profile_id=ontology_profile_id,
        embedding_profile_id=embedding_profile_id,
        ontology_db_name=ontology_db_name,
        triplets_db_name=triplets_db_name,
        ontology_language=lang,
        embedding_model_name=embedding.model_name,
        embedding_dimension=embedding.dimension,
        requires_system_profile_metadata=requires_system_profile_metadata,
        entity_type_aliases_collection_name=entity_type_aliases_col,
        property_aliases_collection_name=property_aliases_col,
        entity_aliases_collection_name=entity_aliases_col,
        entity_type_vector_index_name=entity_type_aliases_col,
        property_vector_index_name=property_aliases_col,
        entity_aliases_vector_index_name=entity_aliases_col,
        display_name=f"{ontology.display_name} + {embedding.display_name}",
    )


# Default profile — English ontology + Contriever.
# Used as fallback when no profile is explicitly selected.
DEFAULT_RUNTIME_PROFILE: RuntimeProfile = resolve_runtime_profile(
    "ontology_en_v1", "contriever_v1"
)
