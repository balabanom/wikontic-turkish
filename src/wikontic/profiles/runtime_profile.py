from dataclasses import dataclass

from .ontology_profiles import ONTOLOGY_PROFILES
from .embedding_profiles import EMBEDDING_PROFILES


@dataclass
class RuntimeProfile:
    """
    Fully resolved execution context for a given ontology + embedding combination.

    Each unique combination maps to its own ontology DB and triplets DB so that
    different embedding models never silently share the same vector workspace.

    DB naming convention (deterministic by runtime_key + embedding_key):
        ontology_db_name = "ontology__{runtime_key}__{embedding_key}"
        triplets_db_name = "triplets__{runtime_key}__{embedding_key}"
    Ontology profile may override ontology_db_name for legacy compatibility.

    Examples:
        en + contriever  →  ontology__en__contriever  /  triplets__en__contriever
        en_legacy + contriever → wikidata_ontology / triplets__en__contriever
        tr + turkish_e5  →  ontology__tr__turkish_e5_large  /  triplets__tr__turkish_e5_large
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
    ontology_db_name = (
        ontology.ontology_db_name_override
        if ontology.ontology_db_name_override
        else f"ontology__{runtime_key}__{emb_key}"
    )
    triplets_db_name = (
        ontology.triplets_db_name_override
        if ontology.triplets_db_name_override
        else f"triplets__{runtime_key}__{emb_key}"
    )

    return RuntimeProfile(
        profile_id=profile_id,
        ontology_profile_id=ontology_profile_id,
        embedding_profile_id=embedding_profile_id,
        ontology_db_name=ontology_db_name,
        triplets_db_name=triplets_db_name,
        ontology_language=lang,
        embedding_model_name=embedding.model_name,
        embedding_dimension=embedding.dimension,
        requires_system_profile_metadata=ontology.requires_system_profile_metadata,
        entity_type_vector_index_name="entity_type_aliases",
        property_vector_index_name="property_aliases",
        entity_aliases_vector_index_name="entity_aliases",
        display_name=f"{ontology.display_name} + {embedding.display_name}",
    )


# Default profile — English ontology + Contriever.
# Used as fallback when no profile is explicitly selected.
DEFAULT_RUNTIME_PROFILE: RuntimeProfile = resolve_runtime_profile(
    "ontology_en_v1", "contriever_v1"
)
