from dataclasses import dataclass

from .ontology_profiles import ONTOLOGY_PROFILES
from .embedding_profiles import EMBEDDING_PROFILES


@dataclass
class RuntimeProfile:
    """
    Fully resolved execution context for a given ontology + embedding combination.

    Each unique combination maps to its own ontology DB and triplets DB so that
    different embedding models never silently share the same vector workspace.

    DB naming convention (deterministic):
        ontology_db_name = "ontology__{language}__{embedding_key}"
        triplets_db_name = "triplets__{language}__{embedding_key}"

    Examples:
        en + contriever  →  ontology__en__contriever  /  triplets__en__contriever
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
    emb_key = embedding.embedding_key
    profile_id = f"{lang}__{emb_key}"

    return RuntimeProfile(
        profile_id=profile_id,
        ontology_profile_id=ontology_profile_id,
        embedding_profile_id=embedding_profile_id,
        ontology_db_name=f"ontology__{lang}__{emb_key}",
        triplets_db_name=f"triplets__{lang}__{emb_key}",
        ontology_language=lang,
        embedding_model_name=embedding.model_name,
        embedding_dimension=embedding.dimension,
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
