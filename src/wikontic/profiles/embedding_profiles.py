from dataclasses import dataclass


@dataclass
class EmbeddingProfile:
    profile_id: str
    model_name: str
    dimension: int
    display_name: str
    compatible_languages: list[str]
    embedding_key: str  # used in deterministic DB name derivation


EMBEDDING_PROFILES: dict[str, EmbeddingProfile] = {
    "contriever_v1": EmbeddingProfile(
        profile_id="contriever_v1",
        model_name="facebook/contriever",
        dimension=768,
        display_name="Contriever",
        compatible_languages=["en"],
        embedding_key="contriever",
    ),
    "turkish_e5_large_v1": EmbeddingProfile(
        profile_id="turkish_e5_large_v1",
        model_name="ytu-ce-cosmos/turkish-e5-large",
        dimension=1024,
        display_name="Turkish E5 Large",
        compatible_languages=["tr"],
        embedding_key="turkish_e5_large",
    ),
}


def get_compatible_embedding_profiles(language: str) -> list[EmbeddingProfile]:
    return [p for p in EMBEDDING_PROFILES.values() if language in p.compatible_languages]
