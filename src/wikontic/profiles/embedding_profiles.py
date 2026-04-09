import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EmbeddingProfile:
    profile_id: str
    model_name: str
    dimension: int
    display_name: str
    compatible_languages: list[str]
    embedding_key: str  # used in deterministic DB name derivation
    family: str = "hf_local"
    supports_turkish: str = "limited"
    available: bool = True
    notes: str = ""


def _default_profiles() -> dict[str, EmbeddingProfile]:
    # Built-in minimal fallback for environments where config file is unavailable.
    return {
        "contriever_v1": EmbeddingProfile(
            profile_id="contriever_v1",
            model_name="facebook/contriever",
            dimension=768,
            display_name="Contriever",
            compatible_languages=["en"],
            embedding_key="contriever",
            family="hf_local",
            supports_turkish="limited",
            available=True,
        )
    }


def _load_profiles_from_config() -> dict[str, EmbeddingProfile]:
    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = repo_root / "configs" / "embedding_profiles.json"
    if not cfg_path.exists():
        return _default_profiles()

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    profiles: dict[str, EmbeddingProfile] = {}
    for item in raw:
        p = EmbeddingProfile(
            profile_id=item["embedding_profile_id"],
            model_name=item["embedding_model_name"],
            dimension=int(item["embedding_dimension"]),
            display_name=item["display_name"],
            compatible_languages=item["compatible_languages"],
            embedding_key=item["embedding_key"],
            family=item.get("family", "hf_local"),
            supports_turkish=item.get("supports_turkish", "limited"),
            available=bool(item.get("available", True)),
            notes=item.get("notes", ""),
        )
        profiles[p.profile_id] = p

    return profiles or _default_profiles()


EMBEDDING_PROFILES: dict[str, EmbeddingProfile] = _load_profiles_from_config()


def get_available_embedding_profiles() -> list[EmbeddingProfile]:
    return [p for p in EMBEDDING_PROFILES.values() if p.available]


def get_unavailable_embedding_profiles() -> list[EmbeddingProfile]:
    return [p for p in EMBEDDING_PROFILES.values() if not p.available]


def get_compatible_embedding_profiles(language: str) -> list[EmbeddingProfile]:
    return [
        p
        for p in EMBEDDING_PROFILES.values()
        if p.available and language in p.compatible_languages
    ]
