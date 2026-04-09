from dataclasses import dataclass


@dataclass
class OntologyProfile:
    profile_id: str
    language: str
    display_name: str
    alias_source_version: str
    available: bool = True
    notes: str = ""


ONTOLOGY_PROFILES: dict[str, OntologyProfile] = {
    "ontology_en_v1": OntologyProfile(
        profile_id="ontology_en_v1",
        language="en",
        display_name="English Ontology",
        alias_source_version="en_aliases_v1",
        available=True,
    ),
    "ontology_tr_v1": OntologyProfile(
        profile_id="ontology_tr_v1",
        language="tr",
        display_name="Turkish Ontology",
        alias_source_version="tr_aliases_v1",
        available=False,
        notes=(
            "Turkish ontology data not yet available. "
            "Set available=True and run `python init_dbs.py --profile tr__turkish_e5_large` "
            "when data is ready."
        ),
    ),
}


def get_available_ontology_profiles() -> list[OntologyProfile]:
    return [p for p in ONTOLOGY_PROFILES.values() if p.available]
