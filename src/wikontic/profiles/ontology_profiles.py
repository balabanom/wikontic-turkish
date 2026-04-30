from dataclasses import dataclass


@dataclass
class OntologyProfile:
    profile_id: str
    language: str
    runtime_key: str
    display_name: str
    alias_source_version: str
    ontology_db_name_override: str | None = None
    triplets_db_name_override: str | None = None
    requires_system_profile_metadata: bool = True
    available: bool = True
    notes: str = ""


ONTOLOGY_PROFILES: dict[str, OntologyProfile] = {
    "ontology_en_v1": OntologyProfile(
        profile_id="ontology_en_v1",
        language="en",
        runtime_key="en",
        display_name="English Ontology",
        alias_source_version="en_aliases_v1",
        available=True,
    ),
    "ontology_en_legacy_v1": OntologyProfile(
        profile_id="ontology_en_legacy_v1",
        language="en",
        runtime_key="en_legacy",
        display_name="English Ontology (Legacy DB)",
        alias_source_version="en_aliases_legacy",
        ontology_db_name_override="wikidata_ontology",
        triplets_db_name_override="demo",
        requires_system_profile_metadata=False,
        available=True,
        notes=(
            "Uses existing legacy ontology DB `wikidata_ontology` and keeps triplets "
            "workspace in a separate legacy DB (`demo` by default)."
        ),
    ),
    "ontology_tr_v1": OntologyProfile(
        profile_id="ontology_tr_v1",
        language="tr",
        runtime_key="tr",
        display_name="Turkish Ontology",
        alias_source_version="tr_aliases_v1",
        available=True,
        notes=(
            "TR labels/aliases sourced from Wikidata SPARQL via "
            "`scripts/fetch_turkish_ontology.py`. Coverage: ~63% entity types, "
            "~74% properties; missing TR labels fall back to EN (see "
            "`src/wikontic/utils/ontology_mappings/tr/_coverage_report.json`)."
        ),
    ),
}


def get_available_ontology_profiles() -> list[OntologyProfile]:
    return [p for p in ONTOLOGY_PROFILES.values() if p.available]
