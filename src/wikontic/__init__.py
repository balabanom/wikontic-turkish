"""
Wikontic - Extract ontology-aware, Wikidata-aligned knowledge graphs from raw text using LLMs.
"""

from .create_triplets_db import create_triplets_database
from .create_ontological_triplets_db import create_ontological_triplets_database
from .create_wikidata_ontology_db import create_wikidata_ontology_database

from . import utils
from . import profiles
from .profiles import (
    RuntimeProfile,
    resolve_runtime_profile,
    DEFAULT_RUNTIME_PROFILE,
)
from .profile_readiness import check_profile_readiness, ReadinessResult

__all__ = [
    "create_triplets_database",
    "create_ontological_triplets_database",
    "create_wikidata_ontology_database",
    "utils",
    "profiles",
    "RuntimeProfile",
    "resolve_runtime_profile",
    "DEFAULT_RUNTIME_PROFILE",
    "check_profile_readiness",
    "ReadinessResult",
]
