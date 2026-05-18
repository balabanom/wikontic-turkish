from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any


def _edge_key(triplet: dict) -> tuple:
    return (
        triplet.get("subject", ""),
        triplet.get("relation", ""),
        triplet.get("object", ""),
        triplet.get("subject_type", ""),
        triplet.get("object_type", ""),
    )


def _unique_entities(triplets: list[dict]) -> set[str]:
    entities: set[str] = set()
    for triplet in triplets:
        subject = triplet.get("subject")
        obj = triplet.get("object")
        if subject:
            entities.add(str(subject))
        if obj:
            entities.add(str(obj))
    return entities


def _slim_triplet(triplet: dict) -> dict:
    keys = [
        "subject",
        "subject_type",
        "relation",
        "object",
        "object_type",
        "qualifiers",
        "sentence_id",
        "source_text_id",
        "sample_id",
        "reason_code",
        "exception_text",
    ]
    return {key: triplet.get(key) for key in keys if key in triplet}


def build_paper_report(
    *,
    run_id: str,
    sample_id: str,
    model: str,
    prompt_type: str | None,
    runtime_profile,
    input_text: str,
    initial_triplets: list[dict],
    final_triplets: list[dict],
    filtered_triplets: list[dict],
    ontology_filtered_triplets: list[dict],
    entity_merges: list[dict],
    db_write_results: dict[str, Any],
    stats: dict[str, Any],
    extra_config: dict[str, Any] | None = None,
) -> dict:
    raw_entities = _unique_entities(initial_triplets)
    final_entities = _unique_entities(final_triplets)
    relation_counts = Counter(
        str(t.get("relation")) for t in final_triplets if t.get("relation")
    )

    final_edge_counts = Counter(_edge_key(t) for t in final_triplets)
    duplicate_final_edges = [
        {
            "subject": key[0],
            "relation": key[1],
            "object": key[2],
            "subject_type": key[3],
            "object_type": key[4],
            "count": count,
        }
        for key, count in final_edge_counts.items()
        if count > 1
    ]

    ontology_reasons = Counter(
        t.get("reason_code", "UNKNOWN") for t in ontology_filtered_triplets
    )
    pipeline_reasons = Counter(
        t.get("reason_code", "UNKNOWN") for t in filtered_triplets
    )

    final_db = db_write_results.get("final_triplets", {}) or {}

    return {
        "schema_version": "paper-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_info": {
            "run_id": run_id,
            "sample_id": sample_id,
            "person_name": sample_id,
            "embedding_model": getattr(runtime_profile, "embedding_model_name", ""),
            "embedding_profile_id": getattr(runtime_profile, "embedding_profile_id", ""),
            "ontology_language": getattr(runtime_profile, "ontology_language", ""),
            "ontology_profile_id": getattr(runtime_profile, "ontology_profile_id", ""),
            "llm_model": model,
            "prompt_type": prompt_type,
            "profile_id": getattr(runtime_profile, "profile_id", ""),
            "ontology_db_name": getattr(runtime_profile, "ontology_db_name", ""),
            "triplets_db_name": getattr(runtime_profile, "triplets_db_name", ""),
            "extra_config": extra_config or {},
        },
        "input": {
            "text_char_count": len(input_text or ""),
            "text_preview": (input_text or "")[:500],
        },
        "triple_counts": {
            "initial_raw_triple_count": len(initial_triplets),
            "final_triple_count": len(final_triplets),
            "filtered_count": len(filtered_triplets) + len(ontology_filtered_triplets),
            "pipeline_filtered_count": len(filtered_triplets),
            "ontology_filtered_count": len(ontology_filtered_triplets),
            "kg_inserted_count": final_db.get("inserted_count", 0),
            "kg_already_existing_count": final_db.get("already_existing_count", 0),
        },
        "entity_info": {
            "raw_unique_entity_count": len(raw_entities),
            "final_unique_entity_count": len(final_entities),
            "raw_unique_entities": sorted(raw_entities),
            "final_unique_entities": sorted(final_entities),
        },
        "merge_info": {
            "merge_count": len(entity_merges),
            "merges": entity_merges,
        },
        "ontology_filtering": {
            "ontology_filtered_count": len(ontology_filtered_triplets),
            "reason_counts": dict(ontology_reasons),
            "triples": [_slim_triplet(t) for t in ontology_filtered_triplets],
        },
        "pipeline_filtering": {
            "pipeline_filtered_count": len(filtered_triplets),
            "reason_counts": dict(pipeline_reasons),
            "triples": [_slim_triplet(t) for t in filtered_triplets],
        },
        "final_triples": [_slim_triplet(t) for t in final_triplets],
        "relation_info": {
            "unique_relation_count": len(relation_counts),
            "top_relations": [
                {"relation": relation, "count": count}
                for relation, count in relation_counts.most_common(20)
            ],
        },
        "error_examples": {
            "pipeline_filtered_examples": [
                _slim_triplet(t) for t in filtered_triplets[:10]
            ],
            "ontology_filtered_examples": [
                _slim_triplet(t) for t in ontology_filtered_triplets[:10]
            ],
            "duplicate_final_edges_in_chunk": duplicate_final_edges[:10],
        },
        "telemetry": stats or {},
        "db_write_results": db_write_results,
    }


def build_batch_report(
    *,
    batch_id: str,
    source_url: str,
    sample_id: str,
    runtime_profile,
    model: str,
    prompt_type: str | None,
    chunk_summaries: list[dict],
    chunk_reports: list[dict],
    failed_chunks: list[dict] | None = None,
    status: str,
    error: str | None = None,
) -> dict:
    total = Counter()
    all_final_triples: list[dict] = []
    all_ontology_filtered: list[dict] = []
    all_merges: list[dict] = []
    all_relations = Counter()
    all_final_entities: set[str] = set()

    rows = []
    failed_chunks = failed_chunks or []
    for report in chunk_reports:
        counts = report.get("triple_counts", {})
        total.update(counts)
        all_final_triples.extend(report.get("final_triples", []))
        all_ontology_filtered.extend(
            report.get("ontology_filtering", {}).get("triples", [])
        )
        all_merges.extend(report.get("merge_info", {}).get("merges", []))
        all_final_entities.update(report.get("entity_info", {}).get("final_unique_entities", []))
        for item in report.get("relation_info", {}).get("top_relations", []):
            all_relations[item["relation"]] += item["count"]

        extra = report.get("run_info", {}).get("extra_config", {})
        rows.append(
            {
                "chunk_index": extra.get("chunk_index"),
                "run_id": report.get("run_info", {}).get("run_id"),
                "status": "DONE",
                "raw": counts.get("initial_raw_triple_count", 0),
                "final": counts.get("final_triple_count", 0),
                "filtered": counts.get("filtered_count", 0),
                "ontology_filtered": counts.get("ontology_filtered_count", 0),
                "pipeline_filtered": counts.get("pipeline_filtered_count", 0),
                "merges": report.get("merge_info", {}).get("merge_count", 0),
                "kg_inserted": counts.get("kg_inserted_count", 0),
                "kg_already_existing": counts.get("kg_already_existing_count", 0),
                "runtime_ms": report.get("telemetry", {}).get("total_time_ms"),
            }
        )

    for failed in failed_chunks:
        rows.append(
            {
                "chunk_index": failed.get("chunk_index"),
                "run_id": failed.get("run_id"),
                "status": "FAILED",
                "error": failed.get("error"),
                "error_type": failed.get("error_type"),
                "raw": failed.get("raw"),
                "final": failed.get("final"),
                "filtered": failed.get("filtered"),
                "ontology_filtered": failed.get("ontology_filtered"),
                "pipeline_filtered": failed.get("pipeline_filtered"),
                "runtime_ms": failed.get("runtime_ms"),
            }
        )

    return {
        "schema_version": "wiki-batch-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": error,
        "batch_info": {
            "batch_id": batch_id,
            "source_url": source_url,
            "sample_id": sample_id,
            "person_name": sample_id,
            "embedding_model": getattr(runtime_profile, "embedding_model_name", ""),
            "ontology_language": getattr(runtime_profile, "ontology_language", ""),
            "llm_model": model,
            "prompt_type": prompt_type,
            "profile_id": getattr(runtime_profile, "profile_id", ""),
            "triplets_db_name": getattr(runtime_profile, "triplets_db_name", ""),
        },
        "chunk_plan": chunk_summaries,
        "chunk_results": rows,
        "failed_chunks": failed_chunks,
        "totals": dict(total),
        "entity_info": {
            "final_unique_entity_count": len(all_final_entities),
            "final_unique_entities": sorted(all_final_entities),
        },
        "merge_info": {
            "merge_count": len(all_merges),
            "merges": all_merges,
        },
        "relation_info": {
            "unique_relation_count": len(all_relations),
            "top_relations": [
                {"relation": relation, "count": count}
                for relation, count in all_relations.most_common(30)
            ],
        },
        "final_triples": all_final_triples,
        "ontology_filtered_triples": all_ontology_filtered,
        "chunk_reports": chunk_reports,
    }
