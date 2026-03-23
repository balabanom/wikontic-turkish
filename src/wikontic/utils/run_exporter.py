import io
import json
import zipfile
from datetime import datetime, timezone
from typing import Tuple

from .run_reader import get_run, get_all_artifacts

SCHEMA_VERSION = "1.1"

_STAGE_ORDER = [
    "raw_llm_output",
    "parsed_triplets",
    "merge_map_entities",
    "filtered_out",
    "final_triplets",
]


def _serialize(obj) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _safe_json(data) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=_serialize)
    except Exception as e:
        return json.dumps({"error": f"Serialization failed: {str(e)}"})


def _build_repro_config(run_meta: dict) -> dict:
    extra = run_meta.get("extra_config") or {}
    return {
        "model":            run_meta.get("model", "unknown"),
        "temperature":      extra.get("temperature"),
        "max_tokens":       extra.get("max_tokens"),
        "thresholds":       extra.get("thresholds"),
        "ontology_db_name": extra.get("ontology_db_name", "wikidata_ontology"),
        "triplets_db_name": extra.get("triplets_db_name", "demo"),
        "source_text_id":   extra.get("source_text_id"),
        "app_version":      extra.get("app_version"),
        "git_commit":       extra.get("git_commit"),
    }


def _enrich_triplets_with_sentences(triplets: list, sentences: list) -> list:
    """
    Her triplet'e sentence_id → sentences lookup ile "sentence" alanı ekler.
    sentences: [{id, text, start, end}]
    """
    if not sentences:
        return triplets

    sid_to_text = {s["id"]: s["text"] for s in sentences}
    enriched = []
    for t in triplets:
        t = dict(t)
        sid = t.get("sentence_id")
        if sid is not None and sid in sid_to_text:
            t["sentence"] = sid_to_text[sid]
        else:
            t["sentence"] = None
        enriched.append(t)
    return enriched


def _build_export_dict(run_id: str) -> dict:
    run_meta     = get_run(run_id) or {}
    all_artifacts = get_all_artifacts(run_id)

    clean_meta = json.loads(_safe_json(dict(run_meta)))
    clean_meta.pop("_id", None)
    clean_meta["run_id"] = run_id

    artifacts = {}
    for stage in _STAGE_ORDER:
        payload = all_artifacts.get(stage)
        if payload is None:
            artifacts[stage] = None
            continue

        payload = json.loads(_safe_json(payload))

        # sentence alanını triplet'lere ekle (parsed_triplets ve final_triplets)
        if stage in ("parsed_triplets", "final_triplets", "filtered_out"):
            sentences = payload.get("sentences", [])
            triplets  = payload.get("triplets", [])
            if triplets and sentences:
                payload["triplets"] = _enrich_triplets_with_sentences(triplets, sentences)

        artifacts[stage] = payload

    # Tanımlı sıra dışındaki stage'ler
    for stage, payload in all_artifacts.items():
        if stage not in artifacts:
            artifacts[stage] = json.loads(_safe_json(payload)) if payload else None

    return {
        "schema_version": SCHEMA_VERSION,
        "exported_at":    datetime.now(timezone.utc).isoformat(),
        "repro_config":   _build_repro_config(run_meta),
        "run":            clean_meta,
        "artifacts":      artifacts,
    }


def export_run(run_id: str) -> Tuple[bytes, str, str]:
    """
    Verilen run_id için ZIP export paketi üretir.

    ZIP içeriği:
        run.json               — tüm metadata + artifacts (sentence alanı dahil)
        stages/raw_llm_output.json
        stages/parsed_triplets.json
        ... (mevcut stage'ler)
    """
    export_dict = _build_export_dict(run_id)

    exported_at = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    short_id    = run_id[:8]
    filename    = f"wikontic_run_{short_id}_{exported_at}.zip"

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("run.json", _safe_json(export_dict))
        for stage, payload in (export_dict.get("artifacts") or {}).items():
            if payload is not None:
                zf.writestr(f"stages/{stage}.json", _safe_json(payload))

    return zip_buffer.getvalue(), filename, "application/zip"