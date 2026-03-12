import io
import json
import zipfile
from datetime import datetime, timezone
from typing import Tuple

from .run_reader import get_run, get_all_artifacts

SCHEMA_VERSION = "1.0"

# Export paketine dahil edilecek stage sırası
_STAGE_ORDER = [
    "raw_llm_output",
    "parsed_triplets",
    "merge_map_entities",
    "filtered_out",
    "final_triplets",
]


def _serialize(obj) -> str:
    """datetime gibi JSON-serializable olmayan nesneleri stringe çevirir."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _safe_json(data) -> str:
    """Hata durumunda boş JSON döner, crash etmez."""
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=_serialize)
    except Exception as e:
        return json.dumps({"error": f"Serialization failed: {str(e)}"})


def _build_repro_config(run_meta: dict) -> dict:
    """
    Replay/A-B karşılaştırması için gerekli config bilgisini toplar.
    Metadata'da eksik alanlar için güvenli default döner.
    """
    extra = run_meta.get("extra_config") or {}
    return {
        "model": run_meta.get("model", "unknown"),
        "temperature": extra.get("temperature"),
        "max_tokens": extra.get("max_tokens"),
        "thresholds": extra.get("thresholds"),
        "ontology_db_name": extra.get("ontology_db_name", "wikidata_ontology"),
        "triplets_db_name": extra.get("triplets_db_name", "demo"),
        "source_text_id": extra.get("source_text_id"),
        "app_version": extra.get("app_version"),
        "git_commit": extra.get("git_commit"),
    }


def _build_export_dict(run_id: str) -> dict:
    """
    Export paketinin ana sözlüğünü oluşturur.
    Run veya artifact eksikse None/boş bırakır, crash etmez.
    """
    run_meta = get_run(run_id) or {}
    all_artifacts = get_all_artifacts(run_id)

    # Mongo ObjectId ve datetime'ları temizle
    clean_meta = json.loads(_safe_json(dict(run_meta)))
    # _id alanını run_id olarak yeniden adlandır (okunabilirlik)
    clean_meta.pop("_id", None)
    clean_meta["run_id"] = run_id

    # Artifacts — tanımlı sırayla, eksik stage'ler null
    artifacts = {}
    for stage in _STAGE_ORDER:
        payload = all_artifacts.get(stage)
        artifacts[stage] = json.loads(_safe_json(payload)) if payload else None

    # Tanımlı sıra dışında kalan stage'leri de ekle
    for stage, payload in all_artifacts.items():
        if stage not in artifacts:
            artifacts[stage] = json.loads(_safe_json(payload))

    return {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "repro_config": _build_repro_config(run_meta),
        "run": clean_meta,
        "artifacts": artifacts,
    }


def export_run(run_id: str) -> Tuple[bytes, str, str]:
    """
    Verilen run_id için ZIP export paketi üretir.

    Returns:
        (zip_bytes, filename, mimetype)

    ZIP içeriği:
        run.json               — tüm metadata + artifacts birleşik
        stages/raw_llm_output.json
        stages/parsed_triplets.json
        ... (mevcut stage'ler)

    Crash etmez; eksik stage'ler için null/boş bırakır.
    """
    export_dict = _build_export_dict(run_id)

    # Dosya adı
    exported_at = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    short_id = run_id[:8]
    filename = f"wikontic_run_{short_id}_{exported_at}.zip"

    # ZIP oluştur
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # run.json — hepsi bir arada
        zf.writestr("run.json", _safe_json(export_dict))

        # stages/ — her stage ayrı dosya (hızlı inceleme için)
        artifacts = export_dict.get("artifacts", {})
        for stage, payload in artifacts.items():
            if payload is not None:
                zf.writestr(f"stages/{stage}.json", _safe_json(payload))

    zip_bytes = zip_buffer.getvalue()
    return zip_bytes, filename, "application/zip"