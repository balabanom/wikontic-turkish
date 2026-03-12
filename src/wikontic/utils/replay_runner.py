"""
replay_runner.py

Verilen run_id için aynı pipeline'ı tekrar çalıştırır.
Yeni bir run_id üretir ve eski run'a parent_run_id olarak bağlar.

Kullanım:
    new_run_id = replay_run("some-uuid", overrides={"model": "gpt-4.1"})
"""
import os
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient

from .run_reader import get_run

_ = load_dotenv(find_dotenv())


def replay_run(
    run_id: str,
    overrides: Optional[dict] = None,
) -> str:
    """
    Verilen run_id'nin input_text + config'ini kullanarak pipeline'ı
    yeniden çalıştırır.

    Args:
        run_id:    Tekrar çalıştırılacak orijinal run'ın ID'si
        overrides: İsteğe bağlı override'lar:
                   {
                     "model": "gpt-4.1",          # model override
                     "sample_id": "...",           # session override
                   }

    Returns:
        new_run_id (str)

    Raises:
        ValueError:  run bulunamazsa veya input_text eksikse
        Exception:   pipeline hatası (finish_run FAILED ile işaretlenir)
    """
    # ── 1. Orijinal run metadata'yı oku ──────────────────────────────────────
    run_meta = get_run(run_id)
    if run_meta is None:
        raise ValueError(f"Run bulunamadı: {run_id}")

    input_text = run_meta.get("input_text", "")
    if not input_text:
        raise ValueError(
            f"Run '{run_id}' için input_text boş. "
            "Replay yapabilmek için run'ın input_text alanı dolu olmalı."
        )

    # ── 2. Config hazırla (override varsa uygula) ─────────────────────────────
    overrides = overrides or {}
    model = overrides.get("model") or run_meta.get("model", "google/gemini-2.5-flash-lite")
    sample_id = overrides.get("sample_id") or run_meta.get("sample_id", "replay")

    extra_config = dict(run_meta.get("extra_config") or {})
    extra_config.update({k: v for k, v in overrides.items() if k not in ("model", "sample_id")})
    source_text_id = extra_config.get("source_text_id")

    # ── 3. Pipeline bağımlılıklarını yükle ───────────────────────────────────
    # Import'lar burada — döngüsel import ve ağır yükleme önlenir
    from pymongo import MongoClient
    from .openai_utils import LLMTripletExtractor
    from .structured_aligner import Aligner
    from .structured_inference_with_db import StructuredInferenceWithDB

    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    api_key = os.environ.get("KEY")
    proxy_url = os.environ.get("PROXY_URL")

    mongo_client = MongoClient(mongo_uri)
    ontology_db = mongo_client.get_database("wikidata_ontology")
    triplets_db = mongo_client.get_database("demo")

    extractor = LLMTripletExtractor(
        model=model,
        api_key=api_key,
        proxy=proxy_url,
    )
    aligner = Aligner(ontology_db=ontology_db, triplets_db=triplets_db)
    inference = StructuredInferenceWithDB(
        extractor=extractor,
        aligner=aligner,
        triplets_db=triplets_db,
    )

    # ── 4. parent_run_id'yi extra_config üzerinden start_run'a ilet ──────────
    # StructuredInferenceWithDB.extract_triplets_with_ontology_filtering_and_add_to_db
    # içinde start_run çağrılır; parent_run_id'yi extra_config'e gömüyoruz,
    # sonra run_logger içinde özel parametre olarak alınabilmesi için
    # _patch_start_run ile geçici override kullanıyoruz.
    #
    # Daha temiz yol: pipeline fonksiyonuna parent_run_id geçmek.
    # Burada en az invazif çözümü tercih ediyoruz: monkey-patch yerine
    # run oluştuktan sonra DB'de parent_run_id'yi güncelliyoruz.

    # ── 5. Pipeline'ı çalıştır ────────────────────────────────────────────────
    (
        _initial,
        _final,
        _filtered,
        _ontology_filtered,
        new_run_id,
    ) = inference.extract_triplets_with_ontology_filtering_and_add_to_db(
        text=input_text,
        sample_id=sample_id,
        source_text_id=source_text_id,
    )

    # ── 6. Yeni run'a parent_run_id yaz ──────────────────────────────────────
    try:
        triplets_db["extraction_runs"].update_one(
            {"_id": new_run_id},
            {"$set": {"parent_run_id": run_id}},
        )
    except Exception:
        pass  # parent_run_id yazılamazsa run geçersiz değil, sessizce geç

    return new_run_id