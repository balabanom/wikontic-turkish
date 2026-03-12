#!/usr/bin/env python3
"""
cleanup_runs.py — Wikontic Run Retention & Cleanup CLI

Kullanım örnekleri:

  # Kaç run silineceğini göster (güvenli)
  python scripts/cleanup_runs.py --dry-run --older-than-days 30

  # Son 200'den fazla run'ı temizle (sample_id bazında)
  python scripts/cleanup_runs.py --apply --keep-last 200

  # Sadece FAILED run'ları temizle
  python scripts/cleanup_runs.py --apply --status FAILED

  # Belirli sample_id
  python scripts/cleanup_runs.py --apply --sample-id user123 --keep-last 50

  # Orphan artifact'ları temizle
  python scripts/cleanup_runs.py --apply --orphan-sweep

  # Hepsini birlikte
  python scripts/cleanup_runs.py --apply --older-than-days 30 --keep-last 200 --orphan-sweep
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient

# ── Retention defaults (.env ile override edilebilir) ─────────────────────────
DEFAULT_RETENTION_DAYS      = int(os.environ.get("RETENTION_DAYS",         30))
DEFAULT_MAX_RUNS_PER_SAMPLE = int(os.environ.get("MAX_RUNS_PER_SAMPLE_ID", 200))
DEFAULT_MAX_TOTAL_RUNS      = int(os.environ.get("MAX_TOTAL_RUNS",         5000))


def _get_db():
    load_dotenv(find_dotenv())
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_uri)
    return client["demo"]


# ── Collectors ────────────────────────────────────────────────────────────────

def _collect_by_age(db, older_than_days: int) -> list:
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    cursor = db["extraction_runs"].find({"created_at": {"$lt": cutoff}}, {"_id": 1})
    return [doc["_id"] for doc in cursor]


def _collect_by_keep_last(db, keep_last: int, sample_id: str | None = None) -> list:
    query = {"sample_id": sample_id} if sample_id else {}
    cursor = db["extraction_runs"].find(
        query, {"_id": 1, "sample_id": 1}
    ).sort("created_at", -1)

    by_sample: dict[str, list] = defaultdict(list)
    for doc in cursor:
        by_sample[doc.get("sample_id", "unknown")].append(doc["_id"])

    to_delete = []
    for run_ids in by_sample.values():
        to_delete.extend(run_ids[keep_last:])
    return to_delete


def _collect_by_status(db, status: str, sample_id: str | None = None) -> list:
    query: dict = {"status": status}
    if sample_id:
        query["sample_id"] = sample_id
    cursor = db["extraction_runs"].find(query, {"_id": 1})
    return [doc["_id"] for doc in cursor]


def _collect_global_excess(db, max_total: int) -> list:
    total = db["extraction_runs"].count_documents({})
    if total <= max_total:
        return []
    excess = total - max_total
    cursor = db["extraction_runs"].find({}, {"_id": 1}).sort("created_at", 1).limit(excess)
    return [doc["_id"] for doc in cursor]


def _collect_orphan_run_ids(db) -> list:
    artifact_ids = set(db["extraction_artifacts"].distinct("run_id"))
    existing_ids = set(db["extraction_runs"].distinct("_id"))
    return list(artifact_ids - existing_ids)


# ── Delete helpers ────────────────────────────────────────────────────────────

def _delete_runs(db, run_ids: list, dry_run: bool) -> dict:
    """
    Artifacts önce, runs sonra silinir.
    Bu sıra orphan artifact oluşmasını önler:
    artifacts gittikten sonra runs silme başarısız olsa bile
    artifacts temiz kalır.
    """
    if not run_ids:
        return {"runs_deleted": 0, "artifacts_deleted": 0}

    art_count = db["extraction_artifacts"].count_documents({"run_id": {"$in": run_ids}})

    if dry_run:
        return {"runs_deleted": len(run_ids), "artifacts_deleted": art_count}

    art_res = db["extraction_artifacts"].delete_many({"run_id": {"$in": run_ids}})
    run_res = db["extraction_runs"].delete_many({"_id": {"$in": run_ids}})
    return {"runs_deleted": run_res.deleted_count, "artifacts_deleted": art_res.deleted_count}


def _delete_orphan_artifacts(db, dry_run: bool) -> dict:
    orphan_ids = _collect_orphan_run_ids(db)
    if not orphan_ids:
        return {"orphan_artifacts_deleted": 0, "orphan_run_id_count": 0}

    count = db["extraction_artifacts"].count_documents({"run_id": {"$in": orphan_ids}})
    if dry_run:
        return {"orphan_artifacts_deleted": count, "orphan_run_id_count": len(orphan_ids)}

    result = db["extraction_artifacts"].delete_many({"run_id": {"$in": orphan_ids}})
    return {
        "orphan_artifacts_deleted": result.deleted_count,
        "orphan_run_id_count":      len(orphan_ids),
    }


# ── Printer ───────────────────────────────────────────────────────────────────

def _print(label: str, data: dict, dry_run: bool):
    prefix = "[DRY-RUN]" if dry_run else "[APPLIED]"
    print(f"\n{prefix} {label}")
    for k, v in data.items():
        print(f"  {k}: {v}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wikontic Run Retention & Cleanup CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="Kaç kayıt silineceğini göster, silme.")
    mode.add_argument("--apply",   action="store_true",
                      help="Silme işlemini uygula.")

    parser.add_argument("--older-than-days", type=int, default=None, metavar="N",
                        help="Bu günden eski run'ları sil.")
    parser.add_argument("--keep-last",       type=int, default=None, metavar="N",
                        help="Her sample_id için son N run'ı koru.")
    parser.add_argument("--status",          type=str, default=None,
                        choices=["DONE", "FAILED", "STARTED"],
                        help="Sadece bu status'taki run'ları sil.")
    parser.add_argument("--sample-id",       type=str, default=None,
                        help="Sadece bu sample_id için çalış.")
    parser.add_argument("--max-total-runs",  type=int, default=None, metavar="N",
                        help="Global run sayısını N altında tut.")
    parser.add_argument("--orphan-sweep",    action="store_true",
                        help="Runs'ta olmayan orphan artifact'ları temizle.")

    args = parser.parse_args()
    dry_run = args.dry_run

    # Hiçbir filtre yoksa default'ları uygula
    no_filter = not any([
        args.older_than_days, args.keep_last, args.status,
        args.orphan_sweep, args.max_total_runs,
    ])
    if no_filter:
        print(
            f"[INFO] Filtre yok, varsayılanlar uygulanıyor: "
            f"older_than_days={DEFAULT_RETENTION_DAYS}, "
            f"keep_last={DEFAULT_MAX_RUNS_PER_SAMPLE}, orphan_sweep=True"
        )
        args.older_than_days = DEFAULT_RETENTION_DAYS
        args.keep_last       = DEFAULT_MAX_RUNS_PER_SAMPLE
        args.orphan_sweep    = True

    try:
        db = _get_db()
        total_runs = db["extraction_runs"].count_documents({})
        total_arts = db["extraction_artifacts"].count_documents({})
        print(f"\n[INFO] DB: {total_runs} run, {total_arts} artifact")
    except Exception as e:
        print(f"[ERROR] DB bağlantısı: {e}", file=sys.stderr)
        sys.exit(1)

    to_delete: set = set()

    if args.older_than_days:
        ids = _collect_by_age(db, args.older_than_days)
        print(f"[FILTER] older-than-days={args.older_than_days} → {len(ids)} run")
        to_delete.update(ids)

    if args.keep_last:
        ids = _collect_by_keep_last(db, args.keep_last, args.sample_id)
        print(f"[FILTER] keep-last={args.keep_last} → {len(ids)} run")
        to_delete.update(ids)

    if args.status:
        ids = _collect_by_status(db, args.status, args.sample_id)
        print(f"[FILTER] status={args.status} → {len(ids)} run")
        to_delete.update(ids)

    if args.max_total_runs:
        ids = _collect_global_excess(db, args.max_total_runs)
        print(f"[FILTER] max-total-runs={args.max_total_runs} → {len(ids)} run")
        to_delete.update(ids)

    if to_delete:
        result = _delete_runs(db, list(to_delete), dry_run)
        _print(f"Run + Artifact silme ({len(to_delete)} hedef)", result, dry_run)
    else:
        if not args.orphan_sweep:
            print("\n[INFO] Silinecek run bulunamadı.")

    if args.orphan_sweep:
        result = _delete_orphan_artifacts(db, dry_run)
        _print("Orphan artifact sweep", result, dry_run)

    if not dry_run:
        r = db["extraction_runs"].count_documents({})
        a = db["extraction_artifacts"].count_documents({})
        print(f"\n[DONE] Kalan: {r} run, {a} artifact")
    else:
        print("\n[DRY-RUN] Hiçbir şey silinmedi. --apply ile uygula.")


if __name__ == "__main__":
    main()