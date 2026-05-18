"""
scripts/smoke_prompt_techniques.py

End-to-end smoke test for ape / textgrad / dspy prompt techniques after the
C1-C3 logger wiring changes. Builds the optimization cache for each technique
(reusable in later real runs), runs one DSPy extraction, and verifies that
logs/llm_requests.jsonl received entries with the expected stage tags.

Usage:
    .venv/bin/python scripts/smoke_prompt_techniques.py

Cost: ~14 LLM calls on the configured model (Gemini 2.5 Flash Lite by default).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Make repo root importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from prompts.dispatcher import (   # noqa: E402
    get_ape_prompt,
    get_textgrad_prompt,
    run_dspy_extraction,
)
from src.wikontic.utils.llm_client_logger import set_llm_context   # noqa: E402

MODEL  = "google/gemini-2.5-flash-lite"
API_KEY = os.getenv("KEY")
PROXY  = os.getenv("PROXY_URL")

LOG_PATH = Path(os.environ.get("LLM_LOG_PATH", "logs/llm_requests.jsonl"))


def _baseline_log_size() -> int:
    return LOG_PATH.stat().st_size if LOG_PATH.exists() else 0


def _new_entries(baseline: int) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    rows: list[dict] = []
    with LOG_PATH.open("rb") as f:
        f.seek(baseline)
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _clear_all_caches() -> None:
    """Clear APE / TextGrad / DSPy caches so each technique exercises a real LLM call."""
    import shutil
    try:
        import platformdirs
        tg_cache = Path(platformdirs.user_cache_dir("textgrad"))
        if tg_cache.is_dir():
            shutil.rmtree(tg_cache)
            print(f"  cleared textgrad cache: {tg_cache}")
    except ImportError:
        pass

    dspy_disk_cache = Path.home() / ".dspy_cache"
    if dspy_disk_cache.is_dir():
        shutil.rmtree(dspy_disk_cache)
        print(f"  cleared dspy disk cache: {dspy_disk_cache}")

    opt_dir = REPO_ROOT / "prompts" / "optimized"
    for p in opt_dir.glob("ape_*.txt"):
        p.unlink()
        print(f"  cleared {p.name}")
    for p in opt_dir.glob("textgrad_*.txt"):
        p.unlink()
        print(f"  cleared {p.name}")
    for p in opt_dir.glob("dspy_*.json"):
        p.unlink()
        print(f"  cleared {p.name}")


def main():
    if not API_KEY:
        sys.exit("ERROR: KEY env var not set. Aborting smoke test.")

    if "--clear-caches" in sys.argv:
        print("[*] Clearing all prompt-technique caches ...")
        _clear_all_caches()

    set_llm_context(run_id="smoke-prompt-techniques", stage="bootstrap")

    base_size = _baseline_log_size()
    print(f"[*] Baseline log size: {base_size} bytes")

    print("\n[1/3] Building APE cache (will run ~6 LLM calls if not cached) ...")
    ape_prompt = get_ape_prompt(MODEL, API_KEY, PROXY)
    print(f"    ✅ APE cache ready ({len(ape_prompt)} chars)")

    print("\n[2/3] Building TextGrad cache (will run ~6 LLM calls if not cached) ...")
    tg_prompt = get_textgrad_prompt(MODEL, API_KEY, PROXY)
    print(f"    ✅ TextGrad cache ready ({len(tg_prompt)} chars)")

    print("\n[3/3] Running one DSPy extraction (compile + 1 inference call) ...")
    sample_text = (
        "Mustafa Kemal Atatürk (1881 - 10 Kasım 1938), Türkiye Cumhuriyeti'nin "
        "kurucusu ve ilk Cumhurbaşkanıdır. 1923 yılında cumhuriyeti ilan etmiştir."
    )
    out = run_dspy_extraction(sample_text, MODEL, API_KEY, PROXY)
    triplets = out.get("triplets") if isinstance(out, dict) else None
    print(
        f"    ✅ DSPy extraction returned "
        f"{len(triplets) if isinstance(triplets, list) else 'non-list'} triplets"
    )

    print("\n[*] Verifying logger entries ...")
    rows = _new_entries(base_size)
    by_stage: dict[str, int] = {}
    for r in rows:
        stage = r.get("stage") or "unknown"
        by_stage[stage] = by_stage.get(stage, 0) + 1

    print(f"    Total new log entries: {len(rows)}")
    for st, cnt in sorted(by_stage.items()):
        print(f"      stage={st!r:<22} count={cnt}")

    expected_stages = {"ape_optimize", "textgrad_optimize", "dspy_inference"}
    missing = expected_stages - set(by_stage.keys())
    if missing:
        print(
            f"\n⚠️  Missing log stages: {sorted(missing)}. "
            f"If caches were not cleared, this is expected — rerun with "
            f"`--clear-caches` to force a real LLM call for every technique."
        )
        sys.exit(1)
    print("\n✅ All expected stages logged. Wiring OK.")


if __name__ == "__main__":
    main()
