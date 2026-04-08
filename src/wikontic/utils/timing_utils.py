"""
timing_utils.py

Lightweight per-stage timer for pipeline instrumentation.

Usage:
    from .timing_utils import StageTimer

    timer = StageTimer()

    with timer.measure("llm_extract"):
        result = llm.call(...)

    with timer.measure("merge"):
        ...

    stats = timer.to_stats()
    # → {"llm_extract_time_ms": 320, "merge_time_ms": 45, "total_time_ms": 365, ...}

    finish_run(run_id, stats=stats)
"""
import time
from contextlib import contextmanager
from typing import Optional


class StageTimer:
    """
    Context-manager based stage timer.
    Each `with timer.measure(stage_name)` block records elapsed time in ms.
    total_time_ms is derived from the wall-clock start time, not stage sums.
    """

    def __init__(self):
        self._times: dict[str, float] = {}       # stage → ms
        self._session_start: float = time.perf_counter()
        self._failed_stage: Optional[str] = None
        self.token_usage: dict = {}              # prompt/completion/total tokens
        self.cost_estimate: Optional[float] = None

    @contextmanager
    def measure(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._times[stage] = round(elapsed_ms, 2)

    def mark_failed_at(self, stage: str):
        """Record which pipeline stage raised an exception."""
        self._failed_stage = stage

    def record_token_usage(self, usage):
        """
        Accept an OpenAI SDK usage object or a plain dict; silently no-ops on None.

        Supported formats:
            usage.prompt_tokens / completion_tokens / total_tokens  (SDK object)
            {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
        """
        if usage is None:
            return
        try:
            if hasattr(usage, "prompt_tokens"):
                self.token_usage = {
                    "prompt_tokens":     getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens":      getattr(usage, "total_tokens", 0),
                }
            elif isinstance(usage, dict):
                self.token_usage = {
                    "prompt_tokens":     usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens":      usage.get("total_tokens", 0),
                }
        except Exception:
            pass  # never let token-parsing errors surface to the caller

    def record_cost(self, cost: Optional[float]):
        """Record estimated cost. Pass 0.0 when pricing data is unavailable."""
        self.cost_estimate = cost

    def to_stats(self) -> dict:
        """
        Build the stats dict passed to finish_run(stats=...).

        Example output:
        {
            "llm_extract_time_ms": 320.5,
            "parse_time_ms": 12.1,
            "ontology_alignment_time_ms": 210.3,
            "db_write_time_ms": 35.2,
            "total_time_ms": 632.0,
            "failed_stage": null,
            "prompt_tokens": 1200,
            "completion_tokens": 380,
            "total_tokens": 1580,
            "cost_estimate": 0.0,
        }
        """
        total_ms = round((time.perf_counter() - self._session_start) * 1000, 2)

        stats: dict = {}

        for stage, ms in self._times.items():
            stats[f"{stage}_time_ms"] = ms

        stats["total_time_ms"] = total_ms
        stats["failed_stage"]  = self._failed_stage

        if self.token_usage:
            stats.update(self.token_usage)
        else:
            stats["prompt_tokens"]     = None
            stats["completion_tokens"] = None
            stats["total_tokens"]      = None

        stats["cost_estimate"] = self.cost_estimate

        return stats