"""
timing_utils.py

Pipeline stage'leri için izole timer yardımcısı.

Kullanım (structured_inference_with_db.py içinde):

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
    Context manager tabanlı stage timer.
    Her `with timer.measure(stage_name)` bloğu ms cinsinden süreyi kaydeder.
    total_time_ms otomatik hesaplanır.
    """

    def __init__(self):
        self._times: dict[str, float] = {}       # stage → ms
        self._session_start: float = time.perf_counter()
        self._failed_stage: Optional[str] = None
        self.token_usage: dict = {}              # prompt/completion/total tokens
        self.cost_estimate: Optional[float] = None

    @contextmanager
    def measure(self, stage: str):
        """
        Kullanım:
            with timer.measure("llm_extract"):
                ...
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._times[stage] = round(elapsed_ms, 2)

    def mark_failed_at(self, stage: str):
        """Hata hangi stage'de oluştuysa işaretle."""
        self._failed_stage = stage

    def record_token_usage(self, usage):
        """
        OpenAI/OpenRouter usage objesini veya dict'i alır.
        usage None ise sessizce geçer.

        Desteklenen formatlar:
            usage.prompt_tokens / completion_tokens / total_tokens (SDK object)
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
            pass  # token bilgisi alınamazsa sistem bozulmasın

    def record_cost(self, cost: Optional[float]):
        """
        Manuel maliyet kaydı.
        Model price mapping yoksa 0.0 geçilebilir.
        """
        self.cost_estimate = cost

    def to_stats(self) -> dict:
        """
        finish_run(stats=...) içine geçirilecek dict üretir.

        Örnek:
        {
            "llm_extract_time_ms": 320.5,
            "parse_time_ms": 12.1,
            "merge_time_ms": 45.0,
            "ontology_alignment_time_ms": 210.3,
            "filter_time_ms": 8.7,
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

        # Stage süreleri — her stage için "{stage}_time_ms" key'i
        for stage, ms in self._times.items():
            stats[f"{stage}_time_ms"] = ms

        stats["total_time_ms"] = total_ms
        stats["failed_stage"]  = self._failed_stage

        # Token/cost — varsa ekle, yoksa None
        if self.token_usage:
            stats.update(self.token_usage)
        else:
            stats["prompt_tokens"]     = None
            stats["completion_tokens"] = None
            stats["total_tokens"]      = None

        stats["cost_estimate"] = self.cost_estimate

        return stats