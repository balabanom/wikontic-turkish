"""
llm_client_logger.py

Logs all LLM API calls in JSONL format.

Usage:
    from .llm_client_logger import set_llm_context, LoggedOpenAIClient

    # Before each LLM call in the pipeline:
    set_llm_context(run_id="abc-123", stage="triplet_extraction", profile_id="en__contriever")

    # In openai_utils.py __init__:
    raw_client = openai.OpenAI(api_key=..., base_url=...)
    self.client = LoggedOpenAIClient(raw_client, model=self.model)

Config (.env):
    LLM_LOG_LEVEL = full | preview   (default: preview)
    LLM_LOG_PATH  = logs/llm_requests.jsonl
"""

import json
import logging
import os
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Optional

logger = logging.getLogger("LLMClientLogger")
logger.setLevel(logging.WARNING)

# ── Context vars (Streamlit thread-safe) ──────────────────────────────────────
_ctx_run_id:              ContextVar[Optional[str]] = ContextVar("run_id",              default=None)
_ctx_stage:               ContextVar[Optional[str]] = ContextVar("stage",               default=None)
_ctx_profile_id:          ContextVar[Optional[str]] = ContextVar("profile_id",          default=None)
_ctx_ontology_profile_id: ContextVar[Optional[str]] = ContextVar("ontology_profile_id", default=None)
_ctx_embedding_profile_id: ContextVar[Optional[str]] = ContextVar("embedding_profile_id", default=None)
_ctx_ontology_db_name:    ContextVar[Optional[str]] = ContextVar("ontology_db_name",    default=None)
_ctx_triplets_db_name:    ContextVar[Optional[str]] = ContextVar("triplets_db_name",    default=None)
_ctx_ontology_language:   ContextVar[Optional[str]] = ContextVar("ontology_language",   default=None)
_ctx_embedding_model_name: ContextVar[Optional[str]] = ContextVar("embedding_model_name", default=None)
_ctx_embedding_dimension: ContextVar[Optional[int]] = ContextVar("embedding_dimension", default=None)

# ── File lock (atomic append) ─────────────────────────────────────────────────
_file_lock = threading.Lock()

# ── Config ────────────────────────────────────────────────────────────────────
_LOG_LEVEL = os.environ.get("LLM_LOG_LEVEL", "preview").lower()   # full | preview
_LOG_PATH  = Path(os.environ.get("LLM_LOG_PATH", "logs/llm_requests.jsonl"))


def set_llm_context(
    run_id: Optional[str],
    stage: Optional[str],
    profile_id: Optional[str] = None,
    ontology_profile_id: Optional[str] = None,
    embedding_profile_id: Optional[str] = None,
    ontology_db_name: Optional[str] = None,
    triplets_db_name: Optional[str] = None,
    ontology_language: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
) -> None:
    """
    Set the current run_id, pipeline stage, and profile context for the LLM audit log.
    Call this before each LLM invocation within a pipeline stage.

    Profile fields are optional — omit them to leave existing context vars unchanged.
    Pass them once at run start (in StructuredInferenceWithDB) to propagate through all stages.
    """
    _ctx_run_id.set(run_id)
    _ctx_stage.set(stage)
    if profile_id is not None:
        _ctx_profile_id.set(profile_id)
    if ontology_profile_id is not None:
        _ctx_ontology_profile_id.set(ontology_profile_id)
    if embedding_profile_id is not None:
        _ctx_embedding_profile_id.set(embedding_profile_id)
    if ontology_db_name is not None:
        _ctx_ontology_db_name.set(ontology_db_name)
    if triplets_db_name is not None:
        _ctx_triplets_db_name.set(triplets_db_name)
    if ontology_language is not None:
        _ctx_ontology_language.set(ontology_language)
    if embedding_model_name is not None:
        _ctx_embedding_model_name.set(embedding_model_name)
    if embedding_dimension is not None:
        _ctx_embedding_dimension.set(embedding_dimension)


def set_llm_stage(stage: Optional[str]) -> object:
    """
    Update only the stage context var; preserve all other context.
    Returns a token usable with reset_llm_stage() to restore the prior stage.
    """
    return _ctx_stage.set(stage)


def reset_llm_stage(token: object) -> None:
    """Restore the stage context var to the value held before set_llm_stage()."""
    try:
        _ctx_stage.reset(token)
    except (ValueError, LookupError):
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _build_request_payload(messages: list, model: str, temperature: float) -> dict:
    """Build the request payload according to LLM_LOG_LEVEL."""
    if _LOG_LEVEL == "full":
        return {
            "messages":    messages,
            "model":       model,
            "temperature": temperature,
        }
    else:  # preview
        preview_messages = []
        for m in messages:
            content = m.get("content", "")
            preview_messages.append({
                "role":        m.get("role"),
                "char_count":  len(content),
                "preview":     content[:200],
            })
        return {
            "messages_preview": preview_messages,
            "model":            model,
            "temperature":      temperature,
        }


def _build_response_preview(content: str) -> str:
    if _LOG_LEVEL == "full":
        return content
    return content[:300] + ("…" if len(content) > 300 else "")


def _append_log(entry: dict) -> None:
    """Append a log entry to the JSONL file, thread-safe."""
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with _file_lock:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        logger.warning("LLM log write failed: %s", e)


def _current_profile_context() -> dict:
    """Return a dict of current profile context vars for inclusion in log entries."""
    return {
        "profile_id":           _ctx_profile_id.get(),
        "ontology_profile_id":  _ctx_ontology_profile_id.get(),
        "embedding_profile_id": _ctx_embedding_profile_id.get(),
        "ontology_db_name":     _ctx_ontology_db_name.get(),
        "triplets_db_name":     _ctx_triplets_db_name.get(),
        "ontology_language":    _ctx_ontology_language.get(),
        "embedding_model_name": _ctx_embedding_model_name.get(),
        "embedding_dimension":  _ctx_embedding_dimension.get(),
    }


# ── Wrapper ───────────────────────────────────────────────────────────────────

class _CompletionsWrapper:
    """Mirrors the chat.completions interface; logs every call."""

    def __init__(self, real_completions, base_url: str, model: str):
        self._real     = real_completions
        self._base_url = base_url
        self._model    = model

    def create(self, model: str, messages: list, temperature: float = 0, **kwargs):
        run_id  = _ctx_run_id.get()
        stage   = _ctx_stage.get()
        ts      = _now_iso()
        t0      = perf_counter()

        request_payload = _build_request_payload(messages, model, temperature)

        try:
            response = self._real.create(
                model=model, messages=messages, temperature=temperature, **kwargs
            )
        except Exception as exc:
            latency_ms = round((perf_counter() - t0) * 1000, 2)
            _append_log({
                "ts":                ts,
                "run_id":            run_id,
                "stage":             stage,
                "provider_base_url": self._base_url,
                "model":             model,
                "endpoint":          "chat.completions",
                "request":           request_payload,
                "response_meta":     {"latency_ms": latency_ms},
                "response_preview":  None,
                "error":             str(exc),
                **_current_profile_context(),
            })
            raise

        latency_ms = round((perf_counter() - t0) * 1000, 2)

        usage = getattr(response, "usage", None)
        usage_dict = None
        if usage:
            usage_dict = {
                "prompt_tokens":     getattr(usage, "prompt_tokens",     None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens":      getattr(usage, "total_tokens",      None),
            }

        choice        = response.choices[0] if response.choices else None
        finish_reason = getattr(choice, "finish_reason", None) if choice else None
        content       = (choice.message.content or "") if choice else ""

        response_meta = {
            "id":            getattr(response, "id", None),
            "finish_reason": finish_reason,
            "usage":         usage_dict,
            "latency_ms":    latency_ms,
        }

        _append_log({
            "ts":                ts,
            "run_id":            run_id,
            "stage":             stage,
            "provider_base_url": self._base_url,
            "model":             model,
            "endpoint":          "chat.completions",
            "request":           request_payload,
            "response_meta":     response_meta,
            "response_preview":  _build_response_preview(content),
            "error":             None,
            **_current_profile_context(),
        })

        return response


class _ChatWrapper:
    def __init__(self, real_chat, base_url: str, model: str):
        self.completions = _CompletionsWrapper(
            real_chat.completions, base_url=base_url, model=model
        )


class LoggedOpenAIClient:
    """
    Wraps openai.OpenAI, intercepting chat.completions.create for logging.
    All other attributes are transparently delegated to the underlying client.

    Usage:
        raw = openai.OpenAI(api_key=..., base_url=...)
        self.client = LoggedOpenAIClient(raw, model="gpt-4o-mini")
    """

    def __init__(self, raw_client, model: str = "unknown"):
        self._raw   = raw_client
        self._model = model

        base_url = str(getattr(raw_client, "base_url", "unknown"))
        self.chat = _ChatWrapper(raw_client.chat, base_url=base_url, model=model)

    def __getattr__(self, name: str):
        """Delegate any attribute not explicitly overridden to the raw client."""
        return getattr(self._raw, name)


# ── litellm bridge (for DSPy / any litellm-backed caller) ─────────────────────

_litellm_logger_installed = False


def install_litellm_logger() -> None:
    """
    Register a litellm CustomLogger that mirrors LoggedOpenAIClient's JSONL shape.

    Idempotent — calling repeatedly only installs the logger once. Safe to call
    even when litellm is not installed (no-op with a warning).
    """
    global _litellm_logger_installed
    if _litellm_logger_installed:
        return

    try:
        import litellm
        from litellm.integrations.custom_logger import CustomLogger
    except ImportError:
        logger.warning(
            "litellm not installed; DSPy LLM calls will not be logged to %s",
            _LOG_PATH,
        )
        return

    class _WikonticLitellmLogger(CustomLogger):
        def _entry(self, kwargs, response_obj, start_time, end_time, error=None):
            try:
                start_ts = start_time.astimezone(timezone.utc) if start_time else datetime.now(timezone.utc)
                end_ts   = end_time.astimezone(timezone.utc)   if end_time   else datetime.now(timezone.utc)
                latency_ms = round((end_ts - start_ts).total_seconds() * 1000, 2)

                messages    = kwargs.get("messages") or []
                model       = kwargs.get("model") or "unknown"
                temperature = (kwargs.get("optional_params") or {}).get("temperature", 0)
                request_payload = _build_request_payload(messages, model, temperature)

                usage_dict = None
                content    = ""
                finish_reason = None
                response_id   = None
                if response_obj is not None and not error:
                    try:
                        choice = response_obj.choices[0]
                        content = (choice.message.content or "") if choice and choice.message else ""
                        finish_reason = getattr(choice, "finish_reason", None)
                        response_id   = getattr(response_obj, "id", None)
                    except (AttributeError, IndexError):
                        pass
                    usage = getattr(response_obj, "usage", None)
                    if usage:
                        usage_dict = {
                            "prompt_tokens":     getattr(usage, "prompt_tokens",     None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                            "total_tokens":      getattr(usage, "total_tokens",      None),
                        }

                base_url = (kwargs.get("litellm_params") or {}).get("api_base") or "litellm"

                _append_log({
                    "ts":                start_ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                    "run_id":            _ctx_run_id.get(),
                    "stage":             _ctx_stage.get(),
                    "provider_base_url": base_url,
                    "model":             model,
                    "endpoint":          "chat.completions",
                    "request":           request_payload,
                    "response_meta":     {
                        "id":            response_id,
                        "finish_reason": finish_reason,
                        "usage":         usage_dict,
                        "latency_ms":    latency_ms,
                    },
                    "response_preview":  _build_response_preview(content) if content else None,
                    "error":             str(error) if error else None,
                    **_current_profile_context(),
                })
            except Exception as e:
                logger.warning("litellm log write failed: %s", e)

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self._entry(kwargs, response_obj, start_time, end_time)

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            self._entry(kwargs, response_obj, start_time, end_time, error=response_obj)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self._entry(kwargs, response_obj, start_time, end_time)

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            self._entry(kwargs, response_obj, start_time, end_time, error=response_obj)

    litellm.callbacks = list(getattr(litellm, "callbacks", []) or []) + [_WikonticLitellmLogger()]
    _litellm_logger_installed = True
