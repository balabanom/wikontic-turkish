"""
prompts/dispatcher.py

Routes triplet extraction to one of several prompt-engineering techniques:
  - "temel"    : default Wikontic system prompt (caller uses its own flow)
  - "ape"      : Automatic Prompt Engineer optimized prompt
  - "textgrad" : TextGrad-optimized prompt
  - "dspy"     : DSPy compiled ChainOfThought module

Each technique caches its optimized artifact under prompts/optimized/.
For ape/textgrad the artifact is a plain text system prompt that the caller
plugs into its existing LLM client. For dspy the artifact is a compiled
module that must be invoked through the dspy runtime — `run_dspy_extraction`
handles that.

All techniques accept the same (model, api_key, proxy) so the user's selected
LLM is honored end-to-end.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import httpx
import openai

from src.wikontic.utils.llm_client_logger import (
    LoggedOpenAIClient,
    install_litellm_logger,
    set_llm_stage,
    reset_llm_stage,
)

logger = logging.getLogger("PromptDispatcher")

CACHE_DIR = Path(__file__).parent / "optimized"
CACHE_DIR.mkdir(exist_ok=True)

VALID_PROMPT_TYPES = {"temel", "ape", "dspy", "textgrad"}

DEFAULT_DSPY_MAX_TOKENS = 32768


_BASE_PROMPT = """You are an algorithm designed to extract structured knowledge from texts to build a Wikidata-like knowledge graph consisting of triplets (subject, relation, object) and their qualifiers.

- **Subject**: A named entity or a concept that describes a group of people, events, or any abstract objects that serves as the source of the relation.
- **Relation**: A Wikidata-style predicate that connects the subject and object.
- **Object**: A named entity or a concept that describes a group of people, events, or any abstract objects that is related to the subject.

Additionally, some triplets may have **qualifiers** that provide more context (e.g., date, place, or other attributes). Qualifiers should have relations and object like triplets do, but instead of subject their relation connects an object and the triplet qualifier belongs to. **Qualifiers must always be attached to a triplet** and never exist as standalone triplets.

**IMPORTANT NOTE (TURKISH OUTPUT REQUIREMENT):** Regardless of the input text's language, all extracted entities (subject, object), relations, and type labels (subject_type, object_type) MUST BE STRICTLY IN TURKISH. The JSON keys themselves must remain in English.

STRICT RULES:
1. The output MUST BE STRICTLY in JSON format containing a "triplets" list.
2. Each triplet dictionary MUST ONLY contain:
    - "subject": Subject entity.
    - "relation": Relation connecting subject and object.
    - "object": Object entity.
    - "qualifiers": List of dictionaries, where each dictionary contains:
        - "relation": Relation connecting triplet and object,
        - "object": Object entity connected to the main triplet
    - "subject_type": a class that describes the subject
    - "object_type": a class that describes the object
    - "kaynak_cumle": original sentence from the text where this relationship was found
3. Qualifiers must always be attached to a main triplet and must follow the [{'relation': '...', 'object': '...'}] structure.
4. **TURKISH LANGUAGE REQUIREMENT:** The JSON keys must remain in English (subject, relation, etc.), BUT all extracted values corresponding to these keys (entities, relations, types) MUST BE STRICTLY IN TURKISH. If there are no qualifiers, you MUST still include the "qualifiers" key with an empty list [].
5. NEVER compress the JSON output into a single line! DO NOT use Markdown (```json) blocks. Output pure JSON.
"""

_TRAIN_EXAMPLE = {
    "input": (
        "Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü "
        "araştırmalar yapmış bir fizikçi ve kimyagerdi. 1903'te Nobel Fizik Ödülü'nü "
        "ve 1911'de Nobel Kimya Ödülü'nü aldı."
    ),
    "output": (
        '{"triplets":[{"subject":"Marie Curie","relation":"doğum tarihi","object":"7 Kasım 1867","qualifiers":[],"subject_type":"insan","object_type":"tarih","kaynak_cumle":"Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü araştırmalar yapmış bir fizikçi ve kimyagerdi."},'
        '{"subject":"Marie Curie","relation":"ölüm tarihi","object":"4 Temmuz 1934","qualifiers":[],"subject_type":"insan","object_type":"tarih","kaynak_cumle":"Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü araştırmalar yapmış bir fizikçi ve kimyagerdi."},'
        '{"subject":"Marie Curie","relation":"meslek","object":"fizikçi","qualifiers":[],"subject_type":"insan","object_type":"meslek","kaynak_cumle":"Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü araştırmalar yapmış bir fizikçi ve kimyagerdi."},'
        '{"subject":"Marie Curie","relation":"kazandığı ödül","object":"Nobel Fizik Ödülü","qualifiers":[{"relation":"zaman noktası","object":"1903"}],"subject_type":"insan","object_type":"ödül","kaynak_cumle":"1903\'te Nobel Fizik Ödülü\'nü ve 1911\'de Nobel Kimya Ödülü\'nü aldı."}]}'
    ),
}


def is_valid_prompt_type(prompt_type: str) -> bool:
    return prompt_type in VALID_PROMPT_TYPES


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^\w\-.]", "_", model)


def _build_openai_client(
    api_key: str,
    proxy: Optional[str] = None,
    model: str = "unknown",
) -> LoggedOpenAIClient:
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if proxy:
        kwargs["http_client"] = httpx.Client(proxy=proxy)
    raw = openai.OpenAI(**kwargs)
    return LoggedOpenAIClient(raw, model=model)


def _strip_codeblock(text: str) -> str:
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\n?```$", "", stripped.strip())


def extract_json(text: str) -> dict:
    """Parse JSON from possibly-fenced LLM output. Returns {} on failure."""
    for candidate in (text, _strip_codeblock(text)):
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {"triplets": parsed}
        except json.JSONDecodeError:
            continue
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    logger.error("Failed to parse JSON from LLM output (len=%d)", len(text))
    return {"triplets": []}


# ── APE ───────────────────────────────────────────────────────────────────────

def _ape_cache_path(model: str) -> Path:
    return CACHE_DIR / f"ape_{_safe_model_name(model)}.txt"


def _ape_optimize(client: openai.OpenAI, model: str, num_candidates: int = 3) -> str:
    examples_str = (
        f"Example 1:\nInput: {_TRAIN_EXAMPLE['input']}\n"
        f"Output: {_TRAIN_EXAMPLE['output']}\n\n"
    )
    proposal = (
        "You are an expert Prompt Engineer. Optimize and improve the base system "
        "prompt so that it perfectly transforms text into the desired JSON knowledge "
        f"graph.\n\nBase Prompt:\n---\n{_BASE_PROMPT}\n---\n\n"
        f"Ideal example:\n{examples_str}\n"
        f"Write {num_candidates} different candidate system prompts that improve "
        "upon the Base Prompt. Make rules stricter, clearer, and more robust. "
        "Heavily emphasize the TURKISH OUTPUT REQUIREMENT. Explicitly instruct "
        "the model to always include the 'qualifiers' key (empty list if none) "
        "and 'kaynak_cumle' for every triplet.\n\n"
        "Output strictly in this format:\n"
        "Candidate 1: [Improved Prompt text]\n"
        "Candidate 2: [Improved Prompt text]\n"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": proposal}],
        temperature=0,
    )
    parts = re.split(r"Candidate \d+:", resp.choices[0].message.content or "")
    candidates = [c.strip() for c in parts if c.strip()][:num_candidates]
    if not candidates:
        return _BASE_PROMPT

    best_score = -1.0
    best = candidates[0]
    target = _TRAIN_EXAMPLE["input"]
    for cand in candidates:
        try:
            out = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": cand},
                    {"role": "user", "content": f"Text: {target}\nOutput:"},
                ],
                temperature=0.1,
            )
            out_text = out.choices[0].message.content or ""
            eval_prompt = (
                "Score (0-100) this knowledge-graph extraction output. Output ONLY "
                "a number, nothing else.\n"
                f"Input Text: {target}\nSystem Output: {out_text}\n"
                "Criteria: valid JSON 'triplets' list; each triplet has exact keys "
                "subject/relation/object/qualifiers/subject_type/object_type/kaynak_cumle; "
                "all values strictly in Turkish; comprehensive coverage."
            )
            score_resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
            )
            score = float((score_resp.choices[0].message.content or "0").strip())
        except (ValueError, Exception) as e:
            logger.warning("APE candidate eval failed: %s", e)
            score = 0.0
        if score > best_score:
            best_score = score
            best = cand
    logger.info("APE optimization complete (best_score=%s)", best_score)
    return best


def get_ape_prompt(model: str, api_key: str, proxy: Optional[str] = None) -> str:
    cache = _ape_cache_path(model)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    client = _build_openai_client(api_key, proxy, model=model)
    token = set_llm_stage("ape_optimize")
    try:
        best = _ape_optimize(client, model)
    finally:
        reset_llm_stage(token)
    cache.write_text(best, encoding="utf-8")
    return best


# ── TextGrad ──────────────────────────────────────────────────────────────────

def _textgrad_cache_path(model: str) -> Path:
    return CACHE_DIR / f"textgrad_{_safe_model_name(model)}.txt"


def _textgrad_optimize_impl(model: str, api_key: str, proxy: Optional[str] = None) -> str:
    try:
        import textgrad as tg
        from textgrad.engine.openai import ChatOpenAI
    except ImportError as e:
        raise RuntimeError(
            "textgrad is required for prompt_type='textgrad'. "
            "Install via: pip install textgrad"
        ) from e

    os.environ["OPENAI_API_KEY"] = api_key
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    engine = ChatOpenAI(model_string=model, temperature=0.1, max_tokens=8192)
    # Wrap engine's internal OpenAI client so every textgrad LLM call goes
    # through LoggedOpenAIClient (writes to logs/llm_requests.jsonl).
    if hasattr(engine, "client"):
        engine.client = LoggedOpenAIClient(engine.client, model=model)
    tg.set_backward_engine(engine, override=True)

    train_texts = [
        _TRAIN_EXAMPLE["input"],
        (
            "Albert Einstein (14 Mart 1879 - 18 Nisan 1955), görelilik teorisini "
            "geliştiren Alman doğumlu teorik fizikçidir. 1921 yılında fotoelektrik "
            "etki üzerine çalışmaları nedeniyle Nobel Fizik Ödülü'nü kazanmıştır."
        ),
    ]

    system_var = tg.Variable(
        value=_BASE_PROMPT,
        requires_grad=True,
        role_description="System prompt instructing the LLM to extract KG triplets in JSON.",
    )
    optimizer = tg.TGD(parameters=[system_var])
    model_wrap = tg.BlackboxLLM(engine=engine, system_prompt=system_var)

    for text in train_texts:
        user_in = tg.Variable(
            value=f"Text: {text}\nOutput:",
            requires_grad=False,
            role_description="The raw input text to be analyzed.",
        )
        response = model_wrap(user_in)
        critique = (
            f"Review the original input text:\n'{text}'\n\n"
            "Evaluate the model's JSON output on: (1) valid JSON with 'triplets' "
            "list; (2) each triplet has exact keys subject/relation/object/qualifiers/"
            "subject_type/object_type/kaynak_cumle; (3) 'qualifiers' is a list of "
            "{relation, object} objects; (4) ALL values strictly in Turkish; "
            "(5) comprehensive extraction. Provide a textual gradient instructing "
            "the system prompt to enforce these rules more aggressively."
        )
        loss = tg.TextLoss(critique)(response)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return system_var.value


def get_textgrad_prompt(model: str, api_key: str, proxy: Optional[str] = None) -> str:
    cache = _textgrad_cache_path(model)
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    token = set_llm_stage("textgrad_optimize")
    try:
        optimized = _textgrad_optimize_impl(model, api_key, proxy)
    finally:
        reset_llm_stage(token)

    cache.write_text(optimized, encoding="utf-8")
    logger.info("TextGrad optimization complete; saved to %s", cache)
    return optimized


# ── DSPy ──────────────────────────────────────────────────────────────────────

def _dspy_cache_path(model: str) -> Path:
    return CACHE_DIR / f"dspy_{_safe_model_name(model)}.json"


_dspy_runtime_cache: dict = {}


def _get_dspy_max_tokens() -> int:
    raw = os.getenv("WIKONTIC_DSPY_MAX_TOKENS")
    if not raw:
        return DEFAULT_DSPY_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid WIKONTIC_DSPY_MAX_TOKENS=%r; using default %s",
            raw,
            DEFAULT_DSPY_MAX_TOKENS,
        )
        return DEFAULT_DSPY_MAX_TOKENS
    if value <= 0:
        logger.warning(
            "Invalid WIKONTIC_DSPY_MAX_TOKENS=%r; using default %s",
            raw,
            DEFAULT_DSPY_MAX_TOKENS,
        )
        return DEFAULT_DSPY_MAX_TOKENS
    return value


def run_dspy_extraction(
    text: str,
    model: str,
    api_key: str,
    proxy: Optional[str] = None,
) -> dict:
    """Run DSPy-based extraction. Returns {'triplets': [...]} dict."""
    try:
        import dspy
    except ImportError as e:
        raise RuntimeError(
            "dspy is required for prompt_type='dspy'. Install via: pip install dspy-ai"
        ) from e

    # Route DSPy's litellm-backed LLM calls into logs/llm_requests.jsonl.
    install_litellm_logger()

    dspy_max_tokens = _get_dspy_max_tokens()
    lm_key = f"lm::{model}::max_tokens={dspy_max_tokens}"
    if lm_key not in _dspy_runtime_cache:
        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
        # litellm requires a provider prefix. The wider Wikontic codebase uses
        # OpenRouter as the gateway for any model (openrouter passes through
        # google/, anthropic/, openai/ models). Force the "openrouter/" prefix
        # unless the caller has already specified a litellm provider explicitly.
        _LITELLM_PROVIDERS = (
            "openrouter/", "openai/", "anthropic/", "azure/", "gemini/",
            "vertex_ai/", "bedrock/", "ollama/", "groq/", "deepseek/",
            "mistral/", "together_ai/", "cohere/", "huggingface/",
        )
        if model.startswith(_LITELLM_PROVIDERS):
            dspy_model = model
        else:
            dspy_model = f"openrouter/{model}"
        # cache=False ensures every inference call hits litellm (and therefore
        # our logger). Without this, DSPy's on-disk cache at ~/.dspy_cache
        # short-circuits repeat extractions and zeroes out cost telemetry.
        lm = dspy.LM(
            api_base=base_url,
            api_key=api_key,
            model=dspy_model,
            max_tokens=dspy_max_tokens,
            temperature=0.1,
            cache=False,
        )
        dspy.settings.configure(lm=lm)
        _dspy_runtime_cache[lm_key] = lm

    class KGExtraction(dspy.Signature):
        girdi_metni = dspy.InputField(desc="The raw text to be analyzed.")
        bilgi_grafigi = dspy.OutputField(
            desc="Valid JSON object strictly containing the 'triplets' key."
        )

    KGExtraction.__doc__ = _BASE_PROMPT

    mod_key = f"mod::{model}"
    if mod_key not in _dspy_runtime_cache:
        kg_module = dspy.ChainOfThought(KGExtraction)
        cache = _dspy_cache_path(model)
        if cache.exists():
            try:
                kg_module.load(str(cache))
            except Exception as e:
                logger.warning("DSPy module load failed (%s); recompiling", e)
                cache.unlink(missing_ok=True)

        if not cache.exists():
            examples = [
                dspy.Example(
                    girdi_metni=_TRAIN_EXAMPLE["input"],
                    bilgi_grafigi=_TRAIN_EXAMPLE["output"],
                ).with_inputs("girdi_metni"),
            ]

            def metric(ex, pred, trace=None):
                # Structural validation: parse JSON, require non-empty triplets list
                # where every triplet has all required keys and well-formed qualifiers.
                parsed = extract_json(pred.bilgi_grafigi or "")
                triplets = parsed.get("triplets") if isinstance(parsed, dict) else None
                if not isinstance(triplets, list) or not triplets:
                    return False
                required_keys = {
                    "subject", "relation", "object",
                    "qualifiers", "subject_type", "object_type", "kaynak_cumle",
                }
                for t in triplets:
                    if not isinstance(t, dict):
                        return False
                    if not required_keys.issubset(t.keys()):
                        return False
                    if not isinstance(t.get("qualifiers"), list):
                        return False
                    for q in t["qualifiers"]:
                        if not isinstance(q, dict):
                            return False
                        if "relation" not in q or "object" not in q:
                            return False
                return True

            compile_token = set_llm_stage("dspy_optimize")
            try:
                optimizer = dspy.teleprompt.BootstrapFewShot(
                    metric=metric, max_bootstrapped_demos=1
                )
                kg_module = optimizer.compile(kg_module, trainset=examples)
                kg_module.save(str(cache))
                logger.info("DSPy module compiled and cached at %s", cache)
            except Exception as e:
                logger.warning("DSPy compile failed (%s); using uncompiled module", e)
            finally:
                reset_llm_stage(compile_token)

        _dspy_runtime_cache[mod_key] = kg_module

    kg_module = _dspy_runtime_cache[mod_key]
    infer_token = set_llm_stage("dspy_inference")
    try:
        result = kg_module(girdi_metni=text)
    finally:
        reset_llm_stage(infer_token)
    return extract_json(result.bilgi_grafigi or "")


# ── Public dispatcher ─────────────────────────────────────────────────────────

def get_optimized_system_prompt(
    prompt_type: str,
    model: str,
    api_key: str,
    proxy: Optional[str] = None,
) -> Optional[str]:
    """
    Return the optimized system prompt for ape/textgrad.
    Return None for temel (caller uses default) or dspy (caller uses run_dspy_extraction).
    """
    if prompt_type == "ape":
        return get_ape_prompt(model, api_key, proxy)
    if prompt_type == "textgrad":
        return get_textgrad_prompt(model, api_key, proxy)
    return None
