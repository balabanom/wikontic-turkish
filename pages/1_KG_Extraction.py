import streamlit as st
from pyvis.network import Network
import io
import tempfile
import os
import json
import re
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from dotenv import load_dotenv, find_dotenv
from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
from src.wikontic.utils.openai_utils import LLMTripletExtractor
from src.wikontic.utils.structured_aligner import Aligner
from src.wikontic.utils.run_reader import get_run, get_artifact, list_recent_runs
from src.wikontic.utils.run_logger import log_artifact
from src.wikontic.utils.run_exporter import export_run
from src.wikontic.utils.paper_report import build_batch_report
from src.wikontic.llm_models import LLM_MODEL_OPTIONS
from src.wikontic.utils.wiki_extractor import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    DEFAULT_TARGET_CHARS,
    extract_wikipedia_chunks,
)
from src.wikontic.profiles import (
    resolve_runtime_profile,
    DEFAULT_RUNTIME_PROFILE,
    get_available_ontology_profiles,
    get_compatible_embedding_profiles,
    get_unavailable_embedding_profiles,
)
from src.wikontic.profile_readiness import check_profile_readiness
from pymongo import MongoClient
from urllib.parse import unquote, urlparse
import uuid
import logging
import sys
import base64
import pandas as pd

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("KGExtraction")
logger.setLevel(logging.INFO)

ALIGNER_INTERFACE_VERSION = "structured-aligner-hierarchy-by-id-v1"

st.set_page_config(
    page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)

_ = load_dotenv(find_dotenv())
mongo_client = MongoClient(os.getenv("MONGO_URI"))
api_key   = os.getenv("KEY")
proxy_url = os.getenv("PROXY_URL")


# ── Profile helpers ───────────────────────────────────────────────────────────

def _get_active_profile():
    """Return the RuntimeProfile stored in session state, or the default."""
    return st.session_state.get("active_runtime_profile", DEFAULT_RUNTIME_PROFILE)


def _build_profile_from_selection(ontology_display: str, embedding_display: str):
    """Resolve a RuntimeProfile from UI display names."""
    ont_profile = next(
        (p for p in get_available_ontology_profiles() if p.display_name == ontology_display),
        None,
    )
    if ont_profile is None:
        return None, f"Unknown ontology profile: {ontology_display}"

    emb_profiles = get_compatible_embedding_profiles(ont_profile.language)
    emb_profile = next(
        (p for p in emb_profiles if p.display_name == embedding_display), None
    )
    if emb_profile is None:
        return None, f"No compatible embedding profile: {embedding_display}"

    try:
        profile = resolve_runtime_profile(ont_profile.profile_id, emb_profile.profile_id)
        return profile, None
    except ValueError as e:
        return None, str(e)


# ── Logo ──────────────────────────────────────────────────────────────────────
with open("media/wikontic.png", "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">KG Extraction + Visualization</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── User identity (sidebar) ───────────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state["user_id"] = ""

with st.sidebar:
    st.markdown("### 👤 Kullanıcı")
    username_input = st.text_input(
        "Kullanıcı adın:",
        value=st.session_state["user_id"],
        placeholder="örn. omer",
        key="username_field",
    ).strip()

    if username_input:
        if username_input != st.session_state["user_id"]:
            st.session_state["user_id"]             = username_input
            st.session_state["last_run_id"]         = None
            st.session_state["input_text"]          = ""
            st.session_state["selected_predefined"] = None
            st.rerun()
        user_id = username_input
        st.success(f"KG: **{user_id}**")
    else:
        if not st.session_state["user_id"]:
            st.session_state["user_id"] = f"guest_{str(uuid.uuid4())[:8]}"
        user_id = st.session_state["user_id"]
        st.info("Kalıcı KG için bir kullanıcı adı gir.")

    st.divider()

    # ── Profile selector ──────────────────────────────────────────────────────
    st.markdown("### ⚙️ Runtime Profile")

    available_ont_profiles = get_available_ontology_profiles()
    ont_display_names = [p.display_name for p in available_ont_profiles]

    current_profile = _get_active_profile()
    current_ont_display = next(
        (p.display_name for p in available_ont_profiles
         if p.profile_id == current_profile.ontology_profile_id),
        ont_display_names[0] if ont_display_names else "",
    )

    selected_ont_display = st.selectbox(
        "Ontology profile:",
        ont_display_names,
        index=ont_display_names.index(current_ont_display) if current_ont_display in ont_display_names else 0,
        key="sidebar_ontology_selector",
        help="Select the ontology language variant to use for extraction.",
    )

    # Get compatible embedding profiles for selected ontology
    selected_ont_profile = next(
        (p for p in available_ont_profiles if p.display_name == selected_ont_display), None
    )
    compatible_emb_profiles = (
        get_compatible_embedding_profiles(selected_ont_profile.language)
        if selected_ont_profile else []
    )
    unavailable_emb = [
        p for p in get_unavailable_embedding_profiles()
        if selected_ont_profile and selected_ont_profile.language in p.compatible_languages
    ]
    if unavailable_emb:
        st.caption(
            "🔒 Unavailable embeddings: " + ", ".join(p.display_name for p in unavailable_emb)
        )
    emb_display_names = [p.display_name for p in compatible_emb_profiles]

    current_emb_display = next(
        (p.display_name for p in compatible_emb_profiles
         if p.profile_id == current_profile.embedding_profile_id),
        emb_display_names[0] if emb_display_names else "",
    )

    if emb_display_names:
        selected_emb_display = st.selectbox(
            "Embedding model:",
            emb_display_names,
            index=emb_display_names.index(current_emb_display) if current_emb_display in emb_display_names else 0,
            key="sidebar_embedding_selector",
            help="Changing the embedding model changes the vector workspace. "
                 "Embedding-specific vectors are stored in model-specific collections.",
        )
    else:
        st.warning(
            "No available embedding profile for selected ontology language. "
            "Enable one in `configs/embedding_profiles.json`."
        )
        selected_emb_display = ""

    # Resolve profile from selection
    new_profile, profile_error = _build_profile_from_selection(
        selected_ont_display, selected_emb_display
    )

    profile_changed = (
        new_profile is not None
        and new_profile.profile_id != current_profile.profile_id
    )

    if profile_changed:
        # Reset cached objects when profile changes (architecture rule 5.2)
        st.session_state["active_runtime_profile"] = new_profile
        st.session_state["_aligner_profile_id"] = None  # force aligner rebuild
        st.session_state["last_run_id"] = None
        current_profile = new_profile
        st.rerun()
    elif new_profile is not None:
        current_profile = new_profile
        st.session_state["active_runtime_profile"] = current_profile

    # DBs are resolved from the selected runtime profile. Old per-profile DBs
    # may still exist in MongoDB, but they are intentionally hidden from the UI.
    ontology_db_options = [current_profile.ontology_db_name]
    stored_override_db = st.session_state.get("ontology_db_override_name")
    default_selected_db = (
        stored_override_db
        if stored_override_db in ontology_db_options
        else current_profile.ontology_db_name
    )
    if stored_override_db not in (None, current_profile.ontology_db_name):
        st.session_state["ontology_db_override_name"] = None

    selected_ontology_db = st.selectbox(
        "Ontology DB:",
        ontology_db_options,
        index=ontology_db_options.index(default_selected_db),
        key="sidebar_ontology_db_selector",
        help="Resolved from the selected runtime profile.",
    )
    st.session_state["ontology_db_override_name"] = (
        None if selected_ontology_db == current_profile.ontology_db_name else selected_ontology_db
    )
    is_external_ontology_override = st.session_state["ontology_db_override_name"] is not None

    triplets_db_options = [current_profile.triplets_db_name]
    stored_triplets_override = st.session_state.get("triplets_db_override_name")
    default_triplets_db = (
        stored_triplets_override
        if stored_triplets_override in triplets_db_options
        else current_profile.triplets_db_name
    )
    if stored_triplets_override not in (None, current_profile.triplets_db_name):
        st.session_state["triplets_db_override_name"] = None

    selected_triplets_db = st.selectbox(
        "Triplets DB:",
        triplets_db_options,
        index=triplets_db_options.index(default_triplets_db),
        key="sidebar_triplets_db_selector",
        help="Resolved from the selected runtime profile.",
    )
    st.session_state["triplets_db_override_name"] = (
        None if selected_triplets_db == current_profile.triplets_db_name else selected_triplets_db
    )
    is_external_triplets_override = st.session_state["triplets_db_override_name"] is not None

    effective_profile = replace(
        current_profile,
        ontology_db_name=selected_ontology_db,
        triplets_db_name=selected_triplets_db,
    )

    # ── Profile readiness check ───────────────────────────────────────────────
    if profile_error:
        st.error(f"Profile error: {profile_error}")
        readiness = None
    else:
        readiness = check_profile_readiness(
            effective_profile,
            mongo_client,
            relax_ontology_metadata=is_external_ontology_override,
            relax_triplets_metadata=is_external_triplets_override,
        )

    st.divider()
    st.markdown("### 📋 Active Profile")
    st.code(current_profile.profile_id, language=None)

    cols = st.columns(2)
    cols[0].caption("Ontology DB")
    cols[0].code(effective_profile.ontology_db_name, language=None)
    cols[1].caption("Triplets DB")
    cols[1].code(effective_profile.triplets_db_name, language=None)

    st.caption(f"Model: `{effective_profile.embedding_model_name}`  |  dim: `{effective_profile.embedding_dimension}`")

    if readiness is None:
        st.warning("Profile could not be resolved.")
    elif readiness.ready:
        st.success("✅ Profile ready")
    else:
        st.error("❌ Profile not ready")
        for issue in readiness.issues:
            st.caption(f"• {issue}")
        st.info(
            f"Run: `python init_dbs.py --profile {effective_profile.profile_id}`"
        )

logger.info(f"User ID: {user_id}")

# ── Block extraction if profile not ready ─────────────────────────────────────
_profile_ready = readiness is not None and readiness.ready


# ── Aligner / DB — rebuilt per profile, cached in session state ───────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def _build_aligner(
    profile_id: str,
    ontology_db_name: str,
    triplets_db_name: str,
    embedding_model_name: str,
    interface_version: str,
):
    """Cache aligner per profile_id. Rebuilt automatically when profile changes."""
    od = mongo_client.get_database(ontology_db_name)
    td = mongo_client.get_database(triplets_db_name)
    profile = st.session_state.get("active_runtime_profile", DEFAULT_RUNTIME_PROFILE)
    return Aligner(
        ontology_db=od,
        triplets_db=td,
        embedding_model_name=embedding_model_name,
        runtime_profile=profile,
    )


if _profile_ready:
    aligner = _build_aligner(
        effective_profile.profile_id,
        effective_profile.ontology_db_name,
        effective_profile.triplets_db_name,
        effective_profile.embedding_model_name,
        ALIGNER_INTERFACE_VERSION,
    )
    triplets_db = mongo_client.get_database(effective_profile.triplets_db_name)
else:
    aligner = None
    triplets_db = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_related_triplets(entities, sample_id_override: str | None = None):
    if triplets_db is None:
        return []
    sid = sample_id_override if sample_id_override else user_id
    collection = triplets_db.get_collection("triplets")
    query = {
        "$or": [{"subject": {"$in": entities}}, {"object": {"$in": entities}}],
        "sample_id": sid,
    }
    results = collection.find(query, {"_id": 0, "subject": 1, "relation": 1, "object": 1})
    return [(doc["subject"], doc["relation"], doc["object"]) for doc in results]


def visualize_knowledge_graph(triplets, highlight_entities=None):
    net = Network(height="600px", width="100%", bgcolor="#ffffff",
                  font_color="black", directed=True)
    highlight_entities = highlight_entities or set()
    added_nodes = set()

    for s, r, o in triplets:
        if not s or not o:
            continue
        s, r, o = str(s), str(r), str(o)
        for node in [s, o]:
            if node not in added_nodes:
                net.add_node(node, label=node,
                             color="#B2CD9C" if node in highlight_entities else "#C7C8CC")
                added_nodes.add(node)
        net.add_edge(s, o, label=r, color="#000000")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html_path = tmp.name
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=600, scrolling=True)
    os.remove(html_path)


def visualize_initial_knowledge_graph(initial_triplets):
    net = Network(height="600px", width="100%", bgcolor="#ffffff",
                  font_color="black", directed=True)
    for t in initial_triplets:
        s = t.get("subject") or ""
        r = t.get("relation") or ""
        o = t.get("object") or ""
        if not s or not o:
            continue
        s, o = str(s), str(o)
        net.add_node(s, label=s, color="#B2CD9C")
        net.add_node(o, label=o, color="#B2CD9C")
        net.add_edge(s, o, label=str(r), color="#000000")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html_path = tmp.name
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=600, scrolling=True)
    os.remove(html_path)


def visualize_ontology_neighborhood(neighborhood: dict):
    net = Network(height="500px", width="100%", bgcolor="#ffffff",
                  font_color="black", directed=True)
    center = neighborhood["center"]
    net.add_node(center["id"], label=f"{center['label']}\n({center['id']})",
                 color="#4A90D9", size=25)
    for parent in neighborhood.get("parents", []):
        net.add_node(parent["id"], label=f"{parent['label']}\n({parent['id']})",
                     color="#F5A623", size=18)
        net.add_edge(center["id"], parent["id"], label="is a", color="#F5A623", dashes=True)
    for prop in neighborhood.get("properties", []):
        prop_node_id    = f"prop_{prop['id']}"
        color           = "#5CB85C" if prop["direction"] == "subject" else "#9B59B6"
        direction_label = "→ subject" if prop["direction"] == "subject" else "← object"
        net.add_node(prop_node_id,
                     label=f"{prop['label']}\n({prop['id']})\n{direction_label}",
                     color=color, size=14, shape="box")
        net.add_edge(center["id"], prop_node_id, label=prop["label"], color=color)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html_path = tmp.name
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=500, scrolling=True)
    os.remove(html_path)


def _show_sentence_detail(triplets: list, key_prefix: str):
    """Render expandable sentence provenance detail below a triplet table."""
    has_sentences = any(t.get("sentence_full") for t in triplets)
    if not has_sentences:
        return
    if st.checkbox("📖 Cümle detaylarını göster", key=f"{key_prefix}_sent_detail"):
        for t in triplets:
            if t.get("sentence_full"):
                st.caption(
                    f"[{t.get('sentence_id', '?')}] "
                    f"**{t.get('subject','')}** — {t.get('relation','')} — **{t.get('object','')}**"
                )
                st.info(t["sentence_full"])


def _json_download_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def _paper_report_summary(report: dict) -> dict:
    counts = report.get("triple_counts", {})
    extra = report.get("run_info", {}).get("extra_config", {})
    return {
        "chunk": extra.get("chunk_index"),
        "run_id": report.get("run_info", {}).get("run_id"),
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


def _json_text(payload: dict | list) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _build_wiki_preview(url: str, target_chars: int, max_chars: int, min_chars: int) -> dict:
    wiki_result = extract_wikipedia_chunks(
        url,
        target_chars=target_chars,
        max_chars=max_chars,
        min_chars=min_chars,
    )
    return {
        "url": wiki_result.url,
        "paragraph_count": wiki_result.paragraph_count,
        "chunk_count": wiki_result.chunk_count,
        "chunk_summaries": wiki_result.chunk_summaries(),
        "chunks": [chunk.to_dict() for chunk in wiki_result.chunks],
    }


def _parse_wikipedia_url_list(raw_text: str) -> list[str]:
    urls: list[str] = []
    for part in re.split(r"[\s,]+", raw_text or ""):
        url = part.strip()
        if url:
            urls.append(url)
    return urls


def _wiki_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    title = unquote(parsed.path.rstrip("/").split("/")[-1] or parsed.netloc)
    return title.replace("_", " ") or url


def _safe_filename_part(text: str, fallback: str = "wikipedia") -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
    safe = safe.strip("._-")
    return (safe or fallback)[:80]


def _wiki_multi_sample_id(owner_user_id: str, multi_batch_id: str, article_index: int) -> str:
    return f"{owner_user_id}:wiki:{multi_batch_id}:article:{article_index:02d}"


def _progress_value(offset: float, span: float, fraction: float) -> float:
    return max(0.0, min(1.0, offset + span * fraction))


def _run_wiki_batch_from_preview(
    *,
    wiki_preview: dict,
    batch_id: str,
    sample_id: str,
    owner_user_id: str,
    selected_model: str,
    selected_prompt_type: str,
    inference_with_db: StructuredInferenceWithDB,
    start_chunk: int = 1,
    end_chunk_input: int = 0,
    batch_type: str = "wikipedia_url",
    status_box=None,
    table_box=None,
    progress=None,
    progress_label_prefix: str = "",
    progress_offset: float = 0.0,
    progress_span: float = 1.0,
    extra_batch_config: dict | None = None,
) -> dict:
    total_chunks = int(wiki_preview["chunk_count"])
    start_chunk = max(1, min(int(start_chunk), total_chunks))
    end_chunk = (
        total_chunks
        if int(end_chunk_input) <= 0
        else max(start_chunk, min(int(end_chunk_input), total_chunks))
    )
    chunks_to_run = [
        chunk for chunk in wiki_preview["chunks"]
        if start_chunk <= int(chunk["index"]) <= end_chunk
    ]

    chunk_reports = []
    failed_chunks = []
    rows = []
    batch_status = "DONE"
    batch_error = None
    last_run_id = None
    extra_batch_config = extra_batch_config or {}

    run_total = max(1, len(chunks_to_run))
    for run_pos, chunk in enumerate(chunks_to_run, start=1):
        pos = int(chunk["index"])
        prefix = f"{progress_label_prefix} " if progress_label_prefix else ""
        if status_box is not None:
            status_box.info(
                f"{prefix}Chunk {pos}/{wiki_preview['chunk_count']} çalışıyor "
                f"({chunk['char_count']} chars, {chunk['paragraph_count']} paragraphs)."
            )
        if progress is not None:
            progress.progress(
                _progress_value(progress_offset, progress_span, (run_pos - 1) / run_total),
                text=f"{prefix}Chunk {pos}/{wiki_preview['chunk_count']} çalışıyor...",
            )

        try:
            chunk_extra_config = {
                "batch_id": batch_id,
                "batch_type": batch_type,
                "owner_user_id": owner_user_id,
                "source_url": wiki_preview["url"],
                "chunk_index": pos,
                "chunk_count": wiki_preview["chunk_count"],
                "resume_start_chunk": start_chunk,
                "resume_end_chunk": end_chunk,
                "resume_run_position": run_pos,
                "resume_run_total": run_total,
                "chunk_char_count": chunk["char_count"],
                "chunk_paragraph_count": chunk["paragraph_count"],
            }
            chunk_extra_config.update(extra_batch_config)

            (
                _initial_triplets,
                _final_triplets,
                _filtered_triplets,
                _ontology_filtered_triplets,
                run_id,
            ) = inference_with_db.extract_triplets_with_ontology_filtering_and_add_to_db(
                text=chunk["text"],
                sample_id=sample_id,
                source_text_id=f"{batch_id}:chunk:{pos}",
                extra_config=chunk_extra_config,
            )
            last_run_id = run_id
            paper_report = get_artifact(
                run_id,
                "paper_report",
                db_name=effective_profile.triplets_db_name,
            )
            if paper_report:
                chunk_reports.append(paper_report)
                row = _paper_report_summary(paper_report)
            else:
                row = {
                    "chunk": pos,
                    "run_id": run_id,
                    "raw": len(_initial_triplets),
                    "final": len(_final_triplets),
                    "filtered": len(_filtered_triplets) + len(_ontology_filtered_triplets),
                    "ontology_filtered": len(_ontology_filtered_triplets),
                    "pipeline_filtered": len(_filtered_triplets),
                }
            row["status"] = "DONE"
            rows.append(row)
            if table_box is not None:
                table_box.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        except Exception as e:
            failed_run_id = getattr(inference_with_db, "_last_run_id", None)
            failed_art = (
                get_artifact(
                    failed_run_id,
                    "failure_report",
                    db_name=effective_profile.triplets_db_name,
                )
                if failed_run_id else None
            )
            failed_meta = (
                get_run(failed_run_id, db_name=effective_profile.triplets_db_name)
                if failed_run_id else None
            )
            parsed_art = (
                get_artifact(
                    failed_run_id,
                    "parsed_triplets",
                    db_name=effective_profile.triplets_db_name,
                )
                if failed_run_id else None
            )
            final_art = (
                get_artifact(
                    failed_run_id,
                    "final_triplets",
                    db_name=effective_profile.triplets_db_name,
                )
                if failed_run_id else None
            )
            filtered_art = (
                get_artifact(
                    failed_run_id,
                    "filtered_out",
                    db_name=effective_profile.triplets_db_name,
                )
                if failed_run_id else None
            )
            failed_chunk = {
                "chunk_index": pos,
                "run_id": failed_run_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "raw": (parsed_art or {}).get("count"),
                "final": (final_art or {}).get("count"),
                "filtered": (filtered_art or {}).get("count"),
                "ontology_filtered": (filtered_art or {}).get("ontology_filtered_count"),
                "pipeline_filtered": (filtered_art or {}).get("pipeline_exception_count"),
                "runtime_ms": ((failed_meta or {}).get("stats") or {}).get("total_time_ms"),
                "failure_report": failed_art,
            }
            failed_chunks.append(failed_chunk)
            batch_status = "FAILED"
            batch_error = f"Chunk {pos} failed: {e}"
            rows.append(
                {
                    "chunk": pos,
                    "run_id": failed_run_id,
                    "status": "FAILED",
                    "error": str(e),
                    "raw": failed_chunk["raw"],
                    "final": failed_chunk["final"],
                    "filtered": failed_chunk["filtered"],
                }
            )
            if table_box is not None:
                table_box.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            if status_box is not None:
                status_box.error(batch_error)
            last_run_id = failed_run_id
            break

        if progress is not None:
            progress.progress(
                _progress_value(progress_offset, progress_span, run_pos / run_total),
                text=f"{prefix}Chunk {pos}/{wiki_preview['chunk_count']} tamamlandı.",
            )

    batch_report = build_batch_report(
        batch_id=batch_id,
        source_url=wiki_preview["url"],
        sample_id=sample_id,
        runtime_profile=effective_profile,
        model=selected_model,
        prompt_type=selected_prompt_type,
        chunk_summaries=wiki_preview["chunk_summaries"],
        chunk_reports=chunk_reports,
        failed_chunks=failed_chunks,
        status=batch_status,
        error=batch_error,
    )
    batch_report["batch_info"].update(
        {
            "owner_user_id": owner_user_id,
            "article_sample_id": sample_id,
            **extra_batch_config,
        }
    )
    return {
        "batch_report": batch_report,
        "rows": rows,
        "status": batch_status,
        "error": batch_error,
        "last_run_id": last_run_id,
        "chunk_reports": chunk_reports,
        "failed_chunks": failed_chunks,
    }


def _build_wiki_multi_report(
    *,
    multi_batch_id: str,
    owner_user_id: str,
    requested_urls: list[str],
    article_reports: list[dict],
    model: str,
    prompt_type: str,
) -> dict:
    totals: dict[str, int] = {}
    articles = []
    for report in article_reports:
        info = report.get("batch_info", {})
        report_totals = report.get("totals", {})
        for key, value in report_totals.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + int(value)
        run_ids = [
            row.get("run_id")
            for row in report.get("chunk_results", [])
            if row.get("run_id")
        ]
        articles.append(
            {
                "article_index": info.get("article_index"),
                "title": info.get("article_title") or _wiki_title_from_url(info.get("source_url", "")),
                "source_url": info.get("source_url"),
                "batch_id": info.get("batch_id"),
                "sample_id": info.get("article_sample_id") or info.get("sample_id"),
                "status": report.get("status"),
                "error": report.get("error"),
                "chunk_count": len(report.get("chunk_plan", [])),
                "run_ids": run_ids,
                "totals": report_totals,
            }
        )

    done_count = sum(1 for article in articles if article["status"] == "DONE")
    failed_count = sum(1 for article in articles if article["status"] != "DONE")
    status = "DONE" if failed_count == 0 else ("FAILED" if done_count == 0 else "PARTIAL")

    return {
        "schema_version": "wiki-multi-batch-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "multi_batch_info": {
            "multi_batch_id": multi_batch_id,
            "owner_user_id": owner_user_id,
            "requested_url_count": len(requested_urls),
            "completed_url_count": done_count,
            "failed_url_count": failed_count,
            "llm_model": model,
            "prompt_type": prompt_type,
            "profile_id": effective_profile.profile_id,
            "triplets_db_name": effective_profile.triplets_db_name,
        },
        "requested_urls": requested_urls,
        "articles": articles,
        "totals": totals,
        "article_reports": article_reports,
    }


def _build_wiki_multi_zip(multi_report: dict, db_name: str) -> bytes:
    zip_buffer = io.BytesIO()
    exported_at = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("multi_batch_report.json", _json_text(multi_report))
        zf.writestr(
            "README.txt",
            "Each articles/<index>_<title>/ folder is isolated with its own sample_id.\n"
            "wiki_batch_report.json contains the per-link aggregate report.\n"
            "runs/ contains one ZIP export per chunk run.\n",
        )
        for fallback_index, report in enumerate(multi_report.get("article_reports", []), start=1):
            info = report.get("batch_info", {})
            article_index = int(info.get("article_index") or fallback_index)
            title = info.get("article_title") or _wiki_title_from_url(info.get("source_url", ""))
            prefix = f"articles/{article_index:02d}_{_safe_filename_part(title, f'article_{article_index:02d}')}"
            zf.writestr(f"{prefix}/wiki_batch_report.json", _json_text(report))
            zf.writestr(f"{prefix}/chunk_plan.json", _json_text(report.get("chunk_plan", [])))
            zf.writestr(f"{prefix}/chunk_results.json", _json_text(report.get("chunk_results", [])))
            zf.writestr(f"{prefix}/final_triples.json", _json_text(report.get("final_triples", [])))
            zf.writestr(
                f"{prefix}/ontology_filtered_triples.json",
                _json_text(report.get("ontology_filtered_triples", [])),
            )
            for row in report.get("chunk_results", []):
                run_id = row.get("run_id")
                if not run_id:
                    continue
                chunk_index = int(row.get("chunk_index") or row.get("chunk") or 0)
                try:
                    run_zip, run_filename, _ = export_run(run_id, db_name=db_name)
                    zf.writestr(
                        f"{prefix}/runs/chunk_{chunk_index:03d}_{run_filename}",
                        run_zip,
                    )
                except Exception as e:
                    zf.writestr(
                        f"{prefix}/runs/chunk_{chunk_index:03d}_{run_id[:8]}_export_error.json",
                        _json_text({"run_id": run_id, "error": str(e)}),
                    )
        zf.writestr("export_info.json", _json_text({"exported_at": exported_at}))
    return zip_buffer.getvalue()


def _render_wiki_multi_report(multi_report: dict) -> None:
    info = multi_report.get("multi_batch_info", {})
    st.markdown("**Son Çoklu Wikipedia Batch Raporu**")
    mc = st.columns(5)
    mc[0].metric("Status", multi_report.get("status", "—"))
    mc[1].metric("URL", info.get("requested_url_count", 0))
    mc[2].metric("Done", info.get("completed_url_count", 0))
    mc[3].metric("Failed", info.get("failed_url_count", 0))
    mc[4].metric("Final", multi_report.get("totals", {}).get("final_triple_count", 0))

    articles = multi_report.get("articles", [])
    if articles:
        article_rows = []
        for article in articles:
            totals = article.get("totals", {})
            article_rows.append(
                {
                    "article": article.get("article_index"),
                    "title": article.get("title"),
                    "status": article.get("status"),
                    "chunks": article.get("chunk_count"),
                    "final": totals.get("final_triple_count", 0),
                    "inserted": totals.get("kg_inserted_count", 0),
                    "existing": totals.get("kg_already_existing_count", 0),
                    "sample_id": article.get("sample_id"),
                    "batch_id": article.get("batch_id"),
                }
            )
        st.dataframe(pd.DataFrame(article_rows), width="stretch", hide_index=True)

    for fallback_index, report in enumerate(multi_report.get("article_reports", []), start=1):
        batch_info = report.get("batch_info", {})
        article_index = int(batch_info.get("article_index") or fallback_index)
        title = batch_info.get("article_title") or _wiki_title_from_url(batch_info.get("source_url", ""))
        label = f"{article_index:02d}. {title} — {report.get('status', 'UNKNOWN')}"
        with st.expander(label, expanded=article_index == 1):
            st.caption(
                f"URL: `{batch_info.get('source_url', '')}`  |  "
                f"sample_id: `{batch_info.get('article_sample_id') or batch_info.get('sample_id', '')}`  |  "
                f"batch_id: `{batch_info.get('batch_id', '')}`"
            )
            if report.get("error"):
                st.error(report["error"])

            totals = report.get("totals", {})
            cols = st.columns(5)
            cols[0].metric("Raw", totals.get("initial_raw_triple_count", 0))
            cols[1].metric("Final", totals.get("final_triple_count", 0))
            cols[2].metric("Ontology Filtered", totals.get("ontology_filtered_count", 0))
            cols[3].metric("KG Inserted", totals.get("kg_inserted_count", 0))
            cols[4].metric("KG Existing", totals.get("kg_already_existing_count", 0))

            chunk_results = report.get("chunk_results", [])
            if chunk_results:
                st.markdown("**Chunk sonuçları**")
                st.dataframe(pd.DataFrame(chunk_results), width="stretch", hide_index=True)

            final_triples = report.get("final_triples", [])
            if final_triples:
                st.markdown("**Makale KG Tripletleri**")
                final_df = pd.DataFrame(final_triples)
                display_cols = [
                    col for col in [
                        "subject", "subject_type", "relation", "object",
                        "object_type", "sentence_id", "sample_id",
                    ]
                    if col in final_df.columns
                ]
                st.dataframe(final_df[display_cols], width="stretch", hide_index=True)
                if st.checkbox(
                    "Makale KG grafını göster",
                    key=f"multi_article_graph_{batch_info.get('batch_id', article_index)}",
                ):
                    visualize_initial_knowledge_graph(final_triples)
            else:
                st.info("Bu link için final triplet yok.")

            if st.checkbox(
                "Chunk KG'lerini göster",
                key=f"multi_article_chunks_{batch_info.get('batch_id', article_index)}",
            ):
                chunk_reports = report.get("chunk_reports", [])
                if not chunk_reports:
                    st.info("Gösterilecek chunk raporu yok.")
                for chunk_report in chunk_reports:
                    extra = chunk_report.get("run_info", {}).get("extra_config", {})
                    chunk_index = extra.get("chunk_index", "?")
                    chunk_triples = chunk_report.get("final_triples", [])
                    st.markdown(f"**Chunk {chunk_index}**")
                    if chunk_triples:
                        chunk_df = pd.DataFrame(chunk_triples)
                        chunk_cols = [
                            col for col in [
                                "subject", "subject_type", "relation", "object",
                                "object_type", "sentence_id",
                            ]
                            if col in chunk_df.columns
                        ]
                        st.dataframe(chunk_df[chunk_cols], width="stretch", hide_index=True)
                        if st.checkbox(
                            f"Chunk {chunk_index} grafını göster",
                            key=(
                                f"multi_chunk_graph_"
                                f"{batch_info.get('batch_id', article_index)}_{chunk_index}"
                            ),
                        ):
                            visualize_initial_knowledge_graph(chunk_triples)
                    else:
                        st.info(f"Chunk {chunk_index} için final triplet yok.")


# ── Transparency Panel ────────────────────────────────────────────────────────

def render_transparency_panel(selected_run_id: str):
    if not selected_run_id:
        st.info("Henüz extraction yapılmadı.")
        return

    run_meta = get_run(selected_run_id, db_name=current_profile.triplets_db_name)
    if run_meta:
        mc = st.columns(4)
        mc[0].metric("Status", run_meta.get("status", "—"))
        mc[1].metric("Model",  run_meta.get("model",  "—"))
        created_at = run_meta.get("created_at", "")
        if hasattr(created_at, "strftime"):
            created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")
        mc[2].metric("Created At", str(created_at))
        mc[3].metric("Sample ID", str(run_meta.get("sample_id", "—"))[:12] + "…")
        if run_meta.get("status") == "FAILED" and run_meta.get("error"):
            st.error(f"Run hatası: {run_meta['error']}")

        # Show profile metadata for the run
        if run_meta.get("profile_id"):
            st.caption(
                f"🗂 Profile: `{run_meta['profile_id']}`  |  "
                f"Lang: `{run_meta.get('ontology_language', '—')}`  |  "
                f"Embedding: `{run_meta.get('embedding_model_name', '—')}`"
            )
    else:
        st.warning(f"Run bulunamadı: `{selected_run_id}`")
        return

    st.caption(f"🔑 Run ID: `{selected_run_id}`")

    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 Raw-0: LLM Output",
        "🟡 Raw-1: Parsed Triplets",
        "🔀 Merge Log",
        "🚫 Filtered Out",
        "🟢 Final Triplets",
    ])

    # ── Tab 0: Raw LLM Output ────────────────────────────────────────────────
    with tab0:
        art = get_artifact(selected_run_id, "raw_llm_output", db_name=current_profile.triplets_db_name)
        if art is None:
            st.warning("Bu stage için kayıt bulunamadı.")
        else:
            with st.expander("Ham LLM Çıktısı", expanded=True):
                st.code(art.get("text", ""), language="json")

    with tab1:
        art = get_artifact(selected_run_id, "parsed_triplets", db_name=current_profile.triplets_db_name)
        if art is None:
            st.warning("Bu stage için kayıt bulunamadı.")
        else:
            triplets = art.get("triplets", [])
            st.caption(f"**{art.get('count', len(triplets))} triplet** parse edildi")
            if triplets:
                cols     = ["subject", "relation", "object", "sentence_id", "sentence_preview"]
                existing = [c for c in cols if c in pd.DataFrame(triplets).columns]
                st.dataframe(pd.DataFrame(triplets)[existing],
                             width="stretch", hide_index=True)
                _show_sentence_detail(triplets, key_prefix=f"parsed_{selected_run_id}")
            else:
                st.info("Parse edilmiş triplet bulunamadı.")

    with tab2:
        art = get_artifact(selected_run_id, "merge_map_entities", db_name=current_profile.triplets_db_name)
        if art is None:
            st.warning("Bu stage için kayıt bulunamadı.")
        else:
            merges = art.get("merges", [])
            if not merges:
                st.info("Bu run'da hiçbir entity merge edilmedi.")
            else:
                st.caption(f"**{len(merges)} entity** merge edildi")
                df = pd.DataFrame(merges)
                existing = [c for c in ["from", "to", "entity_type", "method"] if c in df.columns]
                st.dataframe(df[existing], width="stretch", hide_index=True)

    with tab3:
        art = get_artifact(selected_run_id, "filtered_out", db_name=current_profile.triplets_db_name)
        if art is None:
            st.warning("Bu stage için kayıt bulunamadı.")
        else:
            triplets     = art.get("triplets", [])
            total        = art.get("count", len(triplets))
            pipeline_exc = art.get("pipeline_exception_count", 0)
            ontology_flt = art.get("ontology_filtered_count", 0)

            if total == 0:
                st.success("Bu run'da hiçbir triplet elenmedi.")
            else:
                fc = st.columns(3)
                fc[0].metric("🚫 Toplam Elenen",     total)
                fc[1].metric("⚠️ Pipeline Exception", pipeline_exc)
                fc[2].metric("🔴 Ontology Violation", ontology_flt)

                cols     = ["subject", "relation", "object", "reason_code",
                            "filter_stage", "sentence_id", "sentence_preview", "exception_text"]
                df       = pd.DataFrame(triplets)
                existing = [c for c in cols if c in df.columns]
                st.dataframe(df[existing], width="stretch", hide_index=True)
                _show_sentence_detail(triplets, key_prefix=f"filtered_{selected_run_id}")

    with tab4:
        art = get_artifact(selected_run_id, "final_triplets", db_name=current_profile.triplets_db_name)
        if art is None:
            st.warning("Bu stage için kayıt bulunamadı.")
        else:
            triplets = art.get("triplets", [])
            count    = art.get("count", len(triplets))
            fc       = st.columns(3)
            fc[0].metric("✅ Final", count)
            if art.get("filtered_count") is not None:
                fc[1].metric("⚠️ Filtered", art["filtered_count"])
            if art.get("ontology_filtered_count") is not None:
                fc[2].metric("🚫 Ontology Filtered", art["ontology_filtered_count"])

            if triplets:
                cols     = ["subject", "relation", "object", "subject_type", "object_type",
                            "sentence_id", "sentence_preview"]
                df       = pd.DataFrame(triplets)
                existing = [c for c in cols if c in df.columns]
                st.dataframe(df[existing], width="stretch", hide_index=True)
                _show_sentence_detail(triplets, key_prefix=f"final_{selected_run_id}")
            else:
                st.info("Final triplet bulunamadı.")


# ── Ontology Neighborhood ─────────────────────────────────────────────────────

def render_ontology_neighborhood_panel(selected_run_id: str):
    st.subheader("🗺️ Ontoloji Neighborhood")
    entity_types = []
    if selected_run_id:
        art = get_artifact(selected_run_id, "final_triplets", db_name=current_profile.triplets_db_name)
        if art:
            type_set = set()
            for t in art.get("triplets", []):
                if t.get("subject_type"):
                    type_set.add(t["subject_type"])
                if t.get("object_type"):
                    type_set.add(t["object_type"])
            entity_types = sorted(type_set)

    if not entity_types:
        st.info("Gösterilecek entity type bulunamadı.")
        return

    selected_type = st.selectbox("Entity type seç:", entity_types,
                                  key="ontology_type_selector")
    if selected_type and aligner:
        with st.spinner(f"'{selected_type}' için ontoloji neighborhood yükleniyor..."):
            neighborhood = aligner.get_ontology_neighborhood(selected_type)

        if neighborhood is None:
            ontology_db_name = getattr(aligner.ontology_db, "name", "?")
            et_collection = aligner.entity_type_collection_name
            try:
                total = aligner.ontology_db.get_collection(et_collection).count_documents({})
                exact = aligner.ontology_db.get_collection(et_collection).count_documents(
                    {"label": selected_type}
                )
            except Exception as e:
                total, exact = -1, f"err: {e}"
            st.warning(
                f"'{selected_type}' için ontoloji verisi bulunamadı.\n\n"
                f"- Sorgulanan DB: `{ontology_db_name}`\n"
                f"- `{et_collection}` toplam doc: `{total}`\n"
                f"- `label == \"{selected_type}\"` eşleşme: `{exact}`\n\n"
                f"Eğer DB adı beklediğinizden farklıysa, sidebar'daki "
                f"**Ontology DB** seçicisini kontrol edin."
            )
            return

        center     = neighborhood["center"]
        parents    = neighborhood.get("parents", [])
        properties = neighborhood.get("properties", [])

        nc = st.columns(3)
        nc[0].metric("🔵 Merkez Type",   center["label"])
        nc[1].metric("🟠 Parent Sayısı", len(parents))
        nc[2].metric("🟢 Property Sayısı", len(properties))

        visualize_ontology_neighborhood(neighborhood)

        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**Parent Types**")
            if parents:
                df = pd.DataFrame(parents)[["label", "id"]]
                df.columns = ["Label", "Wikidata ID"]
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.info("Parent type bulunamadı.")
        with dc2:
            st.markdown("**Allowed Properties**")
            if properties:
                df = pd.DataFrame(properties)[["label", "id", "direction"]]
                df.columns = ["Label", "Wikidata ID", "Direction"]
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.info("Property bulunamadı.")


# ── Model selection ───────────────────────────────────────────────────────────
model_options  = LLM_MODEL_OPTIONS
selected_model = st.selectbox("Choose a model for KG extraction:", model_options, index=0)

# ── Prompt technique selection ────────────────────────────────────────────────
prompt_type_options = ["temel", "ape", "dspy", "textgrad"]
prompt_type_labels  = {
    "temel":    "Temel (default Wikontic prompt)",
    "ape":      "APE (Automatic Prompt Engineer)",
    "dspy":     "DSPy (compiled module)",
    "textgrad": "TextGrad (textual gradient)",
}
selected_prompt_type = st.selectbox(
    "Prompt technique:",
    prompt_type_options,
    index=0,
    format_func=lambda k: prompt_type_labels.get(k, k),
    help="Temel = default. Diğer seçenekler yalnızca prompt aşamasını değiştirir; "
         "ontology/merge pipeline aynı kalır. İlk kullanımda optimize edilip cache'lenir.",
)

# ── Profile not ready guard ───────────────────────────────────────────────────
if not _profile_ready:
    st.error(
        f"❌ Profile **{current_profile.profile_id}** is not initialized. "
        f"Extraction is disabled until the profile is ready."
    )
    if readiness and readiness.issues:
        with st.expander("Issues"):
            for issue in readiness.issues:
                st.warning(issue)
    st.stop()

WIKIPEDIA_TEXTS = {
    "Albert Einstein": "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, Einstein also made important contributions to quantum mechanics. His mass–energy equivalence formula E = mc², which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect.",
    "The Renaissance": "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries. It occurred after the Crisis of the Late Middle Ages and was associated with great social change. In addition to the standard periodization, proponents of a 'long Renaissance' may put its beginning in the 14th century and its end in the 17th century. The traditional view focuses more on the early modern aspects of the Renaissance and argues that it was a break from the past, but many historians today focus more on its medieval aspects and argue that it was an extension of the Middle Ages.",
    "The Great Wall of China": "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups from the Eurasian Steppe. Several walls were built from as early as the 7th century BC, with selective stretches later joined by Qin Shi Huang (220–206 BC), the first emperor of China. Little of the Qin wall remains. Later on, many successive dynasties built and maintained multiple stretches of border walls. The most well-known sections of the wall were built by the Ming dynasty (1368–1644).",
    "Shakespeare": "Shakespeare was an English playwright, poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. He is often called England's national poet and the 'Bard of Avon'. His extant works, including collaborations, consist of some 39 plays, 154 sonnets, three long narrative poems, and a few other verses, some of uncertain authorship. His plays have been translated into every major living language and are performed more often than those of any other playwright.",
    "The Industrial Revolution": "The Industrial Revolution was the transition from creating goods by hand to using machines. Its start and end are widely debated by scholars, but the period generally spanned from about 1760 to 1840. According to some, this turning point in history is responsible for an increase in population, an increase in the standard of living, and the emergence of the capitalist economy. The Industrial Revolution began in Great Britain, and many of the technological and architectural innovations were of British origin. By the mid-18th century, Britain was the world's leading commercial nation, controlling a global trading empire with colonies in North America and the Caribbean.",
}

if "input_text"          not in st.session_state: st.session_state.input_text          = ""
if "selected_predefined" not in st.session_state: st.session_state.selected_predefined = None

# ── Resolve selected_run_id before rendering ──────────────────────────────────
_rv_navigated: bool = bool(st.session_state.get("selected_run_id"))
_last_run_id_early: str | None = st.session_state.get("last_run_id")
if _rv_navigated:
    _last_run_id_early = st.session_state["selected_run_id"]

try:
    _recent_runs_early = list_recent_runs(
        limit=20, sample_id=user_id, db_name=current_profile.triplets_db_name
    )
except Exception:
    _recent_runs_early = []

if _recent_runs_early:
    _run_ids_early = [r["run_id"] for r in _recent_runs_early]
    _default_early = (
        _run_ids_early.index(_last_run_id_early)
        if _last_run_id_early in _run_ids_early else 0
    )
    selected_run_id: str | None = _run_ids_early[_default_early]
else:
    selected_run_id = _last_run_id_early

_nav_sample_id: str | None = None
if _rv_navigated and selected_run_id:
    _nav_meta      = get_run(selected_run_id, db_name=current_profile.triplets_db_name)
    _nav_sample_id = (_nav_meta or {}).get("sample_id") or user_id

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Text Examples")
    predefined_options = ["Custom Text"] + list(WIKIPEDIA_TEXTS.keys())
    initial_index = (
        predefined_options.index(st.session_state.selected_predefined)
        if st.session_state.selected_predefined in predefined_options else 0
    )
    selected_predefined = st.radio(
        "Choose a text option:", predefined_options,
        index=initial_index, key="predefined_selector",
    )
    if selected_predefined != st.session_state.selected_predefined:
        st.session_state.selected_predefined = selected_predefined
        if selected_predefined != "Custom Text" and selected_predefined in WIKIPEDIA_TEXTS:
            st.session_state.input_text = WIKIPEDIA_TEXTS[selected_predefined]
            st.rerun()

with col2:
    st.subheader("Text Input")
    input_text = st.text_area(
        "Enter or modify text:",
        value=st.session_state.input_text,
        placeholder="Paste your text here or select a text option from the left...",
        height=300,
        key="text_area",
    )
    st.session_state.input_text = input_text

st.divider()
st.subheader("Wikipedia Link ile Çek")

for _k, _default in {
    "wiki_url": "",
    "wiki_preview": None,
    "wiki_batch_report": None,
    "wiki_batch_rows": [],
    "wiki_multi_urls": "",
    "wiki_multi_report": None,
    "wiki_multi_zip_bytes": None,
    "wiki_multi_zip_filename": "",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _default

wiki_url = st.text_input(
    "Wikipedia URL",
    value=st.session_state["wiki_url"],
    placeholder="https://tr.wikipedia.org/wiki/Cristiano_Ronaldo",
    key="wiki_url_input",
).strip()
st.session_state["wiki_url"] = wiki_url

wch1, wch2, wch3 = st.columns(3)
wiki_target_chars = wch1.number_input(
    "Target chars",
    min_value=200,
    max_value=8000,
    value=DEFAULT_TARGET_CHARS,
    step=100,
    key="wiki_target_chars",
)
wiki_max_chars = wch2.number_input(
    "Max chars",
    min_value=300,
    max_value=12000,
    value=DEFAULT_MAX_CHARS,
    step=100,
    key="wiki_max_chars",
)
wiki_min_chars = wch3.number_input(
    "Min chars",
    min_value=0,
    max_value=5000,
    value=DEFAULT_MIN_CHARS,
    step=100,
    key="wiki_min_chars",
)

resume_col1, resume_col2, resume_col3 = st.columns([2, 1, 1])
wiki_batch_id_override = resume_col1.text_input(
    "Mevcut batch_id ile devam et (opsiyonel)",
    value=st.session_state.get("wiki_batch_id_override", ""),
    placeholder="örn. wiki_20260518_101430_0bd50c91",
    key="wiki_batch_id_override",
).strip()
wiki_start_chunk = int(
    resume_col2.number_input(
        "Başlangıç chunk",
        min_value=1,
        max_value=999,
        value=int(st.session_state.get("wiki_start_chunk", 1)),
        step=1,
        key="wiki_start_chunk",
    )
)
wiki_end_chunk_input = int(
    resume_col3.number_input(
        "Bitiş chunk",
        min_value=0,
        max_value=999,
        value=int(st.session_state.get("wiki_end_chunk", 0)),
        step=1,
        help="0 = son chunka kadar",
        key="wiki_end_chunk",
    )
)

wp_col1, wp_col2 = st.columns([1, 1])
preview_wiki = wp_col1.button("Linki Önizle", width="stretch", disabled=not wiki_url)
run_wiki_batch = wp_col2.button(
    "Chunkları Sırayla KG'ye Ekle",
    width="stretch",
    disabled=not wiki_url,
)

if preview_wiki:
    with st.spinner("Wikipedia sayfası çekiliyor ve chunklara ayrılıyor..."):
        try:
            wiki_result = extract_wikipedia_chunks(
                wiki_url,
                target_chars=int(wiki_target_chars),
                max_chars=int(wiki_max_chars),
                min_chars=int(wiki_min_chars),
            )
            st.session_state["wiki_preview"] = {
                "url": wiki_result.url,
                "paragraph_count": wiki_result.paragraph_count,
                "chunk_count": wiki_result.chunk_count,
                "chunk_summaries": wiki_result.chunk_summaries(),
                "chunks": [chunk.to_dict() for chunk in wiki_result.chunks],
            }
            st.session_state["wiki_batch_report"] = None
            st.session_state["wiki_batch_rows"] = []
        except Exception as e:
            st.session_state["wiki_preview"] = None
            st.error(f"Wikipedia çekimi başarısız: {e}")

wiki_preview = st.session_state.get("wiki_preview")
if wiki_preview:
    wc = st.columns(3)
    wc[0].metric("Paragraf", wiki_preview["paragraph_count"])
    wc[1].metric("Chunk", wiki_preview["chunk_count"])
    wc[2].metric("URL", wiki_preview["url"].split("/wiki/")[-1][:30])
    st.dataframe(
        pd.DataFrame(wiki_preview["chunk_summaries"]),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Chunk metinlerini göster", expanded=False):
        for chunk in wiki_preview["chunks"]:
            st.markdown(
                f"**Chunk {chunk['index']:03d}** · "
                f"{chunk['char_count']} chars · {chunk['paragraph_count']} paragraphs"
            )
            st.text_area(
                "",
                value=chunk["text"],
                height=180,
                disabled=True,
                key=f"wiki_preview_chunk_{chunk['index']}",
            )

if run_wiki_batch:
    if not selected_model:
        st.warning("Please select a model for KG extraction.")
    else:
        if not wiki_preview or wiki_preview.get("url") != wiki_url:
            with st.spinner("Önizleme yok veya URL değişti; yeniden çekiliyor..."):
                wiki_result = extract_wikipedia_chunks(
                    wiki_url,
                    target_chars=int(wiki_target_chars),
                    max_chars=int(wiki_max_chars),
                    min_chars=int(wiki_min_chars),
                )
                wiki_preview = {
                    "url": wiki_result.url,
                    "paragraph_count": wiki_result.paragraph_count,
                    "chunk_count": wiki_result.chunk_count,
                    "chunk_summaries": wiki_result.chunk_summaries(),
                    "chunks": [chunk.to_dict() for chunk in wiki_result.chunks],
                }
                st.session_state["wiki_preview"] = wiki_preview

        total_chunks = int(wiki_preview["chunk_count"])
        start_chunk = max(1, min(int(wiki_start_chunk), total_chunks))
        end_chunk = (
            total_chunks
            if int(wiki_end_chunk_input) <= 0
            else max(start_chunk, min(int(wiki_end_chunk_input), total_chunks))
        )
        chunks_to_run = [
            chunk for chunk in wiki_preview["chunks"]
            if start_chunk <= int(chunk["index"]) <= end_chunk
        ]
        batch_id = (
            wiki_batch_id_override
            or f"wiki_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        progress = st.progress(0, text="Wikipedia batch başlıyor...")
        status_box = st.empty()
        table_box = st.empty()
        chunk_reports = []
        failed_chunks = []
        rows = []
        batch_status = "DONE"
        batch_error = None

        extractor = LLMTripletExtractor(
            model=selected_model,
            api_key=api_key,
            proxy=proxy_url,
            prompt_type=selected_prompt_type,
        )
        inference_with_db = StructuredInferenceWithDB(
            extractor=extractor,
            aligner=aligner,
            triplets_db=triplets_db,
            runtime_profile=effective_profile,
        )

        run_total = len(chunks_to_run)
        for run_pos, chunk in enumerate(chunks_to_run, start=1):
            pos = int(chunk["index"])
            status_box.info(
                f"Chunk {pos}/{wiki_preview['chunk_count']} çalışıyor "
                f"({chunk['char_count']} chars, {chunk['paragraph_count']} paragraphs)."
            )
            progress.progress(
                (run_pos - 1) / run_total,
                text=f"Chunk {pos}/{wiki_preview['chunk_count']} çalışıyor...",
            )
            try:
                (
                    _initial_triplets,
                    _final_triplets,
                    _filtered_triplets,
                    _ontology_filtered_triplets,
                    run_id,
                ) = inference_with_db.extract_triplets_with_ontology_filtering_and_add_to_db(
                    text=chunk["text"],
                    sample_id=user_id,
                    source_text_id=f"{batch_id}:chunk:{pos}",
                    extra_config={
                        "batch_id": batch_id,
                        "batch_type": "wikipedia_url",
                        "source_url": wiki_preview["url"],
                        "chunk_index": pos,
                        "chunk_count": wiki_preview["chunk_count"],
                        "resume_start_chunk": start_chunk,
                        "resume_end_chunk": end_chunk,
                        "resume_run_position": run_pos,
                        "resume_run_total": run_total,
                        "chunk_char_count": chunk["char_count"],
                        "chunk_paragraph_count": chunk["paragraph_count"],
                    },
                )
                paper_report = get_artifact(
                    run_id,
                    "paper_report",
                    db_name=effective_profile.triplets_db_name,
                )
                if paper_report:
                    chunk_reports.append(paper_report)
                    row = _paper_report_summary(paper_report)
                else:
                    row = {
                        "chunk": pos,
                        "run_id": run_id,
                        "raw": len(_initial_triplets),
                        "final": len(_final_triplets),
                        "filtered": len(_filtered_triplets) + len(_ontology_filtered_triplets),
                        "ontology_filtered": len(_ontology_filtered_triplets),
                        "pipeline_filtered": len(_filtered_triplets),
                    }
                rows.append(row)
                table_box.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                st.session_state["last_run_id"] = run_id
                selected_run_id = run_id
            except Exception as e:
                failed_run_id = getattr(inference_with_db, "_last_run_id", None)
                failed_art = (
                    get_artifact(
                        failed_run_id,
                        "failure_report",
                        db_name=effective_profile.triplets_db_name,
                    )
                    if failed_run_id else None
                )
                failed_meta = (
                    get_run(failed_run_id, db_name=effective_profile.triplets_db_name)
                    if failed_run_id else None
                )
                parsed_art = (
                    get_artifact(
                        failed_run_id,
                        "parsed_triplets",
                        db_name=effective_profile.triplets_db_name,
                    )
                    if failed_run_id else None
                )
                final_art = (
                    get_artifact(
                        failed_run_id,
                        "final_triplets",
                        db_name=effective_profile.triplets_db_name,
                    )
                    if failed_run_id else None
                )
                filtered_art = (
                    get_artifact(
                        failed_run_id,
                        "filtered_out",
                        db_name=effective_profile.triplets_db_name,
                    )
                    if failed_run_id else None
                )
                failed_chunk = {
                    "chunk_index": pos,
                    "run_id": failed_run_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "raw": (parsed_art or {}).get("count"),
                    "final": (final_art or {}).get("count"),
                    "filtered": (filtered_art or {}).get("count"),
                    "ontology_filtered": (filtered_art or {}).get("ontology_filtered_count"),
                    "pipeline_filtered": (filtered_art or {}).get("pipeline_exception_count"),
                    "runtime_ms": ((failed_meta or {}).get("stats") or {}).get("total_time_ms"),
                    "failure_report": failed_art,
                }
                failed_chunks.append(failed_chunk)
                batch_status = "FAILED"
                batch_error = f"Chunk {pos} failed: {e}"
                rows.append(
                    {
                        "chunk": pos,
                        "run_id": failed_run_id,
                        "status": "FAILED",
                        "error": str(e),
                        "raw": failed_chunk["raw"],
                        "final": failed_chunk["final"],
                        "filtered": failed_chunk["filtered"],
                    }
                )
                table_box.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                status_box.error(batch_error)
                break

            progress.progress(
                run_pos / run_total,
                text=f"Chunk {pos}/{wiki_preview['chunk_count']} tamamlandı.",
            )

        batch_report = build_batch_report(
            batch_id=batch_id,
            source_url=wiki_preview["url"],
            sample_id=user_id,
            runtime_profile=effective_profile,
            model=selected_model,
            prompt_type=selected_prompt_type,
            chunk_summaries=wiki_preview["chunk_summaries"],
            chunk_reports=chunk_reports,
            failed_chunks=failed_chunks,
            status=batch_status,
            error=batch_error,
        )
        st.session_state["wiki_batch_report"] = batch_report
        st.session_state["wiki_batch_rows"] = rows

        if rows and rows[-1].get("run_id"):
            try:
                log_artifact(
                    rows[-1]["run_id"],
                    "wiki_batch_report",
                    batch_report,
                    db_name=effective_profile.triplets_db_name,
                    profile_id=effective_profile.profile_id,
                    runtime_profile=effective_profile,
                )
            except Exception as e:
                st.warning(f"Batch report Run Viewer'a yazılamadı: {e}")

        if batch_status == "DONE":
            status_box.success(
                f"Batch tamamlandı: {len(chunk_reports)}/{wiki_preview['chunk_count']} chunk işlendi."
            )
        else:
            status_box.error(batch_error or "Batch failed.")

wiki_batch_report = st.session_state.get("wiki_batch_report")
if wiki_batch_report:
    st.markdown("**Son Wikipedia Batch Raporu**")
    if st.session_state.get("wiki_batch_rows"):
        st.dataframe(
            pd.DataFrame(st.session_state["wiki_batch_rows"]),
            width="stretch",
            hide_index=True,
        )
    totals = wiki_batch_report.get("totals", {})
    tc = st.columns(5)
    tc[0].metric("Raw", totals.get("initial_raw_triple_count", 0))
    tc[1].metric("Final", totals.get("final_triple_count", 0))
    tc[2].metric("Ontology Filtered", totals.get("ontology_filtered_count", 0))
    tc[3].metric("KG Inserted", totals.get("kg_inserted_count", 0))
    tc[4].metric("KG Existing", totals.get("kg_already_existing_count", 0))
    st.download_button(
        "Wikipedia Batch JSON indir",
        data=_json_download_bytes(wiki_batch_report),
        file_name=f"{wiki_batch_report['batch_info']['batch_id']}.json",
        mime="application/json",
        key="download_wiki_batch_report",
    )

st.divider()
st.subheader("Wikipedia Link Listesi ile Toplu Çek")
previous_wiki_multi_urls = st.session_state.get("wiki_multi_urls", "")
wiki_multi_urls_text = st.text_area(
    "Wikipedia URL listesi",
    value=previous_wiki_multi_urls,
    placeholder=(
        "Her satıra bir Wikipedia linki gir:\n"
        "https://tr.wikipedia.org/wiki/Cristiano_Ronaldo\n"
        "https://tr.wikipedia.org/wiki/Lionel_Messi"
    ),
    height=150,
    key="wiki_multi_urls_input",
)
if wiki_multi_urls_text != previous_wiki_multi_urls:
    st.session_state["wiki_multi_report"] = None
    st.session_state["wiki_multi_zip_bytes"] = None
    st.session_state["wiki_multi_zip_filename"] = ""
st.session_state["wiki_multi_urls"] = wiki_multi_urls_text
wiki_multi_urls = _parse_wikipedia_url_list(wiki_multi_urls_text)

if wiki_multi_urls:
    st.caption(
        f"{len(wiki_multi_urls)} link sırayla işlenecek. "
        "Her link kendi `sample_id` değeriyle izole çalışır; önceki linkin tripletleri sonraki linkte kullanılmaz."
    )
    with st.expander("Algılanan linkleri göster", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "index": i,
                        "title": _wiki_title_from_url(url),
                        "url": url,
                    }
                    for i, url in enumerate(wiki_multi_urls, start=1)
                ]
            ),
            width="stretch",
            hide_index=True,
        )

run_wiki_multi_batch = st.button(
    "Listedeki Linkleri Sırayla KG'ye Ekle",
    width="stretch",
    disabled=not wiki_multi_urls,
)

if run_wiki_multi_batch:
    if not selected_model:
        st.warning("Please select a model for KG extraction.")
    else:
        multi_batch_id = (
            f"wiki_multi_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        st.session_state["wiki_multi_report"] = None
        st.session_state["wiki_multi_zip_bytes"] = None
        st.session_state["wiki_multi_zip_filename"] = ""

        progress = st.progress(0, text="Çoklu Wikipedia batch başlıyor...")
        status_box = st.empty()
        table_box = st.empty()
        article_reports: list[dict] = []
        article_rows: list[dict] = []
        last_multi_run_id = None

        extractor = LLMTripletExtractor(
            model=selected_model,
            api_key=api_key,
            proxy=proxy_url,
            prompt_type=selected_prompt_type,
        )
        inference_with_db = StructuredInferenceWithDB(
            extractor=extractor,
            aligner=aligner,
            triplets_db=triplets_db,
            runtime_profile=effective_profile,
        )

        article_total = len(wiki_multi_urls)
        for article_pos, article_url in enumerate(wiki_multi_urls, start=1):
            article_title = _wiki_title_from_url(article_url)
            article_batch_id = f"{multi_batch_id}_article_{article_pos:02d}"
            article_sample_id = _wiki_multi_sample_id(
                user_id, multi_batch_id, article_pos
            )
            article_prefix = f"URL {article_pos}/{article_total}: {article_title}"
            status_box.info(f"{article_prefix} çekiliyor ve chunklara ayrılıyor...")

            try:
                article_preview = _build_wiki_preview(
                    article_url,
                    target_chars=int(wiki_target_chars),
                    max_chars=int(wiki_max_chars),
                    min_chars=int(wiki_min_chars),
                )
            except Exception as e:
                failed_report = build_batch_report(
                    batch_id=article_batch_id,
                    source_url=article_url,
                    sample_id=article_sample_id,
                    runtime_profile=effective_profile,
                    model=selected_model,
                    prompt_type=selected_prompt_type,
                    chunk_summaries=[],
                    chunk_reports=[],
                    failed_chunks=[],
                    status="FAILED",
                    error=f"Wikipedia çekimi/chunklama başarısız: {e}",
                )
                failed_report["batch_info"].update(
                    {
                        "owner_user_id": user_id,
                        "article_sample_id": article_sample_id,
                        "multi_batch_id": multi_batch_id,
                        "article_index": article_pos,
                        "article_count": article_total,
                        "article_title": article_title,
                        "batch_type": "wikipedia_multi_url",
                    }
                )
                article_reports.append(failed_report)
                article_rows.append(
                    {
                        "article": article_pos,
                        "title": article_title,
                        "status": "FAILED",
                        "chunks": 0,
                        "final": 0,
                        "sample_id": article_sample_id,
                        "error": str(e),
                    }
                )
                table_box.dataframe(pd.DataFrame(article_rows), width="stretch", hide_index=True)
                progress.progress(
                    article_pos / article_total,
                    text=f"{article_prefix} başarısız, sonraki linke geçiliyor.",
                )
                continue

            result = _run_wiki_batch_from_preview(
                wiki_preview=article_preview,
                batch_id=article_batch_id,
                sample_id=article_sample_id,
                owner_user_id=user_id,
                selected_model=selected_model,
                selected_prompt_type=selected_prompt_type,
                inference_with_db=inference_with_db,
                start_chunk=1,
                end_chunk_input=0,
                batch_type="wikipedia_multi_url",
                status_box=status_box,
                table_box=None,
                progress=progress,
                progress_label_prefix=article_prefix,
                progress_offset=(article_pos - 1) / article_total,
                progress_span=1 / article_total,
                extra_batch_config={
                    "multi_batch_id": multi_batch_id,
                    "article_index": article_pos,
                    "article_count": article_total,
                    "article_title": article_title,
                },
            )
            article_report = result["batch_report"]
            article_reports.append(article_report)
            if result.get("last_run_id"):
                last_multi_run_id = result["last_run_id"]
                st.session_state["last_run_id"] = last_multi_run_id
                selected_run_id = last_multi_run_id
                try:
                    log_artifact(
                        last_multi_run_id,
                        "wiki_batch_report",
                        article_report,
                        db_name=effective_profile.triplets_db_name,
                        profile_id=effective_profile.profile_id,
                        runtime_profile=effective_profile,
                    )
                except Exception as e:
                    st.warning(f"{article_title} batch report Run Viewer'a yazılamadı: {e}")

            totals = article_report.get("totals", {})
            article_rows.append(
                {
                    "article": article_pos,
                    "title": article_title,
                    "status": article_report.get("status"),
                    "chunks": article_preview.get("chunk_count"),
                    "final": totals.get("final_triple_count", 0),
                    "inserted": totals.get("kg_inserted_count", 0),
                    "existing": totals.get("kg_already_existing_count", 0),
                    "sample_id": article_sample_id,
                    "batch_id": article_batch_id,
                    "error": article_report.get("error"),
                }
            )
            table_box.dataframe(pd.DataFrame(article_rows), width="stretch", hide_index=True)

        multi_report = _build_wiki_multi_report(
            multi_batch_id=multi_batch_id,
            owner_user_id=user_id,
            requested_urls=wiki_multi_urls,
            article_reports=article_reports,
            model=selected_model,
            prompt_type=selected_prompt_type,
        )
        zip_bytes = _build_wiki_multi_zip(
            multi_report,
            db_name=effective_profile.triplets_db_name,
        )
        zip_filename = f"{multi_batch_id}.zip"
        st.session_state["wiki_multi_report"] = multi_report
        st.session_state["wiki_multi_zip_bytes"] = zip_bytes
        st.session_state["wiki_multi_zip_filename"] = zip_filename

        if last_multi_run_id:
            try:
                log_artifact(
                    last_multi_run_id,
                    "wiki_multi_batch_report",
                    multi_report,
                    db_name=effective_profile.triplets_db_name,
                    profile_id=effective_profile.profile_id,
                    runtime_profile=effective_profile,
                )
            except Exception as e:
                st.warning(f"Çoklu batch report Run Viewer'a yazılamadı: {e}")

        progress.progress(1.0, text="Çoklu Wikipedia batch tamamlandı.")
        if multi_report["status"] == "DONE":
            status_box.success(f"Çoklu batch tamamlandı: {len(article_reports)}/{article_total} link işlendi.")
        elif multi_report["status"] == "PARTIAL":
            status_box.warning("Çoklu batch kısmen tamamlandı; başarısız linkler rapora yazıldı.")
        else:
            status_box.error("Çoklu batch başarısız oldu.")

wiki_multi_report = st.session_state.get("wiki_multi_report")
if wiki_multi_report:
    _render_wiki_multi_report(wiki_multi_report)
    wiki_multi_zip_bytes = st.session_state.get("wiki_multi_zip_bytes")
    if wiki_multi_zip_bytes:
        st.download_button(
            "Tüm Wikipedia Link Sonuçlarını ZIP İndir",
            data=wiki_multi_zip_bytes,
            file_name=st.session_state.get("wiki_multi_zip_filename") or "wiki_multi_batch.zip",
            mime="application/zip",
            key="download_wiki_multi_batch_zip",
        )

st.divider()

btn_col_a, btn_col_b = st.columns([1, 1])
trigger          = btn_col_a.button("Extract and Visualize", width="stretch")
trigger_no_db    = btn_col_b.button("Extract (without DB)", width="stretch")

if trigger_no_db:
    if not input_text:
        st.warning("Please enter a text to extract KG.")
    else:
        import requests as _requests
        _api_url = os.getenv("WIKONTIC_API_URL", "http://localhost:8000") + "/extract"
        _payload = {
            "text":            input_text,
            "embedding_model": effective_profile.embedding_model_name.split("/")[-1].lower().replace("-", "_").replace(".", "_"),
            "llm_model":       selected_model,
            "prompt_type":     selected_prompt_type,
        }
        # Map full model name to embedding_key via profile
        _payload["embedding_model"] = next(
            (ep.embedding_key for ep in __import__(
                "src.wikontic.profiles", fromlist=["EMBEDDING_PROFILES"]
            ).EMBEDDING_PROFILES.values()
             if ep.model_name == effective_profile.embedding_model_name),
            "contriever",
        )
        with st.spinner("Extracting (no DB write)..."):
            try:
                _resp = _requests.post(_api_url, json=_payload, timeout=180)
                _resp.raise_for_status()
                _data = _resp.json()
                _triplets = _data.get("triplets", [])
                st.success(f"✅ {_data.get('count', len(_triplets))} triplets extracted (not saved to DB).")
                if _triplets:
                    import pandas as _pd
                    _df = _pd.DataFrame(_triplets)
                    _cols = ["subject", "subject_type", "relation", "object", "object_type"]
                    if "qualifiers" in _df.columns:
                        _df["qualifiers"] = _df["qualifiers"].apply(
                            lambda qs: "; ".join(
                                f"{q.get('relation','')}={q.get('object','')}"
                                for q in (qs or []) if isinstance(q, dict)
                            )
                        )
                        _cols.append("qualifiers")
                    if "kaynak_cumle" in _df.columns:
                        _cols.append("kaynak_cumle")
                    st.dataframe(
                        _df[_cols],
                        width="stretch",
                        hide_index=True,
                    )
            except _requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach Wikontic API. Start it with:\n"
                    "```\nuvicorn api:app --host 0.0.0.0 --port 8000\n```"
                )
            except Exception as _e:
                st.error(f"API call failed: {_e}")

if trigger:
    if not input_text:
        st.warning("Please enter a text to extract KG.")
    elif not selected_model:
        st.warning("Please select a model for KG extraction.")
    else:
        extractor = LLMTripletExtractor(
            model=selected_model,
            api_key=api_key,
            proxy=proxy_url,
            prompt_type=selected_prompt_type,
        )
        inference_with_db = StructuredInferenceWithDB(
            extractor=extractor,
            aligner=aligner,
            triplets_db=triplets_db,
            runtime_profile=effective_profile,
        )
        (
            initial_triplets,
            final_triplets,
            filtered_triplets,
            ontology_filtered_triplets,
            run_id,
        ) = inference_with_db.extract_triplets_with_ontology_filtering_and_add_to_db(
            text=input_text, sample_id=user_id, source_text_id=None
        )
        st.session_state["last_run_id"] = run_id
        selected_run_id = run_id

        new_entities = (
            {t["subject"] for t in final_triplets} |
            {t["object"]  for t in final_triplets}
        )
        subgraph = fetch_related_triplets(list(new_entities))
        st.success(
            f"✅ Extracted {len(final_triplets)} triplets and visualized {len(subgraph)} related ones."
        )

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.subheader("Extracted Triplets")
            visualize_initial_knowledge_graph(initial_triplets)
        with gcol2:
            st.subheader("Expanded KG Subgraph")
            visualize_knowledge_graph(subgraph, highlight_entities=new_entities)

elif _rv_navigated and selected_run_id:
    st.session_state["selected_run_id"]    = None
    st.session_state["_rv_just_navigated"] = False

    parsed_art = get_artifact(selected_run_id, "parsed_triplets", db_name=current_profile.triplets_db_name)
    final_art  = get_artifact(selected_run_id, "final_triplets",  db_name=current_profile.triplets_db_name)

    initial_from_db = parsed_art.get("triplets", []) if parsed_art else []
    final_from_db   = final_art.get("triplets",  []) if final_art  else []

    if initial_from_db or final_from_db:
        new_entities = (
            {t.get("subject") for t in final_from_db} |
            {t.get("object")  for t in final_from_db}
        )
        subgraph = fetch_related_triplets(list(new_entities), sample_id_override=_nav_sample_id)
        st.success(
            f"✅ Run Viewer'dan yüklendi: "
            f"{len(final_from_db)} final triplet, {len(subgraph)} related."
        )
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.subheader("Extracted Triplets")
            visualize_initial_knowledge_graph(initial_from_db)
        with gcol2:
            st.subheader("Expanded KG Subgraph")
            visualize_knowledge_graph(subgraph, highlight_entities=new_entities)
    else:
        st.info("Bu run için gösterilecek triplet bulunamadı.")

# ── Transparency Panel ────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Extraction Transparency")

try:
    recent_runs = list_recent_runs(
        limit=20, sample_id=user_id, db_name=current_profile.triplets_db_name
    )
except Exception as e:
    recent_runs = []
    st.error(f"Run listesi alınamadı: {e}")

if recent_runs:
    run_labels = [r["label"]  for r in recent_runs]
    run_ids    = [r["run_id"] for r in recent_runs]
    default_index = 0
    if selected_run_id and selected_run_id in run_ids:
        default_index = run_ids.index(selected_run_id)
    selected_label  = st.selectbox("Run seç:", run_labels,
                                    index=default_index, key="run_selector")
    selected_run_id = run_ids[run_labels.index(selected_label)]

render_transparency_panel(selected_run_id)

st.divider()
render_ontology_neighborhood_panel(selected_run_id)
