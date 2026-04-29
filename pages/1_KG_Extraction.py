import streamlit as st
from pyvis.network import Network
import tempfile
import os
from dataclasses import replace
from dotenv import load_dotenv, find_dotenv
from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
from src.wikontic.utils.openai_utils import LLMTripletExtractor
from src.wikontic.utils.structured_aligner import Aligner
from src.wikontic.utils.run_reader import get_run, get_artifact, list_recent_runs
from src.wikontic.profiles import (
    resolve_runtime_profile,
    DEFAULT_RUNTIME_PROFILE,
    ONTOLOGY_PROFILES,
    get_available_ontology_profiles,
    get_compatible_embedding_profiles,
    get_unavailable_embedding_profiles,
)
from src.wikontic.profile_readiness import check_profile_readiness
from pymongo import MongoClient
import uuid
import logging
import sys
import base64
import pandas as pd

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("KGExtraction")
logger.setLevel(logging.INFO)

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

    # Show unavailable profiles as informational note
    unavailable = [p for p in ONTOLOGY_PROFILES.values() if not p.available]
    if unavailable:
        st.caption(
            "🔒 Unavailable: " + ", ".join(p.display_name for p in unavailable)
        )

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
                 "Runs from different embedding models are stored in separate databases.",
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

    try:
        available_db_names = sorted(mongo_client.list_database_names())
    except Exception:
        available_db_names = []

    # Optional ontology DB override (keeps runtime profile default unless explicitly changed)
    ontology_db_options = [current_profile.ontology_db_name]
    if "wikidata_ontology" not in ontology_db_options:
        ontology_db_options.append("wikidata_ontology")
    ontology_db_options.extend(
        db for db in available_db_names
        if db.startswith("ontology__") and db not in ontology_db_options
    )

    stored_override_db = st.session_state.get("ontology_db_override_name")
    default_selected_db = stored_override_db or current_profile.ontology_db_name
    if default_selected_db not in ontology_db_options:
        ontology_db_options.append(default_selected_db)

    selected_ontology_db = st.selectbox(
        "Ontology DB:",
        ontology_db_options,
        index=ontology_db_options.index(default_selected_db),
        key="sidebar_ontology_db_selector",
        help="Use profile default DB, or override with 'wikidata_ontology'.",
    )
    st.session_state["ontology_db_override_name"] = (
        None if selected_ontology_db == current_profile.ontology_db_name else selected_ontology_db
    )
    is_external_ontology_override = st.session_state["ontology_db_override_name"] is not None

    # Optional triplets DB override
    triplets_db_options = [current_profile.triplets_db_name]
    triplets_db_options.extend(
        db for db in available_db_names
        if (db == "triplets" or db.startswith("triplets__") or db == "demo") and db not in triplets_db_options
    )
    stored_triplets_override = st.session_state.get("triplets_db_override_name")
    default_triplets_db = stored_triplets_override or current_profile.triplets_db_name
    if default_triplets_db not in triplets_db_options:
        triplets_db_options.append(default_triplets_db)

    selected_triplets_db = st.selectbox(
        "Triplets DB:",
        triplets_db_options,
        index=triplets_db_options.index(default_triplets_db),
        key="sidebar_triplets_db_selector",
        help="Use profile default triplets DB, or manually pick another workspace DB.",
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
def _build_aligner(profile_id: str, ontology_db_name: str, triplets_db_name: str, embedding_model_name: str):
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
            st.warning(f"'{selected_type}' için ontoloji verisi bulunamadı.")
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
model_options  = ["google/gemini-2.5-flash-lite", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
selected_model = st.selectbox("Choose a model for KG extraction:", model_options, index=0)

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
                    st.dataframe(
                        _pd.DataFrame(_triplets)[
                            ["subject", "subject_type", "relation", "object", "object_type"]
                        ],
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
        extractor = LLMTripletExtractor(model=selected_model, api_key=api_key, proxy=proxy_url)
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
