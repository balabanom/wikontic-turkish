# --- File: 1_KG_Extraction.py ---
import streamlit as st
from pyvis.network import Network
import tempfile
import os
from dotenv import load_dotenv, find_dotenv
from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
from src.wikontic.utils.openai_utils import LLMTripletExtractor
from src.wikontic.utils.structured_aligner import Aligner
from src.wikontic.utils.run_reader import get_run, get_artifact, list_recent_runs
from pymongo import MongoClient
import uuid
import logging
import sys
import base64
import pandas as pd

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("KGExtraction")
logger.setLevel(logging.INFO)

st.set_page_config(
    page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)

WIKIDATA_ONTOLOGY_DB_NAME = "wikidata_ontology"
TRIPLETS_DB_NAME = "demo"

# --- Mongo Setup ---
_ = load_dotenv(find_dotenv())
mongo_client = MongoClient(os.getenv("MONGO_URI"))
api_key = os.getenv("KEY")
proxy_url = os.getenv("PROXY_URL")
ontology_db = mongo_client.get_database(WIKIDATA_ONTOLOGY_DB_NAME)
triplets_db = mongo_client.get_database(TRIPLETS_DB_NAME)

aligner = Aligner(ontology_db=ontology_db, triplets_db=triplets_db)

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

# ── Kullanıcı adı girişi ──────────────────────────────────────────────────────
# Aynı kullanıcı adı → aynı KG, uygulama kapansa bile.
# user_id olarak doğrudan username kullanılır (boşluk trim edilir).

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
            # Kullanıcı adı değişti — session'ı sıfırla
            st.session_state["user_id"]          = username_input
            st.session_state["last_run_id"]      = None
            st.session_state["input_text"]       = ""
            st.session_state["selected_predefined"] = None
            st.rerun()
        user_id = username_input
        st.success(f"KG: **{user_id}**")
    else:
        # Kullanıcı adı girilmemişse geçici UUID kullan (session boyunca sabit)
        if not st.session_state["user_id"]:
            st.session_state["user_id"] = f"guest_{str(uuid.uuid4())[:8]}"
        user_id = st.session_state["user_id"]
        st.info("Kalıcı KG için bir kullanıcı adı gir.")

logger.info(f"User ID: {user_id}")


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def fetch_related_triplets(entities, sample_id_override: str | None = None):
    """
    demo.triplets koleksiyonundan entity'lerle ilgili tripletleri çeker.
    sample_id_override: Run Viewer'dan gelince run'ın kendi sample_id'si geçilir.
    """
    sid = sample_id_override if sample_id_override else user_id
    collection = triplets_db.get_collection("triplets")
    query = {
        "$or": [{"subject": {"$in": entities}}, {"object": {"$in": entities}}],
        "sample_id": sid,
    }
    results = collection.find(
        query, {"_id": 0, "subject": 1, "relation": 1, "object": 1}
    )
    return [(doc["subject"], doc["relation"], doc["object"]) for doc in results]


def visualize_knowledge_graph(triplets, highlight_entities=None):
    net = Network(
        height="600px", width="100%", bgcolor="#ffffff",
        font_color="black", directed=True,
    )
    highlight_entities = highlight_entities or set()
    added_nodes = set()

    for s, r, o in triplets:
        for node in [s, o]:
            if node not in added_nodes:
                net.add_node(
                    node, label=node,
                    color="#B2CD9C" if node in highlight_entities else "#C7C8CC",
                )
                added_nodes.add(node)
        net.add_edge(s, o, label=r, color="#000000")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=600, scrolling=True)
    os.remove(html_path)


def visualize_initial_knowledge_graph(initial_triplets):
    net = Network(
        height="600px", width="100%", bgcolor="#ffffff",
        font_color="black", directed=True,
    )
    for t in initial_triplets:
        s, r, o = t["subject"], t["relation"], t["object"]
        net.add_node(s, label=s, color="#B2CD9C")
        net.add_node(o, label=o, color="#B2CD9C")
        net.add_edge(s, o, label=r, color="#000000")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=600, scrolling=True)
    os.remove(html_path)


def visualize_ontology_neighborhood(neighborhood: dict):
    net = Network(
        height="500px", width="100%", bgcolor="#ffffff",
        font_color="black", directed=True,
    )
    center = neighborhood["center"]
    net.add_node(center["id"], label=f"{center['label']}\n({center['id']})",
                 color="#4A90D9", size=25)

    for parent in neighborhood.get("parents", []):
        net.add_node(parent["id"], label=f"{parent['label']}\n({parent['id']})",
                     color="#F5A623", size=18)
        net.add_edge(center["id"], parent["id"], label="is a",
                     color="#F5A623", dashes=True)

    for prop in neighborhood.get("properties", []):
        prop_node_id = f"prop_{prop['id']}"
        color = "#5CB85C" if prop["direction"] == "subject" else "#9B59B6"
        direction_label = "→ subject" if prop["direction"] == "subject" else "← object"
        net.add_node(prop_node_id,
                     label=f"{prop['label']}\n({prop['id']})\n{direction_label}",
                     color=color, size=14, shape="box")
        net.add_edge(center["id"], prop_node_id, label=prop["label"], color=color)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=500, scrolling=True)
    os.remove(html_path)


def render_transparency_panel(selected_run_id: str):
    if not selected_run_id:
        st.info("Henüz extraction yapılmadı.")
        return

    run_meta = get_run(selected_run_id)
    if run_meta:
        meta_cols = st.columns(4)
        meta_cols[0].metric("Status", run_meta.get("status", "—"))
        meta_cols[1].metric("Model", run_meta.get("model", "—"))
        created_at = run_meta.get("created_at", "")
        if hasattr(created_at, "strftime"):
            created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")
        meta_cols[2].metric("Created At", str(created_at))
        meta_cols[3].metric("Sample ID", str(run_meta.get("sample_id", "—"))[:12] + "…")
        if run_meta.get("status") == "FAILED" and run_meta.get("error"):
            st.error(f"Run hatası: {run_meta['error']}")
    else:
        st.warning(f"Run bulunamadı: `{selected_run_id}`")
        return

    st.caption(f"🔑 Run ID: `{selected_run_id}`")

    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 Raw-0: LLM Output", "🟡 Raw-1: Parsed Triplets",
        "🔀 Merge Log", "🚫 Filtered Out", "🟢 Final Triplets",
    ])

    with tab0:
        art = get_artifact(selected_run_id, "raw_llm_output")
        if art is None:
            st.warning("Bu stage için kayıt bulunamadı.")
        else:
            with st.expander("Ham LLM Çıktısı", expanded=True):
                st.code(art.get("text", ""), language="json")

    with tab1:
        art = get_artifact(selected_run_id, "parsed_triplets")
        if art is None:
            st.warning("Bu stage için kayıt bulunamadı.")
        else:
            triplets = art.get("triplets", [])
            st.caption(f"**{art.get('count', len(triplets))} triplet** parse edildi")
            if triplets:
                st.dataframe(pd.DataFrame(triplets)[["subject", "relation", "object"]],
                             use_container_width=True, hide_index=True)
            else:
                st.info("Parse edilmiş triplet bulunamadı.")

    with tab2:
        art = get_artifact(selected_run_id, "merge_map_entities")
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
                st.dataframe(df[existing], use_container_width=True, hide_index=True)

    with tab3:
        art = get_artifact(selected_run_id, "filtered_out")
        if art is None:
            st.warning("Bu stage için kayıt bulunamadı.")
        else:
            triplets = art.get("triplets", [])
            total = art.get("count", len(triplets))
            if total == 0:
                st.success("Bu run'da hiçbir triplet elenmedi.")
            else:
                fc = st.columns(3)
                fc[0].metric("🚫 Toplam Elenen", total)
                fc[1].metric("⚠️ Pipeline Exception", art.get("pipeline_exception_count", 0))
                fc[2].metric("🔴 Ontology Violation", art.get("ontology_filtered_count", 0))
                df = pd.DataFrame(triplets)
                cols = ["subject", "relation", "object", "reason_code", "filter_stage", "exception_text"]
                st.dataframe(df[[c for c in cols if c in df.columns]],
                             use_container_width=True, hide_index=True)

    with tab4:
        art = get_artifact(selected_run_id, "final_triplets")
        if art is None:
            st.warning("Bu stage için kayıt bulunamadı.")
        else:
            triplets = art.get("triplets", [])
            count = art.get("count", len(triplets))
            fc = st.columns(3)
            fc[0].metric("✅ Final", count)
            if art.get("filtered_count") is not None:
                fc[1].metric("⚠️ Filtered", art["filtered_count"])
            if art.get("ontology_filtered_count") is not None:
                fc[2].metric("🚫 Ontology Filtered", art["ontology_filtered_count"])
            if triplets:
                df = pd.DataFrame(triplets)
                cols = ["subject", "relation", "object", "subject_type", "object_type"]
                st.dataframe(df[[c for c in cols if c in df.columns]],
                             use_container_width=True, hide_index=True)
            else:
                st.info("Final triplet bulunamadı.")


def render_ontology_neighborhood_panel(selected_run_id: str):
    st.subheader("🗺️ Ontoloji Neighborhood")
    entity_types = []
    if selected_run_id:
        art = get_artifact(selected_run_id, "final_triplets")
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
    if selected_type:
        with st.spinner(f"'{selected_type}' için ontoloji neighborhood yükleniyor..."):
            neighborhood = aligner.get_ontology_neighborhood(selected_type)

        if neighborhood is None:
            st.warning(f"'{selected_type}' için ontoloji verisi bulunamadı.")
            return

        center     = neighborhood["center"]
        parents    = neighborhood.get("parents", [])
        properties = neighborhood.get("properties", [])

        nc = st.columns(3)
        nc[0].metric("🔵 Merkez Type", center["label"])
        nc[1].metric("🟠 Parent Sayısı", len(parents))
        nc[2].metric("🟢 Property Sayısı", len(properties))

        visualize_ontology_neighborhood(neighborhood)

        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**Parent Types**")
            if parents:
                df = pd.DataFrame(parents)[["label", "id"]]
                df.columns = ["Label", "Wikidata ID"]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Parent type bulunamadı.")
        with dc2:
            st.markdown("**Allowed Properties**")
            if properties:
                df = pd.DataFrame(properties)[["label", "id", "direction"]]
                df.columns = ["Label", "Wikidata ID", "Direction"]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Property bulunamadı.")


# ── Model seçimi ──────────────────────────────────────────────────────────────
model_options = ["google/gemini-2.5-flash-lite", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
selected_model = st.selectbox("Choose a model for KG extraction:", model_options, index=0)

WIKIPEDIA_TEXTS = {
    "Albert Einstein": "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, Einstein also made important contributions to quantum mechanics. His mass–energy equivalence formula E = mc², which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect.",
    "The Renaissance": "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries. It occurred after the Crisis of the Late Middle Ages and was associated with great social change. In addition to the standard periodization, proponents of a 'long Renaissance' may put its beginning in the 14th century and its end in the 17th century. The traditional view focuses more on the early modern aspects of the Renaissance and argues that it was a break from the past, but many historians today focus more on its medieval aspects and argue that it was an extension of the Middle Ages.",
    "The Great Wall of China": "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups from the Eurasian Steppe. Several walls were built from as early as the 7th century BC, with selective stretches later joined by Qin Shi Huang (220–206 BC), the first emperor of China. Little of the Qin wall remains. Later on, many successive dynasties built and maintained multiple stretches of border walls. The most well-known sections of the wall were built by the Ming dynasty (1368–1644).",
    "Shakespeare": "Shakespeare was an English playwright, poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. He is often called England's national poet and the 'Bard of Avon'. His extant works, including collaborations, consist of some 39 plays, 154 sonnets, three long narrative poems, and a few other verses, some of uncertain authorship. His plays have been translated into every major living language and are performed more often than those of any other playwright.",
    "The Industrial Revolution": "The Industrial Revolution was the transition from creating goods by hand to using machines. Its start and end are widely debated by scholars, but the period generally spanned from about 1760 to 1840. According to some, this turning point in history is responsible for an increase in population, an increase in the standard of living, and the emergence of the capitalist economy. The Industrial Revolution began in Great Britain, and many of the technological and architectural innovations were of British origin. By the mid-18th century, Britain was the world's leading commercial nation, controlling a global trading empire with colonies in North America and the Caribbean.",
}

if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "selected_predefined" not in st.session_state:
    st.session_state.selected_predefined = None

# ── selected_run_id'yi ERKEN hesapla ─────────────────────────────────────────
_rv_navigated: bool = bool(st.session_state.get("selected_run_id"))

_last_run_id_early: str | None = st.session_state.get("last_run_id")
if _rv_navigated:
    _last_run_id_early = st.session_state["selected_run_id"]

try:
    _recent_runs_early = list_recent_runs(limit=20, sample_id=user_id)
except Exception:
    _recent_runs_early = []

if _recent_runs_early:
    _run_ids_early = [r["run_id"] for r in _recent_runs_early]
    _default_early = (
        _run_ids_early.index(_last_run_id_early)
        if _last_run_id_early in _run_ids_early
        else 0
    )
    selected_run_id: str | None = _run_ids_early[_default_early]
else:
    selected_run_id = _last_run_id_early

_nav_sample_id: str | None = None
if _rv_navigated and selected_run_id:
    _nav_meta = get_run(selected_run_id)
    _nav_sample_id = (_nav_meta or {}).get("sample_id") or user_id
# ─────────────────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Text Examples")
    predefined_options = ["Custom Text"] + list(WIKIPEDIA_TEXTS.keys())

    if st.session_state.selected_predefined is None:
        initial_index = 0
    elif st.session_state.selected_predefined in predefined_options:
        initial_index = predefined_options.index(st.session_state.selected_predefined)
    else:
        initial_index = 0

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

trigger = st.button("Extract and Visualize")

if trigger:
    if not input_text:
        st.warning("Please enter a text to extract KG.")
    elif not selected_model:
        st.warning("Please select a model for KG extraction.")
    else:
        extractor = LLMTripletExtractor(
            model=selected_model, api_key=api_key, proxy=proxy_url
        )
        inference_with_db = StructuredInferenceWithDB(
            extractor=extractor, aligner=aligner, triplets_db=triplets_db
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

        logger.info("Initial triplets: %s", initial_triplets)
        logger.info("Refined triplets: %s", final_triplets)
        logger.info("filtered_triplets: %s", filtered_triplets)
        logger.info("ontology_filtered_triplets: %s", ontology_filtered_triplets)

        new_entities = {t["subject"] for t in final_triplets} | {
            t["object"] for t in final_triplets
        }
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

    parsed_art = get_artifact(selected_run_id, "parsed_triplets")
    final_art  = get_artifact(selected_run_id, "final_triplets")

    initial_from_db = parsed_art.get("triplets", []) if parsed_art else []
    final_from_db   = final_art.get("triplets", [])  if final_art  else []

    if initial_from_db or final_from_db:
        new_entities = (
            {t.get("subject") for t in final_from_db} |
            {t.get("object")  for t in final_from_db}
        )
        subgraph = fetch_related_triplets(
            list(new_entities),
            sample_id_override=_nav_sample_id,
        )
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

# ── Şeffaflık Paneli ──────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Extraction Transparency")

try:
    recent_runs = list_recent_runs(limit=20, sample_id=user_id)
except Exception as e:
    recent_runs = []
    st.error(f"Run listesi alınamadı: {e}")

if recent_runs:
    run_labels = [r["label"] for r in recent_runs]
    run_ids    = [r["run_id"] for r in recent_runs]

    default_index = 0
    if selected_run_id and selected_run_id in run_ids:
        default_index = run_ids.index(selected_run_id)

    selected_label = st.selectbox(
        "Run seç:", run_labels, index=default_index, key="run_selector",
    )
    selected_run_id = run_ids[run_labels.index(selected_label)]

render_transparency_panel(selected_run_id)

st.divider()
render_ontology_neighborhood_panel(selected_run_id)