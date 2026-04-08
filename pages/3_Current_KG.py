import streamlit as st
from pyvis.network import Network
import tempfile
import os
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
import uuid
import logging
import sys
import base64

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("KGExtraction")
logger.setLevel(logging.INFO)

# Persist user_id across pages via session state.
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id
logger.info(f"User ID: {user_id}")

st.set_page_config(
    page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)

_ = load_dotenv(find_dotenv())
mongo_client = MongoClient(os.getenv("MONGO_URI"))
triplets_db = mongo_client.get_database("demo")


def fetch_triplets():
    collection = triplets_db.get_collection("triplets")
    query = {"sample_id": user_id}
    results = collection.find(
        query, {"_id": 0, "subject": 1, "relation": 1, "object": 1}
    )
    return [(doc["subject"], doc["relation"], doc["object"]) for doc in results]


def visualize_knowledge_graph(
    triplets,
):
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
    )
    added_nodes = set()

    for s, r, o in triplets:
        for node in [s, o]:
            if node not in added_nodes:
                net.add_node(node, label=node, color="#C7C8CC")
                added_nodes.add(node)
        net.add_edge(s, o, label=r, color="#000000")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=600, scrolling=True)
    os.remove(html_path)


with open("media/wikontic.png", "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">KG Viewer</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


subgraph = fetch_triplets()
st.success(f"✅ Retrieved {len(subgraph)} triplets.")
st.subheader("Current Knowledge Graph")
visualize_knowledge_graph(subgraph)

with st.expander("🗑 Drop Knowledge Graph", expanded=True):
    st.markdown(
        """⚠️ This action button will drop the knowledge graph built in current session."""
    )
    confirm = st.checkbox("Confirm Drop")
    drop_button = st.button("Drop")
    if confirm and drop_button:
        collection = triplets_db.get_collection("triplets")
        collection.delete_many({"sample_id": user_id})
        collection = triplets_db.get_collection("filtered_triplets")
        collection.delete_many({"sample_id": user_id})
        collection = triplets_db.get_collection("ontology_filtered_triplets")
        collection.delete_many({"sample_id": user_id})
        collection = triplets_db.get_collection("initial_triplets")
        collection.delete_many({"sample_id": user_id})
        collection = triplets_db.get_collection("entity_aliases")
        collection.delete_many({"sample_id": user_id})

        st.success("Knowledge Graph dropped.")
        logger.info(f"Knowledge Graph dropped for user {user_id}")
        st.stop()
