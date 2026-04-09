import streamlit as st
from pyvis.network import Network

# import networkx as nx
import tempfile
import os
from dotenv import load_dotenv, find_dotenv
from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
from src.wikontic.utils.openai_utils import LLMTripletExtractor
from src.wikontic.utils.structured_aligner import Aligner
from src.wikontic.profiles import DEFAULT_RUNTIME_PROFILE
from pymongo import MongoClient
import uuid
import logging
import sys
import base64

import json
from urllib.parse import quote
from urllib.request import Request, urlopen

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("PersonalKG")
logger.setLevel(logging.INFO)


st.set_page_config(
	page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)

# --- Mongo Setup ---
_ = load_dotenv(find_dotenv())
mongo_client = MongoClient(os.getenv("MONGO_URI"))
api_key = os.getenv("KEY")
proxy_url = os.getenv("PROXY_URL")

current_profile = st.session_state.get("active_runtime_profile", DEFAULT_RUNTIME_PROFILE)
ontology_db = mongo_client.get_database(current_profile.ontology_db_name)
triplets_db = mongo_client.get_database(current_profile.triplets_db_name)


@st.cache_resource
def _build_aligner(profile_id: str, ontology_db_name: str, triplets_db_name: str, embedding_model_name: str):
	return Aligner(
		ontology_db=mongo_client.get_database(ontology_db_name),
		triplets_db=mongo_client.get_database(triplets_db_name),
		embedding_model_name=embedding_model_name,
	)


aligner = _build_aligner(
	current_profile.profile_id,
	current_profile.ontology_db_name,
	current_profile.triplets_db_name,
	current_profile.embedding_model_name,
)

def fetch_wikipedia_summary(name: str) -> str:
	# Wikipedia REST summary endpoint
	title = quote(name.strip().replace(" ", "_"))
	url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
	req = Request(url, headers={"User-Agent": "Wikontic/1.0"})
	try:
		with urlopen(req, timeout=10) as resp:
			data = json.loads(resp.read().decode("utf-8"))
			return (data.get("extract") or "").strip()
	except Exception:
		return ""

def fetch_related_triplets(entities):
	collection = triplets_db.get_collection("triplets")
	query = {
		"$or": [{"subject": {"$in": entities}}, {"object": {"$in": entities}}],
		"sample_id": "personal_kg",
	}
	results = collection.find(
		query, {"_id": 0, "subject": 1, "relation": 1, "object": 1}
	)
	return [(doc["subject"], doc["relation"], doc["object"]) for doc in results]


# --- Visualize ---
def visualize_knowledge_graph(triplets, highlight_entities=None):
	net = Network(
		height="600px",
		width="100%",
		bgcolor="#ffffff",
		font_color="black",
		directed=True,
	)
	highlight_entities = highlight_entities or set()
	added_nodes = set()

	for s, r, o in triplets:
		for node in [s, o]:
			if node not in added_nodes:
				net.add_node(
					node,
					label=node,
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


# --- UI ---
with open("media/wikontic.png", "rb") as f:
	img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()

# Embed in header using HTML + Markdown
st.markdown(
	f"""
	<div style="display: flex; align-items: center;">
		<img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
		<h1 style="margin: 0;">Build your personal Knowledge Graph!</h1>
	</div>
	""",
	unsafe_allow_html=True,
)

model_options = ["google/gemini-2.5-flash-lite", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
selected_model = st.selectbox(
	"Choose a model for KG extraction:", model_options, index=0
)


# Initialize session state
if "input_text" not in st.session_state:
	st.session_state.input_text = ""

st.subheader("Input name and surname of the person you want to extract KG for")
input_text = st.text_area(
	"Enter name and surname:",
	value=st.session_state.input_text,
	placeholder="Enter name and surname of the person you want to extract KG for",
	height=68,
	key="name_surname",
)

trigger = st.button("Extract and Visualize KG for the person")

if trigger:
	if not input_text.strip():
		st.warning("Please enter name and surname of the person you want to extract KG for.")
		st.stop()

	extractor = LLMTripletExtractor(
		model=selected_model, api_key=api_key, proxy=proxy_url
	)

	personal_text = fetch_wikipedia_summary(input_text)

	# Fall back to LLM-generated summary when Wikipedia returns nothing.
	if not personal_text:
		resp = extractor.client.chat.completions.create(
			model=selected_model,
			messages=[
				{"role": "user",
				 "content": f"Write a short factual paragraph about {input_text}. If uncertain, say so explicitly."}
			],
			temperature=0,
		)
		personal_text = resp.choices[0].message.content.strip()

	logger.info(f"Personal text: {personal_text}")

	inference_with_db = StructuredInferenceWithDB(
		extractor=extractor, aligner=aligner, triplets_db=triplets_db,
		runtime_profile=current_profile,
	)

	(
		initial_triplets,
		final_triplets,
		filtered_triplets,
		ontology_filtered_triplets,
	) = inference_with_db.extract_triplets_with_ontology_filtering_and_add_to_db(
		text=personal_text, sample_id="personal_kg", source_text_id=None
	)

	new_entities = {t["subject"] for t in final_triplets} | {t["object"] for t in final_triplets}
	subgraph = fetch_related_triplets(list(new_entities))

	st.success(f"✅ Extracted {len(final_triplets)} triplets and visualized {len(subgraph)} related ones.")
	st.subheader("Expanded KG Subgraph")
	visualize_knowledge_graph(subgraph, highlight_entities=new_entities)