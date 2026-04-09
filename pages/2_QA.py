import streamlit as st
from pyvis.network import Network
import networkx as nx
import tempfile
import os
from dataclasses import replace
from dotenv import load_dotenv, find_dotenv

from pymongo import MongoClient
from src.wikontic.utils.structured_aligner import Aligner
from src.wikontic.utils.openai_utils import LLMTripletExtractor
from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
from src.wikontic.profiles import DEFAULT_RUNTIME_PROFILE
import uuid
import logging
import sys
import base64

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("QA")
logger.setLevel(logging.ERROR)


# Persist user_id across pages via session state.
if "user_id" not in st.session_state:
	st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id

logger.info(f"User ID: {user_id}")

_ = load_dotenv(find_dotenv())

mongo_client = MongoClient(os.getenv("MONGO_URI"))
api_key = os.getenv("KEY")
proxy_url = os.getenv("PROXY_URL")

current_profile = st.session_state.get("active_runtime_profile", DEFAULT_RUNTIME_PROFILE)
ontology_db_override = st.session_state.get("ontology_db_override_name")
triplets_db_override = st.session_state.get("triplets_db_override_name")
effective_profile = (
	replace(
		current_profile,
		ontology_db_name=ontology_db_override or current_profile.ontology_db_name,
		triplets_db_name=triplets_db_override or current_profile.triplets_db_name,
	)
	if (ontology_db_override or triplets_db_override)
	else current_profile
)


@st.cache_resource
def _build_aligner(profile_id: str, ontology_db_name: str, triplets_db_name: str, embedding_model_name: str):
	ontology_db = mongo_client.get_database(ontology_db_name)
	triplets_db = mongo_client.get_database(triplets_db_name)
	return Aligner(
		ontology_db=ontology_db,
		triplets_db=triplets_db,
		embedding_model_name=embedding_model_name,
	)


aligner = _build_aligner(
	effective_profile.profile_id,
	effective_profile.ontology_db_name,
	effective_profile.triplets_db_name,
	effective_profile.embedding_model_name,
)
triplets_db = mongo_client.get_database(effective_profile.triplets_db_name)

st.set_page_config(
	page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)


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

	for t in triplets:
		s, r, o = t["subject"], t["relation"], t["object"]
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


def query_kg(inferer, question_text):
	identified_entities = inferer.identify_relevant_entities_from_question_with_llm(
		question_text, sample_id=user_id
	)
	supporting_triplets, ans = inferer.answer_question_with_llm(
		question_text, identified_entities, sample_id=user_id
	)
	return identified_entities, supporting_triplets, ans


with open("media/wikontic.png", "rb") as f:
	img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()

st.markdown(
	f"""
	<div style="display: flex; align-items: center;">
		<img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
		<h1 style="margin: 0;">Question Answering with KG</h1>
	</div>
	""",
	unsafe_allow_html=True,
)


model_options = ["google/gemini-2.5-flash-lite", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
selected_model = st.selectbox("Choose a model for QA:", model_options, index=0)
question = st.text_input("Ask a question about the Knowledge Graph")
trigger = st.button("Answer question")


if trigger:
	if not question:
		st.warning("Please enter a question.")
	elif not selected_model:
		st.warning("Please select a model.")
	else:
		extractor = LLMTripletExtractor(
			model=selected_model, api_key=api_key, proxy=proxy_url
		)
		inferer = StructuredInferenceWithDB(
			extractor=extractor, aligner=aligner, triplets_db=triplets_db,
			runtime_profile=effective_profile,
		)

		st.markdown(f"#### Results for: *{question}*")
		identified_entities_names, supporting_triplets, ans = query_kg(
			inferer, question
		)

		st.session_state.kg = nx.DiGraph()
		for t in supporting_triplets:
			s, r, o = t["subject"], t["relation"], t["object"]
			st.session_state.kg.add_edge(
				s,
				o,
				label=r,
				highlight=s in identified_entities_names
				or o in identified_entities_names,
			)

		st.success(f"✅ Extracted {len(supporting_triplets)} supporting triplets.")

		st.subheader("Relevant Subgraph")
		st.markdown(
			"""
		- 🟢 <span style='color:#B2CD9C'>**Highlighted Entity**</span> – relevant node from your query  
		- ⚪ <span style='color:#C7C8CC'>**Regular Entity**</span> – node from KG  connected to one of the nodes from your query
		""",
			unsafe_allow_html=True,
		)
		visualize_knowledge_graph(
			supporting_triplets, highlight_entities=identified_entities_names
		)

		st.subheader("Answer")
		st.markdown(
			f"""
		<div style='background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745;'>
		✅ Answer to the question is <strong>{ans}</strong>
		</div>
		""",
			unsafe_allow_html=True,
		)
