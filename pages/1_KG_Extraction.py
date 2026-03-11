# --- File: 0_KG_Extraction.py ---
import streamlit as st
from pyvis.network import Network

# import networkx as nx
import tempfile
import os
from dotenv import load_dotenv, find_dotenv
from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
from src.wikontic.utils.openai_utils import LLMTripletExtractor
from src.wikontic.utils.structured_aligner import Aligner
from pymongo import MongoClient
import uuid
import logging
import sys
import base64

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("KGExtraction")
logger.setLevel(logging.INFO)


# Ensure the same user_id across all pages
if "user_id" not in st.session_state:
	st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id
logger.info(f"User ID: {user_id}")

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


def fetch_related_triplets(entities):
	collection = triplets_db.get_collection("triplets")
	query = {
		"$or": [{"subject": {"$in": entities}}, {"object": {"$in": entities}}],
		"sample_id": user_id,
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


def visualize_initial_knowledge_graph(initial_triplets):
	net = Network(
		height="600px",
		width="100%",
		bgcolor="#ffffff",
		font_color="black",
		directed=True,
	)

	for t in initial_triplets:
		s, r, o = t["subject"], t["relation"], t["object"]
		logger.info(f"Initial triplet: {s} {r} {o}")
		net.add_node(s, label=s, color="#B2CD9C")
		net.add_node(o, label=o, color="#B2CD9C")
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
		<h1 style="margin: 0;">KG Extraction + Visualization</h1>
	</div>
	""",
	unsafe_allow_html=True,
)

model_options = ["google/gemini-2.5-flash-lite", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
selected_model = st.selectbox(
	"Choose a model for KG extraction:", model_options, index=0
)

# Predefined Wikipedia texts
WIKIPEDIA_TEXTS = {
	"Albert Einstein": "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, Einstein also made important contributions to quantum mechanics. His mass–energy equivalence formula E = mc², which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect.",
	"The Renaissance": "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries. It occurred after the Crisis of the Late Middle Ages and was associated with great social change. In addition to the standard periodization, proponents of a 'long Renaissance' may put its beginning in the 14th century and its end in the 17th century. The traditional view focuses more on the early modern aspects of the Renaissance and argues that it was a break from the past, but many historians today focus more on its medieval aspects and argue that it was an extension of the Middle Ages.",
	"The Great Wall of China": "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups from the Eurasian Steppe. Several walls were built from as early as the 7th century BC, with selective stretches later joined by Qin Shi Huang (220–206 BC), the first emperor of China. Little of the Qin wall remains. Later on, many successive dynasties built and maintained multiple stretches of border walls. The most well-known sections of the wall were built by the Ming dynasty (1368–1644).",
	"Shakespeare": "Shakespeare was an English playwright, poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. He is often called England's national poet and the 'Bard of Avon'. His extant works, including collaborations, consist of some 39 plays, 154 sonnets, three long narrative poems, and a few other verses, some of uncertain authorship. His plays have been translated into every major living language and are performed more often than those of any other playwright.",
	"The Industrial Revolution": "The Industrial Revolution was the transition from creating goods by hand to using machines. Its start and end are widely debated by scholars, but the period generally spanned from about 1760 to 1840. According to some, this turning point in history is responsible for an increase in population, an increase in the standard of living, and the emergence of the capitalist economy. The Industrial Revolution began in Great Britain, and many of the technological and architectural innovations were of British origin. By the mid-18th century, Britain was the world's leading commercial nation, controlling a global trading empire with colonies in North America and the Caribbean.",
}

# Initialize session state
if "input_text" not in st.session_state:
	st.session_state.input_text = ""
if "selected_predefined" not in st.session_state:
	st.session_state.selected_predefined = None

# Create two columns: left for predefined texts, right for text area
col1, col2 = st.columns([1, 2])

with col1:
	st.subheader("Text Examples")

	# Add option for custom text
	predefined_options = ["Custom Text"] + list(WIKIPEDIA_TEXTS.keys())

	# Determine initial index
	if st.session_state.selected_predefined is None:
		initial_index = 0
	elif st.session_state.selected_predefined in predefined_options:
		initial_index = predefined_options.index(st.session_state.selected_predefined)
	else:
		initial_index = 0

	selected_predefined = st.radio(
		"Choose a text option:",
		predefined_options,
		index=initial_index,
		key="predefined_selector",
	)

	# Handle selection change
	if selected_predefined != st.session_state.selected_predefined:
		st.session_state.selected_predefined = selected_predefined
		if (
			selected_predefined != "Custom Text"
			and selected_predefined in WIKIPEDIA_TEXTS
		):
			st.session_state.input_text = WIKIPEDIA_TEXTS[selected_predefined]
			st.rerun()
		elif selected_predefined == "Custom Text":
			# Don't clear text when switching to custom - let user keep their edits
			pass

with col2:
	st.subheader("Text Input")
	input_text = st.text_area(
		"Enter or modify text:",
		value=st.session_state.input_text,
		placeholder="Paste your text here or select a text option from the left...",
		height=300,
		key="text_area",
	)
	# Update session state when user manually edits
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
		) = inference_with_db.extract_triplets_with_ontology_filtering_and_add_to_db(
			text=input_text, sample_id=user_id, source_text_id=None
		)
		logger.info("Initial triplets: %s", initial_triplets)
		logger.info("-" * 100)
		logger.info("Refined triplets: %s", final_triplets)
		logger.info("-" * 100)
		logger.info("filtered_triplets: %s", filtered_triplets)
		logger.info("-" * 100)
		logger.info("ontology_filtered_triplets: %s", ontology_filtered_triplets)
		logger.info("-" * 100)
		new_entities = {t["subject"] for t in final_triplets} | {
			t["object"] for t in final_triplets
		}
		subgraph = fetch_related_triplets(list(new_entities))
		st.success(
			f"✅ Extracted {len(final_triplets)} triplets and visualized {len(subgraph)} related ones."
		)

		col1, col2 = st.columns(2)

		with col1:

			st.subheader("Extracted Triplets")
			visualize_initial_knowledge_graph(initial_triplets)

		with col2:
			st.subheader("Expanded KG Subgraph")
			visualize_knowledge_graph(subgraph, highlight_entities=new_entities)
