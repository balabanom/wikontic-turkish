![Wikontic logo](/media/wikotic-wo-text.png)

# Wikontic

**Build ontology-aware, Wikidata-aligned knowledge graphs from raw text using LLMs**

---

## Overview

Knowledge Graphs (KGs) provide structured, verifiable representations of knowledge, enabling fact grounding and empowering large language models (LLMs) with up-to-date, real-world information. However, creating high-quality KGs from open-domain text is challenging due to redundancy, inconsistency, and lack of alignment with formal ontologies.

**Wikontic** is a multi-stage pipeline for constructing ontology-aligned KGs from unstructured text using LLMs and Wikidata. It extracts candidate triplets from raw text, then refines them through ontology-based typing, schema validation, and entity deduplication — resulting in compact, semantically coherent graphs.

![Pipeline overview](/media/KG+LM-pipeline-with-background.png)

---

## Features

- **Ontology-aware extraction**: Validates entity types and property constraints against real Wikidata rules
- **Multi-profile support**: Swap embedding models and ontology languages at runtime without rebuilding the entire DB
- **Multi-hop QA**: Answer complex questions by traversing the extracted KG
- **Run management**: Full lifecycle tracking with per-stage artifacts, A/B comparison, replay, and ZIP export
- **Audit logging**: Every LLM request is logged to a JSONL file for inspection and debugging
- **Interactive web UI**: 6-page Streamlit application for extraction, QA, visualization, and analytics
- **Evaluation ready**: Scripts for HotPotQA and MuSiQue benchmarks included

---

## Pipeline Architecture

```
Raw Text
   │
   ▼
[Stage 1] LLM Triplet Extraction
   │  LLMTripletExtractor (openai_utils.py)
   │  → (subject, relation, object, subject_type, object_type, qualifiers)
   │
   ▼
[Stage 2] Sentence Matching
   │  sentence_splitter.py + sentence_matcher.py
   │  → Attach source sentence to each triplet
   │
   ▼
[Stage 3] Ontology Validation & Refinement  ← structured mode only
   │  StructuredInferenceWithDB
   │  ├─ Entity type resolution via vector search on Wikidata type hierarchy
   │  ├─ Property constraint validation (domain/range rules from Wikidata)
   │  ├─ Relation refinement: LLM ranks candidate Wikidata properties
   │  └─ Entity name refinement: LLM picks canonical names from similar entities
   │
   ▼
[Stage 4] Storage & Deduplication
   │  MongoDB (ontology DB + triplets DB)
   │  → Vector indexes for entity aliases (cosine similarity)
   │
   ▼
Final Knowledge Graph  →  QA  /  Visualization  /  Export
```

**Two extraction modes:**
- **Structured** (default): Wikidata-constrained typing, property validation, full ontology alignment
- **Dynamic**: Faster embedding-based dedup only, no ontology constraints

---

## Repository Structure

```
Wikontic/
├── Wikontic.py                          # Streamlit landing page
├── requirements.txt                     # Python dependencies
├── pyproject.toml                       # Package metadata (v0.0.7)
├── Dockerfile                           # Docker containerization
├── setup_db.sh                          # Legacy DB setup (Docker + create scripts)
├── init_dbs.py                          # Initialize ontology + triplets DB for a profile
├── init_triplets_only.py                # Initialize triplets DB only (skip ontology rebuild)
├── .env.example                         # Environment config template
│
├── configs/
│   └── embedding_profiles.json          # Embedding model registry
│
├── src/wikontic/                        # Main Python package
│   ├── profile_readiness.py             # Validates DB readiness before extraction
│   ├── create_wikidata_ontology_db.py   # Builds ontology DB from Wikidata JSON mappings
│   ├── create_ontological_triplets_db.py # Creates triplets collections + vector indexes
│   │
│   ├── profiles/                        # Runtime profile registry
│   │   ├── ontology_profiles.py         # Ontology profile definitions (en, tr, legacy)
│   │   ├── embedding_profiles.py        # Embedding profile loader (from configs/)
│   │   └── runtime_profile.py           # RuntimeProfile dataclass + resolve_runtime_profile()
│   │
│   └── utils/
│       ├── openai_utils.py              # LLMTripletExtractor — all LLM calls
│       ├── structured_inference_with_db.py  # Ontology-aware extraction pipeline
│       ├── inference_with_db.py         # Dynamic extraction pipeline (no ontology)
│       ├── structured_aligner.py        # Wikidata-constrained alignment + vector search
│       ├── dynamic_aligner.py           # Embedding-only entity/relation dedup
│       ├── run_logger.py                # Run lifecycle: start_run, log_artifact, finish_run
│       ├── run_reader.py                # Read + normalize run and artifact data
│       ├── run_exporter.py              # Export run as ZIP
│       ├── run_compare.py               # A/B diff of two runs
│       ├── replay_runner.py             # Re-execute a past run with model override
│       ├── sentence_splitter.py         # Text → sentence list with char offsets
│       ├── sentence_matcher.py          # Triplet → source sentence assignment
│       ├── timing_utils.py              # Per-stage wall-clock profiling (StageTimer)
│       ├── llm_client_logger.py         # Thread-safe JSONL audit log for LLM requests
│       ├── ontology_mappings/           # Static Wikidata JSON dumps (entity types, properties, constraints)
│       └── prompts/                     # LLM prompt templates
│           ├── triplet_extraction/
│           ├── ontology_refinement/
│           ├── name_refinement/
│           └── qa/
│
├── pages/                               # Streamlit multi-page app
│   ├── 1_KG_Extraction.py               # Main extraction + visualization
│   ├── 2_QA.py                          # Multi-hop Q&A
│   ├── 3_Current_KG.py                  # Browse / drop current KG
│   ├── 4_Wikipedia_vs_Wikidata.py       # Side-by-side KG comparison
│   ├── 5_Personal_KG.py                 # KG from Wikipedia summaries
│   └── 6_Run_Viewer.py                  # Run analytics, compare, export, replay
│
├── inference_and_eval/                  # Offline evaluation scripts
│   ├── hotpot_inference_with_db.py
│   ├── musique_inference_with_db.py
│   ├── qa_eval_hotpot.py
│   └── qa_eval_musique.py
│
├── datasets/                            # Evaluation datasets
│   ├── hotpotqa.json
│   ├── hotpotqa200.json
│   ├── musique.json
│   └── musique_200_test.json
│
├── preprocessing/
│   └── constraint-preprocessing.ipynb  # Extract Wikidata constraint rules
│
├── analysis/                            # Research & analysis notebooks
├── logs/                                # Runtime logs (llm_requests.jsonl)
└── media/                               # UI assets and screenshots
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker (for MongoDB Atlas local)
- An [OpenRouter](https://openrouter.ai) API key (or any OpenAI-compatible endpoint)

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd Wikontic
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27018/?directConnection=true` | MongoDB connection string |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | LLM API base URL |
| `KEY` | *(required)* | OpenRouter API key (`sk-or-v1-...`) |
| `PROXY_URL` | *(empty)* | Optional HTTP proxy URL |
| `LLM_LOG_LEVEL` | `full` | `full` = log full messages, `preview` = first 200 chars |
| `LLM_LOG_PATH` | `logs/llm_requests.jsonl` | Path for LLM audit log |

### 3. Start MongoDB

Wikontic requires **MongoDB Atlas local** for vector search support:

```bash
docker pull mongodb-atlas-local
docker run --name wikontic_mongo -d -p 27018:27018 mongodb-atlas-local:latest
```

### 4. Initialize databases

Initialize the ontology DB and triplets DB for your chosen profile:

```bash
# Default: English ontology + Contriever embeddings
python init_dbs.py

# Explicit profile
python init_dbs.py --profile en__contriever
python init_dbs.py --profile en__mft_random
python init_dbs.py --profile en__turkish_e5_large

# To reset triplets data (WARNING: deletes all extracted KG data)
python init_dbs.py --profile en__contriever --drop_triplets
```

To initialize only the triplets DB for a profile (skips ontology rebuild):

```bash
python init_triplets_only.py --profile en__mft_random
```

### 5. Launch the web app

```bash
streamlit run Wikontic.py
```

Open `http://localhost:8501` in your browser.

![Demo overview](/media/demo-with-background.png)

---

## Profile System

Wikontic uses a **profile registry** to manage combinations of ontology language and embedding model. Every combination gets its own isolated MongoDB database — no data cross-contamination between profiles.

### Ontology Profiles

| Profile ID | Language | Display Name | Runtime Key | Available |
|---|---|---|---|---|
| `ontology_en_v1` | English | English Ontology | `en` | Yes |
| `ontology_en_legacy_v1` | English | English Ontology (Legacy DB) | `en_legacy` | Yes |
| `ontology_tr_v1` | Turkish | Turkish Ontology | `tr` | Not yet |

### Embedding Profiles

| Profile ID | Model | Dimension | Languages | Available |
|---|---|---|---|---|
| `contriever_v1` | `facebook/contriever` | 768 | en | Yes (default) |
| `mft_random_v1` | `alibayram/mft-random` | 768 | en, tr | Yes |
| `turkish_e5_large_v1` | `ytu-ce-cosmos/turkish-e5-large` | 1024 | en, tr | Yes |
| `bge_m3_v1` | `BAAI/bge-m3` | 1024 | en, tr | Requires install |

New embedding models can be added by editing `configs/embedding_profiles.json`.

### Database Naming

Databases are named deterministically from the profile:

```
ontology__{runtime_key}__{embedding_key}   →   ontology__en__contriever
triplets__{runtime_key}__{embedding_key}   →   triplets__en__mft_random
```

The legacy English profile uses fixed legacy names (`wikidata_ontology` / `demo`) for backward compatibility.

---

## Web Application Pages

| Page | Description |
|---|---|
| **1 — KG Extraction** | Enter text, select ontology + embedding profile, run extraction, visualize the resulting knowledge graph |
| **2 — QA** | Ask multi-hop questions over the extracted KG; see supporting triplets and entity links |
| **3 — Current KG** | Browse all triplets in your current KG; drop KG with confirmation |
| **4 — Wikipedia vs Wikidata** | Side-by-side comparison of two knowledge graphs from fixed sample IDs |
| **5 — Personal KG** | Build a KG from Wikipedia article summaries; look up entity context |
| **6 — Run Viewer** | Filter and browse extraction runs; compare two runs A/B; export as ZIP; replay with a different model |

---

## Evaluation

Run offline evaluation against multi-hop QA benchmarks:

```bash
# Build KGs and run QA on HotPotQA
python inference_and_eval/hotpot_inference_with_db.py
python inference_and_eval/qa_eval_hotpot.py

# Build KGs and run QA on MuSiQue
python inference_and_eval/musique_inference_with_db.py
python inference_and_eval/qa_eval_musique.py
```

Datasets are in `datasets/` (HotPotQA and MuSiQue, 200-sample evaluation splits included).

---

## Technology Stack

| Layer | Technology |
|---|---|
| Web UI | Streamlit, PyVis |
| LLM | OpenAI / OpenRouter (GPT-4o, GPT-4.1, Llama 3.3, Qwen3) |
| Embeddings | HuggingFace Transformers (Contriever, BGE-M3, E5, MFT-Random) |
| Vector DB | MongoDB Atlas Vector Search (cosine similarity) |
| Retry / Resilience | Tenacity (exponential backoff) |
| Data Validation | Pydantic |
| Config | python-dotenv |
| Packaging | Docker, pyproject.toml |

---

Enjoy building knowledge graphs with Wikontic!
