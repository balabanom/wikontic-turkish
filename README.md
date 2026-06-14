![Wikontic logo](/media/wikotic-wo-text.png)

# Wikontic

**Build ontology-aware, Wikidata-aligned knowledge graphs from raw text using LLMs.**

Wikontic extracts candidate knowledge graph triplets from unstructured text, aligns them with Wikidata ontology rules, validates entity/property constraints, stores the resulting graph in MongoDB, and supports graph-based QA, run inspection, replay, and export.

![Pipeline overview](/media/KG+LM-pipeline-with-background.png)

## What It Does

- Extracts triplets with an OpenAI-compatible LLM provider such as OpenRouter.
- Refines entity types and relations against Wikidata-derived ontology mappings.
- Supports English and Turkish ontology profiles.
- Supports multiple embedding models through runtime profiles.
- Stores ontology data, triplets, run metadata, artifacts, and vector indexes in MongoDB Atlas Local.
- Provides a 6-page Streamlit UI for extraction, QA, graph browsing, Wikipedia batch extraction, and run analysis.
- Provides a FastAPI endpoint for no-write extraction previews.
- Logs LLM calls and per-stage run artifacts for debugging and comparison.

## Pipeline

```text
Raw text
  -> LLM triplet extraction
  -> sentence matching
  -> ontology-aware type and relation refinement
  -> Wikidata constraint validation
  -> entity deduplication
  -> MongoDB storage
  -> QA / visualization / export
```

The main UI uses the structured ontology-aware pipeline by default. The codebase also contains older dynamic/non-ontology paths for evaluation and legacy experiments.

## Requirements

- Python 3.10+
- Docker
- MongoDB Atlas Local image, required for `$vectorSearch`
- OpenRouter API key or another OpenAI-compatible API endpoint
- Internet access on first DB initialization so HuggingFace embedding models can be downloaded

## Quick Start

Clone and install dependencies:

```bash
git clone <repo-url>
cd Wikontic
```

Run the first-time setup script:

```bash
./scripts/first_setup.sh
```

This script creates `.env` if needed, creates `.venv`, installs dependencies, starts MongoDB Atlas Local, and initializes all current UI profiles:

```text
en__contriever
en__bge_m3
en__turkish_e5_large
en__mft_random
tr__bge_m3
tr__turkish_e5_large
tr__mft_random
```

Start the app later with:

```bash
./scripts/start_wikontic.sh
```

The start script checks Docker, starts or creates the `wikontic_mongo` container if needed, waits for MongoDB, and launches Streamlit at `http://localhost:8501`.

## Manual Setup

Use this section only if you do not want to use the setup scripts.

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create your environment file:

```bash
cp .env.example .env
```

Set at least these values in `.env`:

```bash
MONGO_URI=mongodb://localhost:27018/?directConnection=true
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENAI_BASE_URL=https://openrouter.ai/api/v1
KEY=sk-or-v1-...
LLM_LOG_LEVEL=full
LLM_LOG_PATH=logs/llm_requests.jsonl
```

Optional values:

```bash
PROXY_URL=
WIKONTIC_API_URL=http://localhost:8000
```

Start MongoDB Atlas Local:

```bash
docker pull mongodb-atlas-local
docker run --name wikontic_mongo -d -p 27018:27018 mongodb-atlas-local:latest
```

Initialize the default English profile:

```bash
python init_dbs.py --profile en__contriever
```

Initialize the recommended Turkish profile:

```bash
python init_dbs.py --profile tr__turkish_e5_large
```

Launch the Streamlit app:

```bash
streamlit run Wikontic.py
```

Open `http://localhost:8501`.

## Database Initialization

`init_dbs.py` is the current setup entrypoint. It builds the ontology DB, creates profile-specific vector collections, prepares the shared triplets DB, creates vector search indexes, and writes `system_profile_metadata` used by readiness checks.

Common commands:

```bash
# Default profile: English ontology + Contriever
python init_dbs.py

# Explicit English profiles
python init_dbs.py --profile en__contriever
python init_dbs.py --profile en__mft_random
python init_dbs.py --profile en__turkish_e5_large
python init_dbs.py --profile en__bge_m3

# Explicit Turkish profiles
python init_dbs.py --profile tr__turkish_e5_large
python init_dbs.py --profile tr__mft_random
python init_dbs.py --profile tr__bge_m3

# Resume after an interrupted ontology build
python init_dbs.py --profile tr__turkish_e5_large --resume

# Only initialize triplets collections for a profile
python init_triplets_only.py --profile en__mft_random
```

Use `--drop_triplets` only when you intentionally want to recreate the model-specific entity alias collection for that profile:

```bash
python init_dbs.py --profile en__contriever --drop_triplets
```

`setup_db.sh` is legacy and should not be used for a fresh profile-based setup.

## Runtime Profiles

A runtime profile is the combination of:

- an ontology profile, such as English or Turkish
- an embedding profile, such as Contriever, Turkish E5 Large, BGE-M3, or MFT-Random

Profile IDs use this format:

```text
{runtime_key}__{embedding_key}
```

Examples:

```text
en__contriever
en__mft_random
tr__turkish_e5_large
tr__bge_m3
```

The Streamlit sidebar lets you choose the ontology profile, embedding model, ontology DB, and triplets DB. Before extraction starts, `check_profile_readiness()` verifies that the selected profile has the required collections, metadata, vector indexes, and embedding dimensions.

## Current Ontology Profiles

| Profile ID | Runtime key | Language | Status |
|---|---:|---|---|
| `ontology_en_v1` | `en` | English | Available |
| `ontology_tr_v1` | `tr` | Turkish | Available |
| `ontology_en_legacy_v1` | `en_legacy` | English | Legacy compatibility, hidden from UI by default |

The Turkish ontology uses Turkish labels and aliases under `src/wikontic/utils/ontology_mappings/tr/`, with English fallback coverage where needed.

## Current Embedding Profiles

Defined in `configs/embedding_profiles.json`:

| Profile ID | Runtime key | Model | Dim | Languages | Status |
|---|---|---|---:|---|---|
| `contriever_v1` | `contriever` | `facebook/contriever` | 768 | en | Available |
| `bge_m3_v1` | `bge_m3` | `BAAI/bge-m3` | 1024 | en, tr | Available |
| `turkish_e5_large_v1` | `turkish_e5_large` | `ytu-ce-cosmos/turkish-e5-large` | 1024 | en, tr | Available |
| `mft_random_v1` | `mft_random` | `alibayram/mft-random` | 768 | en, tr | Available |

`tr__contriever` is not valid because Contriever is registered as English-only.

## Database Layout

The current standard layout is:

```text
ontology__en
ontology__tr
triplets
```

Ontology databases are shared per language. The triplets DB is shared, while embedding-dependent vector workspaces are separated by collection name.

Examples:

```text
ontology__en.entity_type_aliases__contriever
ontology__en.property_aliases__contriever

ontology__tr.entity_type_aliases__turkish_e5_large
ontology__tr.property_aliases__turkish_e5_large

triplets.entity_aliases__contriever
triplets.entity_aliases__turkish_e5_large
triplets.triplets
triplets.initial_triplets
triplets.filtered_triplets
triplets.ontology_filtered_triplets
triplets.extraction_runs
triplets.run_artifacts
```

Legacy English + Contriever can still use:

```text
wikidata_ontology
demo
```

## Web App Pages

| Page | Purpose |
|---|---|
| `1_KG_Extraction.py` | Main extraction UI, runtime profile selection, graph visualization, Wikipedia URL batch extraction, no-write API extraction |
| `2_QA.py` | Ask questions over the current KG |
| `3_Current_KG.py` | Browse and manage the current KG |
| `4_Wikipedia_vs_Wikidata.py` | Compare Wikipedia-derived and Wikidata-derived graph outputs |
| `5_Personal_KG.py` | Build personal KGs from Wikipedia summaries |
| `6_Run_Viewer.py` | Inspect runs, artifacts, timings, comparisons, ZIP exports, and replay flows |

## FastAPI Extraction API

Run the API:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/
```

No-write extraction:

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ada Lovelace worked on Charles Babbage's Analytical Engine.",
    "embedding_model": "contriever",
    "ontology_language": "en",
    "llm_model": "gpt-4o-mini",
    "prompt_type": "temel"
  }'
```

API fields:

| Field | Values |
|---|---|
| `embedding_model` | `contriever`, `bge_m3`, `turkish_e5_large`, `mft_random` |
| `ontology_language` | `en`, `tr` |
| `prompt_type` | `temel`, `ape`, `dspy`, `textgrad` |
| `llm_model` | Any configured OpenAI/OpenRouter-compatible model |

The API uses the same ontology and embedding profile resolver as the UI, but does not write triplets or run artifacts to MongoDB.

## Prompt Modes

The extraction UI and API support:

| Mode | Description |
|---|---|
| `temel` | Default Wikontic extraction prompt |
| `ape` | Automatic Prompt Engineer optimized prompt |
| `dspy` | DSPy ChainOfThought extraction path |
| `textgrad` | TextGrad-optimized prompt |

Optimized prompt artifacts are cached under `prompts/optimized/`.

## Project Structure

```text
Wikontic/
├── Wikontic.py
├── api.py
├── init_dbs.py
├── init_triplets_only.py
├── requirements.txt
├── scripts/
│   ├── first_setup.sh
│   └── start_wikontic.sh
├── configs/
│   └── embedding_profiles.json
├── src/wikontic/
│   ├── profile_readiness.py
│   ├── create_wikidata_ontology_db.py
│   ├── create_ontological_triplets_db.py
│   ├── profiles/
│   │   ├── ontology_profiles.py
│   │   ├── embedding_profiles.py
│   │   └── runtime_profile.py
│   └── utils/
│       ├── openai_utils.py
│       ├── structured_inference_with_db.py
│       ├── structured_aligner.py
│       ├── run_logger.py
│       ├── run_reader.py
│       ├── run_exporter.py
│       ├── run_compare.py
│       ├── replay_runner.py
│       ├── paper_report.py
│       ├── wiki_extractor.py
│       ├── llm_client_logger.py
│       └── ontology_mappings/
├── pages/
├── prompts/
├── datasets/
├── inference_and_eval/
├── preprocessing/
├── analysis/
├── media/
└── logs/
```

## Evaluation Scripts

The repository includes offline evaluation scripts for HotPotQA and MuSiQue:

```bash
python inference_and_eval/hotpot_inference_with_db.py
python inference_and_eval/qa_eval_hotpot.py

python inference_and_eval/musique_inference_with_db.py
python inference_and_eval/qa_eval_musique.py
```

These scripts are older than the main UI path in some places, so check their CLI defaults before using them with a non-default runtime profile.

## Troubleshooting

If the UI says a profile is not ready, run the command shown in the sidebar, usually:

```bash
python init_dbs.py --profile <profile-id>
```

If MongoDB is unreachable, confirm the container is running and that `.env` points to port `27018`:

```bash
docker ps
```

If vector search fails, make sure you are using `mongodb-atlas-local`, not a plain MongoDB image.

If first initialization is slow, that is expected: the script embeds ontology labels and aliases and may download the selected HuggingFace model.

If a build was interrupted, prefer:

```bash
python init_dbs.py --profile <profile-id> --resume
```

## Notes

- `.env` is ignored by git and must not be committed.
- `logs/llm_requests.jsonl` can contain full prompt and response payloads when `LLM_LOG_LEVEL=full`.
- Different embedding models must not share vector collections. The runtime profile system enforces this with model-specific collection names and readiness checks.
