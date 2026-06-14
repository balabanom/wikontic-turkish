#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MONGO_CONTAINER="${MONGO_CONTAINER:-wikontic}"
LEGACY_MONGO_CONTAINER="${LEGACY_MONGO_CONTAINER:-wikontic_mongo}"
MONGO_IMAGE="${MONGO_IMAGE:-mongodb/mongodb-atlas-local:latest}"
MONGO_HOST_PORT="${MONGO_HOST_PORT:-27018}"
MONGO_CONTAINER_PORT="${MONGO_CONTAINER_PORT:-27017}"

PROFILES=(
  "en__contriever"
  "en__bge_m3"
  "en__turkish_e5_large"
  "en__mft_random"
  "tr__bge_m3"
  "tr__turkish_e5_large"
  "tr__mft_random"
)

log() {
  printf "\n==> %s\n" "$1"
}

ok() {
  printf "OK: %s\n" "$1"
}

warn() {
  printf "\nWARNING: %s\n" "$1" >&2
}

ensure_python() {
  log "Checking Python"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required but was not found." >&2
    exit 1
  fi
  ok "python3 found: $(python3 --version)"
}

ensure_env_file() {
  if [[ ! -f ".env" ]]; then
    log "Creating .env from .env.example"
    cp .env.example .env
    ok ".env created"
  else
    log ".env already exists; keeping it"
    ok ".env found"
  fi

  if grep -q "KEY=sk-or-v1-xxxxxxxx" .env; then
    if [[ -t 0 ]]; then
      printf "OpenRouter API key (press Enter to skip for now): "
      read -r OPENROUTER_KEY
      if [[ -n "${OPENROUTER_KEY}" ]]; then
        log "Writing OpenRouter API key to .env"
        python3 - "$OPENROUTER_KEY" <<'PY'
from pathlib import Path
import sys

env_path = Path(".env")
key = sys.argv[1]
lines = env_path.read_text(encoding="utf-8").splitlines()
out = []
replaced = False
for line in lines:
    if line.startswith("KEY="):
        out.append(f"KEY={key}")
        replaced = True
    else:
        out.append(line)
if not replaced:
    out.append(f"KEY={key}")
env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
        ok "API key written"
      else
        warn "No API key set. DB setup can continue, but extraction needs KEY in .env."
      fi
    else
      warn "KEY in .env still looks like the placeholder. Edit .env before extraction."
    fi
  fi
}

ensure_venv() {
  if [[ ! -d ".venv" ]]; then
    log "Creating Python virtual environment"
    python3 -m venv .venv
    ok ".venv created"
  else
    log ".venv already exists"
    ok ".venv found"
  fi

  log "Upgrading pip"
  .venv/bin/python -m pip install --upgrade pip
  ok "pip upgraded"

  log "Installing Python dependencies from requirements.txt"
  .venv/bin/pip install -r requirements.txt
  ok "Python dependencies installed"
}

wait_for_docker() {
  local attempts=60

  log "Checking Docker daemon"
  if docker info >/dev/null 2>&1; then
    ok "Docker is already running"
    return 0
  fi

  if [[ "$(uname -s)" == "Darwin" ]]; then
    log "Docker is not running; opening Docker Desktop"
    open -a Docker || true
  else
    warn "Docker daemon is not running. Start Docker, then rerun this script."
  fi

  log "Waiting for Docker daemon"
  for attempt in $(seq 1 "$attempts"); do
    if docker info >/dev/null 2>&1; then
      ok "Docker is running"
      return 0
    fi
    printf "   Docker not ready yet (%s/%s)\n" "$attempt" "$attempts"
    sleep 2
  done

  echo "Docker did not become ready in time." >&2
  exit 1
}

ensure_mongo_container() {
  wait_for_docker

  log "Pulling MongoDB Atlas Local image"
  docker pull "$MONGO_IMAGE"
  ok "Mongo image is available: $MONGO_IMAGE"

  if docker ps --format '{{.Names}}' | grep -qx "$MONGO_CONTAINER"; then
    log "Mongo container '$MONGO_CONTAINER' is already running"
    ok "Mongo container running"
    return 0
  fi

  if docker ps --format '{{.Names}}' | grep -qx "$LEGACY_MONGO_CONTAINER"; then
    log "Legacy Mongo container '$LEGACY_MONGO_CONTAINER' is already running"
    ok "Mongo container running"
    return 0
  fi

  if docker ps -a --format '{{.Names}}' | grep -qx "$MONGO_CONTAINER"; then
    log "Starting existing Mongo container '$MONGO_CONTAINER'"
    docker start "$MONGO_CONTAINER" >/dev/null
    ok "Mongo container started"
    return 0
  fi

  if docker ps -a --format '{{.Names}}' | grep -qx "$LEGACY_MONGO_CONTAINER"; then
    log "Starting existing legacy Mongo container '$LEGACY_MONGO_CONTAINER'"
    docker start "$LEGACY_MONGO_CONTAINER" >/dev/null
    ok "Mongo container started"
    return 0
  fi

  log "Creating Mongo container '$MONGO_CONTAINER'"
  docker run \
    --name "$MONGO_CONTAINER" \
    -d \
    -p "${MONGO_HOST_PORT}:${MONGO_CONTAINER_PORT}" \
    "$MONGO_IMAGE" >/dev/null
  ok "Mongo container created and started"
}

wait_for_mongo() {
  log "Waiting for MongoDB on localhost:${MONGO_HOST_PORT}"
  for attempt in $(seq 1 60); do
    if .venv/bin/python - <<'PY' >/dev/null 2>&1
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(".env")
uri = os.getenv("MONGO_URI", "mongodb://localhost:27018/?directConnection=true")
client = MongoClient(uri, serverSelectionTimeoutMS=1000)
client.admin.command("ping")
PY
    then
      ok "MongoDB is reachable"
      return 0
    fi
    printf "   MongoDB not ready yet (%s/60)\n" "$attempt"
    sleep 2
  done

  echo "MongoDB did not become ready in time." >&2
  exit 1
}

profile_ready() {
  local profile="$1"
  .venv/bin/python - "$profile" <<'PY'
import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient

from src.wikontic.profile_readiness import check_profile_readiness
from src.wikontic.profiles import ONTOLOGY_PROFILES, EMBEDDING_PROFILES, resolve_runtime_profile

profile_id = sys.argv[1]
load_dotenv(".env")

runtime_profile = None
for ontology_id, ontology in ONTOLOGY_PROFILES.items():
    for embedding_id, embedding in EMBEDDING_PROFILES.items():
        candidate_id = f"{ontology.runtime_key}__{embedding.embedding_key}"
        if candidate_id == profile_id:
            runtime_profile = resolve_runtime_profile(ontology_id, embedding_id)
            break
    if runtime_profile is not None:
        break

if runtime_profile is None:
    print(f"Unknown profile: {profile_id}", file=sys.stderr)
    sys.exit(2)

client = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=3000)
readiness = check_profile_readiness(runtime_profile, client)
if readiness.ready:
    sys.exit(0)

print("Profile is not ready yet:")
for issue in readiness.issues:
    print(f"  - {issue}")
sys.exit(1)
PY
}

initialize_profiles() {
  log "Initializing ontology and triplets databases for all current UI profiles"
  for profile in "${PROFILES[@]}"; do
    log "Checking profile readiness: ${profile}"
    if profile_ready "$profile"; then
      ok "Profile already initialized: ${profile}"
      continue
    fi

    log "Initializing missing or incomplete profile: ${profile}"
    .venv/bin/python init_dbs.py --profile "$profile" --resume
    ok "Profile initialized: ${profile}"
  done
}

main() {
  ensure_python
  ensure_env_file
  ensure_venv
  ensure_mongo_container
  wait_for_mongo
  initialize_profiles

  log "First setup complete"
  echo "Start the app with: ./scripts/start_wikontic.sh"
}

main "$@"
