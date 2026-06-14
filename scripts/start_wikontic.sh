#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MONGO_CONTAINER="${MONGO_CONTAINER:-wikontic}"
LEGACY_MONGO_CONTAINER="${LEGACY_MONGO_CONTAINER:-wikontic_mongo}"
MONGO_IMAGE="${MONGO_IMAGE:-mongodb/mongodb-atlas-local:latest}"
MONGO_HOST_PORT="${MONGO_HOST_PORT:-27018}"
MONGO_CONTAINER_PORT="${MONGO_CONTAINER_PORT:-27017}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

log() {
  printf "\n==> %s\n" "$1"
}

ok() {
  printf "OK: %s\n" "$1"
}

warn() {
  printf "\nWARNING: %s\n" "$1" >&2
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

  log "Mongo container not found; creating '$MONGO_CONTAINER'"
  docker pull "$MONGO_IMAGE"
  ok "Mongo image is available: $MONGO_IMAGE"
  docker run \
    --name "$MONGO_CONTAINER" \
    -d \
    -p "${MONGO_HOST_PORT}:${MONGO_CONTAINER_PORT}" \
    "$MONGO_IMAGE" >/dev/null
  ok "Mongo container created and started"
}

wait_for_mongo() {
  if [[ ! -x ".venv/bin/python" ]]; then
    echo ".venv is missing. Run ./scripts/first_setup.sh first." >&2
    exit 1
  fi

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

start_streamlit() {
  if [[ ! -x ".venv/bin/streamlit" ]]; then
    echo "Streamlit is missing from .venv. Run ./scripts/first_setup.sh first." >&2
    exit 1
  fi

  log "Starting Wikontic Streamlit app"
  echo "Open: http://localhost:${STREAMLIT_PORT}"
  ok "Wikontic is starting"
  exec .venv/bin/streamlit run Wikontic.py --server.port "$STREAMLIT_PORT"
}

main() {
  ensure_mongo_container
  wait_for_mongo
  start_streamlit
}

main "$@"
