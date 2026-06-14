#!/usr/bin/env bash
set -euo pipefail

echo "setup_db.sh is legacy. Prefer: ./scripts/first_setup.sh"
echo "Starting only MongoDB Atlas Local for compatibility..."

docker pull mongodb/mongodb-atlas-local:latest
docker run --name wikontic -d -p 27018:27017 mongodb/mongodb-atlas-local:latest
