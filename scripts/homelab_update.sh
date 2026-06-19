#!/usr/bin/env bash
# Manual homelab update after merging to main (Issue #11).
# Run from repo root on the LXC: ./scripts/homelab_update.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Stopping stack..."
docker compose down

echo "==> Pulling latest main..."
git pull origin main

echo "==> Rebuilding images..."
docker compose build api

echo "==> Starting stack (MLflow first, then API)..."
exec "$(dirname "$0")/homelab_start.sh"
