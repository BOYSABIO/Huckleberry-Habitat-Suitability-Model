#!/usr/bin/env bash
# Start API + MLflow on homelab LXC (Issue #11).
# MLflow pip install can take 15+ minutes on first boot; start API only after :5000 is up.
#
# Usage (from repo root on the LXC):
#   chmod +x scripts/homelab_start.sh
#   ./scripts/homelab_start.sh

set -euo pipefail

cd "$(dirname "$0")/.."

MLFLOW_URL="http://127.0.0.1:5000/"
API_HEALTH="http://localhost:8000/health"
MAX_WAIT_MLFLOW="${MAX_WAIT_MLFLOW:-1200}"  # seconds (20 min)
POLL="${POLL:-10}"

wait_for_url() {
  local url="$1"
  local label="$2"
  local max="$3"
  local elapsed=0
  echo "==> Waiting for ${label} (${url})..."
  while [ "$elapsed" -lt "$max" ]; do
    code=$(curl -s -o /dev/null -w "%{http_code}" "$url" || true)
    if [ "$code" = "200" ]; then
      echo "==> ${label} is up."
      return 0
    fi
    sleep "$POLL"
    elapsed=$((elapsed + POLL))
  done
  echo "ERROR: ${label} did not respond within ${max}s" >&2
  return 1
}

echo "==> Starting MLflow..."
docker compose up -d mlflow

wait_for_url "$MLFLOW_URL" "MLflow" "$MAX_WAIT_MLFLOW"

echo "==> Starting API..."
docker compose up -d --no-deps api

wait_for_url "$API_HEALTH" "API /health" 120

echo "==> Health:"
curl -sf "$API_HEALTH" | python3 -m json.tool
echo "Done."
