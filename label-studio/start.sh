#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a
source "$ROOT/.env"
set +a
exec "$ROOT/.venv/bin/label-studio" start --host 0.0.0.0 --port 8080 "$@"