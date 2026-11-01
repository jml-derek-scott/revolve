#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD=1
OPEN_BROWSER=0
FORCE_SETUP=0
BUILD_DIAGRAMS=0

usage() {
  cat <<EOF
Usage: ./run.sh [options]

Options:
  -p, --port <port>   Port to bind (default: 8000, or \$PORT)
  --host <host>       Host to bind (default: 127.0.0.1, or \$HOST)
  --no-reload         Disable uvicorn auto-reload
  --open              Open http://localhost:<port> in default browser
  --setup             Force dependency & vendor setup check
  --diagrams          Regenerate D2 diagram SVGs in _docs/diagrams/svg/
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --no-reload)
      RELOAD=0
      shift
      ;;
    --open)
      OPEN_BROWSER=1
      shift
      ;;
    --setup)
      FORCE_SETUP=1
      shift
      ;;
    --diagrams)
      BUILD_DIAGRAMS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

export PYTORCH_ENABLE_MPS_FALLBACK=1

if ! command -v poetry >/dev/null; then
  echo "Error: poetry not found. Install with: brew install poetry  (or: pipx install poetry)" >&2
  exit 1
fi

# 1. Poetry env & dependencies
if [[ ! -x ".venv/bin/python" || "$FORCE_SETUP" -eq 1 ]]; then
  echo "==> Installing Python dependencies with Poetry..."
  make install
fi

# 2. Vendor check (TripoSR for /api/generate)
if [[ "$FORCE_SETUP" -eq 1 && ! -d "vendor/TripoSR" ]]; then
  echo "==> Cloning vendor/TripoSR..."
  mkdir -p vendor
  git clone https://github.com/VAST-AI-Research/TripoSR vendor/TripoSR
fi

# 3. Optional D2 diagram build
if [[ "$BUILD_DIAGRAMS" -eq 1 ]]; then
  if command -v d2 &>/dev/null; then
    echo "==> Rendering D2 diagrams..."
    mkdir -p _docs/diagrams/svg
    for f in overview texture-wrap full-build hybrid api-triposr; do
      if [[ -f "_docs/diagrams/${f}.d2" ]]; then
        d2 "_docs/diagrams/${f}.d2" "_docs/diagrams/svg/${f}.svg"
      fi
    done
  else
    echo "Warning: d2 CLI not found; skipping diagram generation." >&2
  fi
fi

# 4. Sanity checks for runtime directories & demo assets
mkdir -p uploads outputs
if ! compgen -G "frontend/assets/models/*.glb" >/dev/null; then
  echo "Warning: No .glb files found in frontend/assets/models/ — the demo viewer will have no pre-built models to load." >&2
fi

URL="http://${HOST}:${PORT}"
echo "==> Starting 3DViewer at ${URL}"

if [[ "$OPEN_BROWSER" -eq 1 ]] && command -v open &>/dev/null; then
  (sleep 1 && open "$URL") &
fi

UVICORN_ARGS=(backend.app.main:app --host "$HOST" --port "$PORT")
if [[ "$RELOAD" -eq 1 ]]; then
  UVICORN_ARGS+=(--reload)
fi

exec poetry run uvicorn "${UVICORN_ARGS[@]}"
