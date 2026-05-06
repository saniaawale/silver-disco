#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/aegean-ai/foreign-whispers.git"
REPO_DIR="$(dirname "$0")/../foreign-whispers"

# ── 1. Clone the repo if not already present ───────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning foreign-whispers..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "foreign-whispers already cloned at $REPO_DIR"
fi

cd "$REPO_DIR"

# ── 2. Create .env if missing ─────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp "$(dirname "$0")/.env.example" .env
    echo ""
    echo "Created .env from .env.example"
    echo "  Edit .env and set FW_HF_TOKEN before running the diarization notebook."
    echo ""
fi

# ── 3. Create cookies.txt if missing (required by docker-compose volume mount) ─
 if [ ! -f "cookies.txt" ]; then
    touch cookies.txt
    echo "Created empty cookies.txt"
fi

# ── 4. Detect GPU and choose compose profile ──────────────────────────────────
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    PROFILE="nvidia"
else
    PROFILE="cpu"
    echo ""
    echo "No NVIDIA GPU detected — using CPU profile."
    echo "  Whisper STT and Chatterbox TTS GPU containers will NOT start."
    echo "  The API and frontend will still run."
    echo ""
fi

# ── 5. Build and start the stack ───────────────────────────────────────────
echo "Starting Docker stack (profile: $PROFILE)..."
docker compose --profile "$PROFILE" up -d --build

echo ""
echo "Stack is up. Verify:"
echo "  API:      curl http://localhost:8080/healthz"
echo "  Frontend: open http://localhost:8501 in your browser"
