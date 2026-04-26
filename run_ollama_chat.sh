#!/usr/bin/env bash
#
# Ollama Chat Studio — macOS / Linux launcher
# Creates a virtual environment on first run and installs dependencies.
#

set -e
cd "$(dirname "$0")"

# ── First-run setup ──
if [ ! -d ".venv" ]; then
    echo ""
    echo "  ============================================"
    echo "   First-run setup — creating virtual environment..."
    echo "  ============================================"
    echo ""
    python3 -m venv .venv
    source .venv/bin/activate
    echo "  Installing dependencies..."
    pip install -r requirements.txt --quiet
    echo ""
    echo "  Setup complete!"
    echo ""
else
    source .venv/bin/activate
fi

# ── Check if Ollama is reachable ──
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo ""
    echo "  WARNING: Ollama does not appear to be running on localhost:11434"
    echo "  Start Ollama first:  ollama serve"
    echo "  Download Ollama:     https://ollama.com/download"
    echo ""
fi

# ── Launch the app ──
echo "  Starting Ollama Chat Studio..."
echo ""
python ollama_chat.py
