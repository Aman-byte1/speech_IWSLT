#!/usr/bin/env bash
# ============================================================
# training/run.sh
# Fine-tunes the Whisper model for African speech translation.
#
# Usage:
#   bash run.sh                        # run finetune.py
#   bash run.sh --lang amh             # language-specific run (if supported)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# ── Virtual environment ───────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment ..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── Install dependencies ──────────────────────────────────────
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
if [ -f "$REQUIREMENTS" ]; then
    info "Installing dependencies ..."
    pip install --quiet --upgrade pip
    pip install --quiet -r "$REQUIREMENTS"
else
    warn "requirements.txt not found — skipping install."
fi

# ── HF token check ────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    warn "HF_TOKEN not set. Set it with: export HF_TOKEN=hf_xxxxxxxxxx"
fi

# ── Run fine-tuning ───────────────────────────────────────────
info "Starting fine-tuning ..."
python3 "$SCRIPT_DIR/finetune.py" "$@"
info "Fine-tuning complete."
