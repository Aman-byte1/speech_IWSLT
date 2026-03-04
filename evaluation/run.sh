#!/usr/bin/env bash
# ============================================================
# evaluation/run.sh
# Placeholder — add your evaluation scripts here and update
# this file to run them.
#
# Usage:
#   bash run.sh
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

# ── Run evaluation ────────────────────────────────────────────
info "Add your evaluation script calls below."
# python3 "$SCRIPT_DIR/your_eval_script.py" "$@"
