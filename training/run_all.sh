#!/usr/bin/env bash
# ============================================================
# training/run_all.sh
# Fine-tunes MMS-1b-all on all collected African speech datasets.
#
# Usage:
#   bash run_all.sh                                # all languages, 3 epochs
#   bash run_all.sh --langs amh swh fra            # specific languages
#   bash run_all.sh --resume                       # skip already-done langs
#   bash run_all.sh --epochs 5 --batch_size 8      # custom hyperparams
#   bash run_all.sh --max_samples 5000             # quick experiment
#   bash run_all.sh --push_to_hub                  # push to HF after each
#   bash run_all.sh --dry_run                      # just print what would run
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERR]${NC}   $*"; }

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

# ── Run multilingual fine-tuning ──────────────────────────────
info "Starting multilingual MMS fine-tuning ..."
info "Arguments: $*"
python3 "$SCRIPT_DIR/finetune_mms_all.py" "$@"
info "Multilingual fine-tuning complete."
