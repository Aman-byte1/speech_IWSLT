#!/usr/bin/env bash
# ============================================================
# data_collection/run.sh
# Runs the African Speech data pipeline.
#
# Usage:
#   bash run.sh           # process ALL languages
#   bash run.sh amh       # process a single language
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/config"
DATA_SCRIPT="$SCRIPT_DIR/data.py"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
VENV_DIR="$SCRIPT_DIR/.venv"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

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
    warn "requirements.txt not found at $REQUIREMENTS — skipping install."
fi

# ── HF token check ────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    warn "HF_TOKEN is not set. Gated datasets will be skipped."
    warn "Export it with: export HF_TOKEN=hf_xxxxxxxxxx"
fi

# ── Languages ─────────────────────────────────────────────────
LANG_ARG="${1:-}"

if [ -n "$LANG_ARG" ]; then
    LANGUAGES=("$LANG_ARG")
else
    LANGUAGES=()
    for yaml_file in "$CONFIG_DIR"/*.yaml; do
        lang=$(basename "$yaml_file" .yaml)
        LANGUAGES+=("$lang")
    done
fi

info "Languages to process: ${LANGUAGES[*]}"

# ── Run ───────────────────────────────────────────────────────
SKIP_PROC_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--skip-processing" ]; then
        SKIP_PROC_FLAG="--skip_processing"
        break
    fi
done

FAILED=()

for LANG in "${LANGUAGES[@]}"; do
    CONFIG_FILE="$CONFIG_DIR/${LANG}.yaml"

    if [ ! -f "$CONFIG_FILE" ]; then
        warn "Config not found for '$LANG' — skipping."
        continue
    fi

    echo ""
    info "==========================================="
    info "  Language: $LANG"
    info "==========================================="

    python3 "$DATA_SCRIPT" \
        --config "$CONFIG_FILE" \
        --checkpoint_dir "$CHECKPOINT_DIR/$LANG" \
        $SKIP_PROC_FLAG \
        && info "✓ $LANG done" \
        || { warn "✗ $LANG failed."; FAILED+=("$LANG"); }
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    info "All languages completed successfully."
else
    warn "Failed: ${FAILED[*]}"
    warn "Retry with: bash run.sh <lang>"
    exit 1
fi
