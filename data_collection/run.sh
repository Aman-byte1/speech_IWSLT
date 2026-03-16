#!/usr/bin/env bash
# ============================================================
# data_collection/run.sh
# Runs the unified African Speech data pipeline.
#
# Handles BOTH:
#   - HuggingFace datasets  (source: hf)
#   - Mozilla Common Voice 24.0  (source: mozilla_cv)
#
# Usage:
#   bash run.sh           # process ALL languages
#   bash run.sh amh       # process a single language
#   bash run.sh --status  # show processing report
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/config"
PIPELINE_SCRIPT="$SCRIPT_DIR/pipeline.py"
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
    python3 -m pip install --quiet -r "$REQUIREMENTS"
else
    warn "requirements.txt not found at $REQUIREMENTS — skipping install."
fi

# ── Token checks ─────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    warn "HF_TOKEN is not set. Gated datasets will be skipped."
    warn "Export it with: export HF_TOKEN=hf_xxxxxxxxxx"
fi

if [ -z "${MOZ_API_KEY:-}" ]; then
    warn "MOZ_API_KEY is not set. Mozilla Common Voice datasets will be skipped."
    warn "Export it with: export MOZ_API_KEY=your_key_here"
    warn "(You can also set moz_api_key in the YAML config files)"
fi

# ── Parse arguments ──────────────────────────────────────────
LANG_ARG=""
EXTRA_FLAGS=""

for arg in "$@"; do
    case "$arg" in
        --status)
            EXTRA_FLAGS="$EXTRA_FLAGS --status"
            ;;
        --merge-only)
            EXTRA_FLAGS="$EXTRA_FLAGS --merge-only"
            ;;
        *)
            LANG_ARG="$arg"
            ;;
    esac
done

# ── Languages ─────────────────────────────────────────────────
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

    python3 "$PIPELINE_SCRIPT" \
        --config "$CONFIG_FILE" \
        $EXTRA_FLAGS \
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
