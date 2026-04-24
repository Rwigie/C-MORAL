#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_rl.sh --algo <grpo|gdpo> [--task <name> | --exp <name> | --config <path>] [--extra "..."] [--dry-run]
EOF
}

ALGO=""
TASK=""
EXP=""
CONFIG=""
EXTRA_ARGS=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo) ALGO="${2:-}"; shift 2 ;;
    --task) TASK="${2:-}"; shift 2 ;;
    --exp) EXP="${2:-}"; shift 2 ;;
    --config) CONFIG="${2:-}"; shift 2 ;;
    --extra) EXTRA_ARGS="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$ALGO" ]]; then
  echo "[ERROR] --algo is required." >&2
  usage
  exit 1
fi

ROOT="/work/nvme/bfuy/rgao7/Molo"
ROOT_PARENT="/work/nvme/bfuy/rgao7"
cd "$ROOT"

ARGS=(--algo "$ALGO")
if [[ -n "$TASK" ]]; then ARGS+=(--task "$TASK"); fi
if [[ -n "$EXP" ]]; then ARGS+=(--exp "$EXP"); fi
if [[ -n "$CONFIG" ]]; then ARGS+=(--config "$CONFIG"); fi
if [[ "$DRY_RUN" -eq 1 ]]; then ARGS+=(--dry_run); fi
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SPLIT=($EXTRA_ARGS)
  ARGS+=("${EXTRA_SPLIT[@]}")
fi

source /u/rgao7/miniconda3/etc/profile.d/conda.sh
conda activate molo
export PYTHONPATH="$ROOT_PARENT:${PYTHONPATH:-}"
python -m Molo.train_rl "${ARGS[@]}"
