#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_rl.sh --algo <grpo|gdpo> [--task <name> | --exp <name> | --config <path>] [--extra "..."] [--dry-run]

Examples:
  bash scripts/run_rl.sh --algo grpo --task bpq
  bash scripts/run_rl.sh --algo gdpo --exp elq
  bash scripts/run_rl.sh --algo grpo --config configs/train_config_mistral_bpq.yaml --dry-run
EOF
}

print_run_plan() {
  local line="================================================================================================"
  echo
  echo "$line"
  echo "C-MORAL bash launch plan"
  echo "$line"
  printf "%-16s : %s\n" "Algorithm" "${ALGO^^}"
  printf "%-16s : %s\n" "Task" "${TASK:-auto/from-config}"
  printf "%-16s : %s\n" "Experiment" "${EXP:-none}"
  printf "%-16s : %s\n" "Config" "${CONFIG:-auto}"
  printf "%-16s : %s\n" "Extra args" "${EXTRA_ARGS:-none}"
  printf "%-16s : %s\n" "Conda env" "molo"
  printf "%-16s : %s\n" "Project dir" "$ROOT"
  printf "%-16s : %s\n" "Python module" "Molo.train_rl"
  printf "%-16s : python -m Molo.train_rl %s\n" "Command" "${ARGS[*]}"
  echo "$line"
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

print_run_plan

source /u/rgao7/miniconda3/etc/profile.d/conda.sh
conda activate molo
export PYTHONPATH="$ROOT_PARENT:${PYTHONPATH:-}"
python -m Molo.train_rl "${ARGS[@]}"
