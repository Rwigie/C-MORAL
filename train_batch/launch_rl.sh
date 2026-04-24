#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./train_batch/launch_rl.sh --algo <grpo|gdpo> --task <task_name> [--extra "..."] [--dry-run]

Examples:
  ./train_batch/launch_rl.sh --algo grpo --task elq
  ./train_batch/launch_rl.sh --algo gdpo --task bpq
  ./train_batch/launch_rl.sh --algo gdpo --task elq --dry-run
EOF
}

ALGO=""
TASK=""
EXTRA_ARGS=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo)
      ALGO="${2:-}"
      shift 2
      ;;
    --task)
      TASK="${2:-}"
      shift 2
      ;;
    --extra)
      EXTRA_ARGS="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$ALGO" || -z "$TASK" ]]; then
  echo "[ERROR] --algo and --task are required." >&2
  usage
  exit 1
fi

case "$ALGO" in
  grpo|gdpo) ;;
  *)
    echo "[ERROR] --algo must be grpo or gdpo, got: $ALGO" >&2
    exit 1
    ;;
esac

ROOT_DIR="/work/nvme/bfuy/rgao7"
PROJECT_DIR="$ROOT_DIR/Molo"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

if [[ "$ALGO" == "grpo" ]]; then
  TRAIN_MODULE="Molo.train_grpo"
  CONFIG_PATH="$PROJECT_DIR/configs/train_config_mistral_${TASK}.yaml"
  MODEL_NAME="Mistral"
else
  TRAIN_MODULE="Molo.train_gdpo"
  CONFIG_PATH="$PROJECT_DIR/configs/gdpo/train_config_mistral_${TASK}.yaml"
  MODEL_NAME="Mistral"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

CMD=(python -m "$TRAIN_MODULE" --config "$CONFIG_PATH")
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SPLIT=($EXTRA_ARGS)
  CMD+=("${EXTRA_SPLIT[@]}")
fi

LINE="================================================================================================"
echo
echo "$LINE"
echo "C-MORAL batch launch plan"
echo "$LINE"
printf "%-16s : %s\n" "Algorithm" "${ALGO^^}"
printf "%-16s : %s\n" "Model" "$MODEL_NAME"
printf "%-16s : %s\n" "Task alias" "$TASK"
printf "%-16s : %s\n" "Config" "$CONFIG_PATH"
printf "%-16s : %s\n" "Conda env" "molo"
printf "%-16s : %s\n" "Python module" "$TRAIN_MODULE"
printf "%-16s : %s\n" "Extra args" "${EXTRA_ARGS:-none}"
printf "%-16s : %s\n" "Command" "${CMD[*]}"
echo "$LINE"

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

source /u/rgao7/miniconda3/etc/profile.d/conda.sh
conda activate molo
cd "$ROOT_DIR"
"${CMD[@]}"
