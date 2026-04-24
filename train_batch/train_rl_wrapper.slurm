#!/bin/bash
#SBATCH -A bfuy-delta-gpu
#SBATCH -p gpuH200x8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --job-name=rl_train
#SBATCH --mem=100G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs
cd /work/nvme/bfuy/rgao7/Molo

# Required env vars:
#   ALGO=grpo|gdpo
#   TASK=<task_name>
# Optional:
#   EXTRA_ARGS="..."
#   DRY_RUN=1

ALGO="${ALGO:-}"
TASK="${TASK:-}"
EXP="${EXP:-}"
CONFIG="${CONFIG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
MODEL="${MODEL:-}"
JOB_NAME="${JOB_NAME:-}"

to_lower() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

infer_task_name() {
  local t="${TASK:-${EXP:-}}"
  if [[ -n "$t" ]]; then
    echo "$t"
    return
  fi
  if [[ -n "${CONFIG:-}" ]]; then
    local base
    base="$(basename "$CONFIG")"
    # train_config_<model>_<task>.yaml or train_config_<task>.yaml
    local inferred
    inferred="$(echo "$base" | sed -E 's/^train_config_([a-zA-Z0-9]+_)?([a-zA-Z0-9]+)\.yaml$/\2/')"
    if [[ -n "$inferred" && "$inferred" != "$base" ]]; then
      echo "$inferred"
      return
    fi
  fi
  echo "task"
}

infer_model_name() {
  if [[ -n "${MODEL:-}" ]]; then
    to_lower "$MODEL"
    return
  fi

  # For --task/--exp paths, train_rl.py maps GRPO to mistral configs by default.
  if [[ -z "${CONFIG:-}" ]]; then
    echo "mistral"
    return
  fi

  local cfg_path="$CONFIG"
  if [[ ! -f "$cfg_path" ]]; then
    cfg_path="/work/nvme/bfuy/rgao7/Molo/$CONFIG"
  fi
  if [[ -f "$cfg_path" ]]; then
    local detected
    detected="$(
      awk -F: '
        /^[[:space:]]*base_model[[:space:]]*:/ {
          gsub(/[ "]/, "", $2);
          print tolower($2);
          exit
        }
      ' "$cfg_path"
    )"
    if [[ -n "$detected" ]]; then
      echo "$detected"
      return
    fi
  fi

  echo "model"
}

sanitize_name() {
  echo "$1" | tr '+/' '--' | tr -cs '[:alnum:]_-' '-'
}

resolve_model_config_if_needed() {
  if [[ -n "${CONFIG:-}" ]]; then
    return
  fi
  if [[ -z "${MODEL:-}" ]]; then
    return
  fi

  local model_lc
  model_lc="$(to_lower "$MODEL")"
  local task_name="${TASK:-${EXP:-}}"
  if [[ -z "$task_name" ]]; then
    return
  fi

  local candidate=""
  if [[ "$ALGO" == "grpo" ]]; then
    candidate="configs/train_config_${model_lc}_${task_name}.yaml"
  else
    candidate="configs/gdpo/train_config_${model_lc}_${task_name}.yaml"
  fi

  if [[ -f "/work/nvme/bfuy/rgao7/Molo/$candidate" ]]; then
    CONFIG="$candidate"
    echo "[INFO] Auto-selected config from MODEL: $CONFIG"
    return
  fi

  echo "[ERROR] MODEL=$MODEL was provided but config not found: $candidate" >&2
  echo "[ERROR] Please provide CONFIG explicitly, or add that model-specific config file." >&2
  exit 1
}

if [[ -z "$ALGO" ]]; then
  echo "[ERROR] Please provide ALGO."
  echo "Example: sbatch --export=ALGO=gdpo,TASK=elq train_batch/train_rl_wrapper.slurm"
  exit 1
fi

resolve_model_config_if_needed

if [[ -z "$JOB_NAME" ]]; then
  TASK_NAME="$(infer_task_name)"
  MODEL_NAME="$(infer_model_name)"
  ALGO_NAME="$(to_lower "$ALGO")"
  JOB_NAME="$(sanitize_name "${TASK_NAME}_${MODEL_NAME}_${ALGO_NAME}")"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME" >/dev/null 2>&1 || true
fi
echo "[INFO] JobName: $JOB_NAME"

ARGS=(--algo "$ALGO")
if [[ -n "$TASK" ]]; then
  ARGS+=(--task "$TASK")
fi
if [[ -n "$EXP" ]]; then
  ARGS+=(--exp "$EXP")
fi
if [[ -n "$CONFIG" ]]; then
  ARGS+=(--config "$CONFIG")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  ARGS+=(--extra "$EXTRA_ARGS")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi

bash scripts/run_rl.sh "${ARGS[@]}"
