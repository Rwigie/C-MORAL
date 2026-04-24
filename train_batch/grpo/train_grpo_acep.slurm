#!/bin/bash
#SBATCH -A bfuy-delta-gpu
#SBATCH -p gpuH200x8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --job-name=acep_grpo_train
#SBATCH --mem=100G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

mkdir -p logs



# 固定工作目录：项目根目录（Molo_pipeline 的父目录）
cd /work/nvme/bfuy/rgao7
export PYTHONPATH=/work/nvme/bfuy/rgao7:${PYTHONPATH:-}

# ========================================
# 1) 启动 ADMET 服务 (后台)
# ========================================
source /u/rgao7/miniconda3/etc/profile.d/conda.sh
conda activate molo

python -m Molo.train_grpo --config Molo/configs/train_config_mistral_acep.yaml

kill $ADMET_PID