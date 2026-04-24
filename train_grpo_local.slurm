#!/bin/bash
#SBATCH -A bfuy-delta-gpu
#SBATCH -p gpuH200x8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --job-name=molo_train
#SBATCH --mem=100G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

mkdir -p logs

cd /work/nvme/bfuy/rgao7
export PYTHONPATH=/work/nvme/bfuy/rgao7:${PYTHONPATH:-}

source /u/rgao7/miniconda3/etc/profile.d/conda.sh
conda activate molo
python -m Molo.train_grpo 

kill $ADMET_PID