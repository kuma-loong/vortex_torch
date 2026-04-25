#!/usr/bin/env bash
#SBATCH --job-name=claude_submission
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=medium
#SBATCH --exclude=compute-node-40
#SBATCH --output=/home/xun/vortex_torch/logs/submission/%x_%j.out
#SBATCH --error=/home/xun/vortex_torch/logs/submission/%x_%j.err

set -euo pipefail
set -x

# The AIME24 submission script is locked to a fixed protocol
# (trials=16, model=Qwen/Qwen3-4B, tp_size=1, data=aime24.jsonl, etc.).
# The ONLY argument is the path to the submission's engine JSON.
CONFIG="${1:?usage: sbatch run_submission.slurm <submissions/<name>.json>}"

cd /home/xun/vortex_torch

export PYTHONUNBUFFERED=1
export SLURM_OUTPUT_MODE=standard
export TORCH_NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=0
export TORCH_FR_BUFFER_SIZE=32

source /home/xun/anaconda3/etc/profile.d/conda.sh
conda activate vortex_o

which python
python --version

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CONFIG=${CONFIG}"

nvidia-smi

python algorithm_scientist/run_submission_aime24.py --config "${CONFIG}"
