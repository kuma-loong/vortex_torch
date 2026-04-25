#!/usr/bin/env bash
#SBATCH --job-name=vortex_submission_batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --partition=medium
#SBATCH --exclude=compute-node-40
#SBATCH --output=/home/xun/vortex_torch/logs/submission/%x_%j.out
#SBATCH --error=/home/xun/vortex_torch/logs/submission/%x_%j.err

# -----------------------------------------------------------------------------
# Batched AIME24 runner — up to 8 submissions on one 8-GPU node.
#
# Each positional arg is a path to a submission JSON. For each one we launch
#   python examples/run_submission_aime24.py --config <cfg>
# in the background with a distinct CUDA_VISIBLE_DEVICES pinning.
#
# Usage:
#   sbatch algorithm_scientist/run_submission_batch.slurm \
#       submissions/v1.json submissions/v2.json ... submissions/v8.json
# -----------------------------------------------------------------------------

# Intentionally NOT using `set -e` so one failing submission doesn't kill the
# waits on the others. We handle exit codes manually.
set -uo pipefail
set -x

if [ $# -lt 1 ]; then
    echo "usage: sbatch run_submission_batch.slurm <cfg1.json> [cfg2.json ...]" >&2
    exit 2
fi
if [ $# -gt 8 ]; then
    echo "error: received $# configs but only 8 GPUs are available per node" >&2
    exit 2
fi

CONFIGS=("$@")

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
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
echo "Parent CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CONFIGS=${CONFIGS[*]}"

nvidia-smi

# Per-child logs under a single run directory so you can diff them easily.
LOG_DIR="/home/xun/vortex_torch/logs/submission/batch_${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_DIR}"

declare -a PIDS
declare -a CHILD_GPUS
declare -a CHILD_STEMS

# Fan out: one background process per config, each pinned to its own GPU.
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    if [ ! -f "${cfg}" ]; then
        echo "[batch] ERROR: config not found: ${cfg}" >&2
        exit 2
    fi
    stem="$(basename "${cfg}" .json)"
    log_out="${LOG_DIR}/gpu${i}_${stem}.out"
    log_err="${LOG_DIR}/gpu${i}_${stem}.err"

    echo "[batch] GPU ${i}  <-  ${cfg}   logs: ${log_out}"
    (
        export CUDA_VISIBLE_DEVICES="${i}"
        python algorithm_scientist/run_submission_aime24.py --config "${cfg}" \
            > "${log_out}" 2> "${log_err}"
    ) &
    PIDS+=("$!")
    CHILD_GPUS+=("${i}")
    CHILD_STEMS+=("${stem}")
done

# Collect every child. Propagate a non-zero exit if any child failed, but
# never short-circuit the other waits.
overall_rc=0
for j in "${!PIDS[@]}"; do
    wait "${PIDS[$j]}"
    child_rc=$?
    gpu="${CHILD_GPUS[$j]}"
    stem="${CHILD_STEMS[$j]}"
    echo "[batch] DONE GPU ${gpu} (${stem}) exit=${child_rc}"
    if [ "${child_rc}" -ne 0 ]; then
        overall_rc=1
    fi
done

echo "[batch] all ${#PIDS[@]} children finished; overall_rc=${overall_rc}"
exit "${overall_rc}"
