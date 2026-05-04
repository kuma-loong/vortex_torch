#!/usr/bin/env bash
# Print space-separated indices of GPUs that are currently FREE.
#
# A GPU is considered FREE iff:
#   1. No compute process is running on it (nvidia-smi --query-compute-apps),
#      AND
#   2. memory.used is below the threshold (default 1024 MiB; override via $1).
#
# Exit codes:
#   0 — at least one free GPU; indices printed on stdout (e.g. "0 3 5 7")
#   1 — no free GPUs; nothing on stdout
#   2 — nvidia-smi missing or unusable
#
# Usage:
#   algorithm_scientist/free_gpus.sh                 # threshold = 1024 MiB
#   algorithm_scientist/free_gpus.sh 4096            # custom threshold (MiB)
#
# Bash idioms:
#   FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || { echo "no free GPUs"; exit 1; }
#   N=${#FREE_GPUS[@]}
#   for i in $(seq 0 $((N - 1))); do
#       gpu="${FREE_GPUS[$i]}"
#       CUDA_VISIBLE_DEVICES=$gpu python ... &
#   done
#   wait
#
set -euo pipefail

threshold="${1:-1024}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "free_gpus.sh: nvidia-smi not found in PATH" >&2
    exit 2
fi

# UUIDs of GPUs currently hosting any compute process. May be empty.
busy_uuids=$(nvidia-smi --query-compute-apps=gpu_uuid \
    --format=csv,noheader 2>/dev/null | sort -u || true)

free_indices=()
while IFS=, read -r idx uuid mem; do
    idx=$(echo "$idx" | xargs)
    uuid=$(echo "$uuid" | xargs)
    mem=$(echo "$mem" | xargs)
    [ -n "$idx" ] || continue
    # Skip if a compute process is on this GPU, or if memory.used >= threshold.
    if [ -n "$busy_uuids" ] && grep -qxF "$uuid" <<<"$busy_uuids"; then
        continue
    fi
    if [ "$mem" -ge "$threshold" ]; then
        continue
    fi
    free_indices+=("$idx")
done < <(nvidia-smi --query-gpu=index,uuid,memory.used \
    --format=csv,noheader,nounits)

if [ "${#free_indices[@]}" -eq 0 ]; then
    exit 1
fi

echo "${free_indices[*]}"
