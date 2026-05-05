#!/usr/bin/env bash

set -uo pipefail

usage() {
    echo "Usage: $0 <input_json_dir> <timeout_minutes>" >&2
    echo "Example: $0 submissions/gpt_5 30" >&2
}

if [ "$#" -ne 2 ]; then
    usage
    exit 2
fi

input_dir="$1"
timeout_minutes="$2"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

if ! [[ "$timeout_minutes" =~ ^[1-9][0-9]*$ ]]; then
    echo "run_ruler.sh: timeout_minutes must be a positive integer, got: $timeout_minutes" >&2
    exit 2
fi

if [ ! -d "$input_dir" ] && [ -d "$repo_root/$input_dir" ]; then
    input_dir="$repo_root/$input_dir"
fi

if [ ! -d "$input_dir" ]; then
    echo "run_ruler.sh: input_json_dir is not a directory: $input_dir" >&2
    exit 2
fi

if ! command -v timeout >/dev/null 2>&1; then
    echo "run_ruler.sh: GNU timeout is required but was not found in PATH" >&2
    exit 2
fi

input_dir_abs="$(cd "$input_dir" && pwd)"

mapfile -t configs < <(find "$input_dir_abs" -type f -name "*.json" | sort)

if [ "${#configs[@]}" -eq 0 ]; then
    echo "run_ruler.sh: no JSON files found under $input_dir_abs" >&2
    exit 1
fi

cd "$repo_root" || exit 2

failures=0

echo "[run_ruler] input_dir=$input_dir_abs"
echo "[run_ruler] timeout=${timeout_minutes}m"
echo "[run_ruler] configs=${#configs[@]}"

for config in "${configs[@]}"; do
    echo
    echo "[run_ruler] start: $config"
    timeout --kill-after=1m "${timeout_minutes}m" \
        python algorithm_scientist/run_ruler.py --config "$config"
    status=$?

    if [ "$status" -eq 0 ]; then
        echo "[run_ruler] done: $config"
    elif [ "$status" -eq 124 ]; then
        echo "[run_ruler] timed out after ${timeout_minutes}m: $config" >&2
        failures=$((failures + 1))
    else
        echo "[run_ruler] failed with exit code $status: $config" >&2
        failures=$((failures + 1))
    fi
done

if [ "$failures" -ne 0 ]; then
    echo "[run_ruler] completed with $failures failure(s)" >&2
    exit 1
fi

echo "[run_ruler] completed successfully"
