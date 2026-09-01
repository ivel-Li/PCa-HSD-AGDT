#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-/data/users/lly/.conda/envs/picai/bin/python}"
config_name="config_vitlarge_attention_nomask"
output_root="$project_root/run/VITLargeAttentionClassifier/nomask/42"

mkdir -p "$output_root/logs"
cd "$project_root"

run_single_fold() {
    local fold="$1"
    local gpu="$2"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u main.py \
        --config "$config_name" --gpu cuda:0 --fold "$fold" \
        >"$output_root/logs/fold_${fold}.log" 2>&1
}

# Four independent folds saturate the four GPUs without model synchronization.
pids=()
for fold in 0 1 2 3; do
    run_single_fold "$fold" "$fold" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

# The remaining fold uses all four GPUs to shorten the second wave.
CUDA_VISIBLE_DEVICES=0,1,2,3 "$python_bin" -u main.py \
    --config "$config_name" --gpu cuda:0 --fold 4 --data-parallel \
    >"$output_root/logs/fold_4.log" 2>&1
