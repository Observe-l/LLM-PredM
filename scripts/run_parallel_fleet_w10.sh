#!/usr/bin/env bash
set -euo pipefail

# Shared-clock fleet simulation. Each FD is an independent experiment; all
# engines within that FD start at the same global cycle and share one adaptive
# policy/reflection memory. FD experiments are deliberately run serially.
project_root="/home/lwh/Documents/Code/LLM-PredM"
python_bin="/home/lwh/anaconda3/envs/default/bin/python"
output_root="outputs/CMAPSS/RAG/parallel_fleet_w10"
cd "$project_root"

fds=("$@")
if [[ ${#fds[@]} -eq 0 ]]; then
    fds=(FD001 FD002 FD003 FD004)
fi

for fd in "${fds[@]}"; do
    if [[ "$fd" == "FD001" || "$fd" == "FD003" ]]; then
        lhi_dir="outputs/CMAPSS/cluster_20/lhi_fix"
    else
        lhi_dir="outputs/CMAPSS/history_condition_h20_fd002_fd004/lhi"
    fi
    echo "[$(date --iso-8601=seconds)] Starting parallel fleet experiment $fd"
    "$python_bin" -m src.RAG_cmapss.parallel_fleet_simulation \
        --fds "$fd" \
        --lhi_dir "$lhi_dir" \
        --output_dir "$output_root/$fd" \
        --score_col lhi_rmse \
        --raw_score_col d_rmse \
        --lhi_col lhi_rmse \
        --lhi_trigger 0.25 \
        --health_reference_cycles 50 \
        --prompt_variant kg \
        --model qwen3.5:9b \
        --timeout 600 \
        --ollama_num_predict 512 \
        --save_recent_ollama_outputs 20
    echo "[$(date --iso-8601=seconds)] Completed parallel fleet experiment $fd"
done

echo "[$(date --iso-8601=seconds)] Completed parallel fleet experiments: ${fds[*]}"
