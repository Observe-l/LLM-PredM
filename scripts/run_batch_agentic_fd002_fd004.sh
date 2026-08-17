#!/usr/bin/env bash
set -euo pipefail

project_root="/home/lwh/Documents/Code/LLM-PredM"
python_bin="/home/lwh/anaconda3/envs/default/bin/python"
output_root="outputs/CMAPSS/RAG/agentic_evaluation_batch_w10"
cd "$project_root"

for fd in FD002 FD003 FD004; do
    if [[ "$fd" == "FD003" ]]; then
        lhi_dir="outputs/CMAPSS/cluster_20/lhi_fix"
    else
        lhi_dir="outputs/CMAPSS/history_condition_h20_fd002_fd004/lhi"
    fi
    echo "[$(date --iso-8601=seconds)] Starting batch experiment $fd"
    "$python_bin" -m src.RAG_cmapss.joint_simulation \
        --fds "$fd" \
        --lhi_dir "$lhi_dir" \
        --output_dir "$output_root/$fd" \
        --score_col lhi_rmse \
        --raw_score_col d_rmse \
        --lhi_col lhi_rmse \
        --lhi_trigger 0.25 \
        --evaluation_window 10 \
        --batch_size 10 \
        --processing_mode batch \
        --risk_policy_mode llm_only \
        --prompt_variant kg \
        --model qwen3.5:9b \
        --timeout 600 \
        --ollama_num_predict 512 \
        --save_recent_ollama_outputs 20
    echo "[$(date --iso-8601=seconds)] Completed batch experiment $fd"
done

echo "[$(date --iso-8601=seconds)] Completed FD002-FD004 batch experiments"
