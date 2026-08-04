#!/usr/bin/env bash
set -euo pipefail

project_root="/home/lwh/Documents/Code/LLM-PredM"
python_bin="/home/lwh/anaconda3/envs/default/bin/python"
experiment_root="outputs/CMAPSS/RAG/history_condition_h20_kg_strict_peak_timing"

cd "$project_root"

run_fd() {
    local fd_name="$1"
    local lhi_dir="$2"

    echo "[$(date --iso-8601=seconds)] Starting ${fd_name} with raw lhi_rmse"
    "$python_bin" -m src.RAG_cmapss.joint_simulation \
        --lhi_dir "$lhi_dir" \
        --output_dir "$experiment_root/$fd_name" \
        --fds "$fd_name" \
        --score_col lhi_rmse \
        --raw_score_col d_rmse \
        --lhi_col lhi_rmse \
        --risk_policy_mode llm_only \
        --disable_periodic_evaluation \
        --disable_per_feedback_policy_update \
        --model qwen3.5:9b \
        --timeout 600 \
        --ollama_num_predict 512 \
        --save_recent_ollama_outputs 20
    echo "[$(date --iso-8601=seconds)] Completed ${fd_name}"
}

run_fd FD001 outputs/CMAPSS/cluster_20/lhi_fix
run_fd FD002 outputs/CMAPSS/history_condition_h20_fd002_fd004/lhi
run_fd FD003 outputs/CMAPSS/cluster_20/lhi_fix
run_fd FD004 outputs/CMAPSS/history_condition_h20_fd002_fd004/lhi

echo "[$(date --iso-8601=seconds)] Completed FD001-FD004 raw-LHI baseline rerun"
