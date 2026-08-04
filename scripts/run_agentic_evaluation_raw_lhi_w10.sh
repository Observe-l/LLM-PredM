#!/usr/bin/env bash
set -euo pipefail

# Periodic evaluator experiment using leakage-free, unrolled LHI and W=10.
# No policy cooldown, minimum exposure, counterfactual evaluation, or canary is used.
project_root="/home/lwh/Documents/Code/LLM-PredM"
python_bin="/home/lwh/anaconda3/envs/default/bin/python"
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

    echo "[$(date --iso-8601=seconds)] Starting $fd with raw lhi_rmse and evaluator W=10"
    "$python_bin" -m src.RAG_cmapss.joint_simulation \
        --fds "$fd" \
        --lhi_dir "$lhi_dir" \
        --output_dir "outputs/CMAPSS/RAG/agentic_evaluation_w10/$fd" \
        --score_col lhi_rmse \
        --raw_score_col d_rmse \
        --lhi_col lhi_rmse \
        --evaluation_window 10 \
        --risk_policy_mode llm_only \
        --model qwen3.5:9b \
        --timeout 600 \
        --ollama_num_predict 512 \
        --save_recent_ollama_outputs 20
    echo "[$(date --iso-8601=seconds)] Completed $fd"
done

echo "[$(date --iso-8601=seconds)] Completed evaluator rerun: ${fds[*]}"
