#!/usr/bin/env bash
set -euo pipefail

project_root="/home/lwh/Documents/Code/LLM-PredM"
python_bin="/home/lwh/anaconda3/envs/default/bin/python"
output_root="outputs/CMAPSS/RAG/ablation/lhi_threshold_fixed"
cd "$project_root"

thresholds=(0.25 0.5 0.75 1.0)
fds=(FD001 FD002 FD003 FD004)

for threshold in "${thresholds[@]}"; do
    threshold_tag="$(printf '%s' "$threshold" | tr '.' 'p')"
    for fd in "${fds[@]}"; do
        if [[ "$fd" == "FD001" || "$fd" == "FD003" ]]; then
            lhi_dir="outputs/CMAPSS/cluster_20/lhi_fix"
        else
            lhi_dir="outputs/CMAPSS/history_condition_h20_fd002_fd004/lhi"
        fi

        echo "[$(date --iso-8601=seconds)] Starting fixed-LHI-threshold=$threshold $fd"
        "$python_bin" -m src.RAG_cmapss.joint_simulation \
            --fds "$fd" \
            --lhi_dir "$lhi_dir" \
            --output_dir "$output_root/threshold_${threshold_tag}/$fd" \
            --score_col lhi_rmse \
            --raw_score_col d_rmse \
            --lhi_col lhi_rmse \
            --lhi_trigger "$threshold" \
            --evaluation_window 10 \
            --risk_policy_mode llm_only \
            --prompt_variant kg \
            --disable_per_feedback_policy_update \
            --disable_update_tool \
            --model qwen3.5:9b \
            --timeout 600 \
            --ollama_num_predict 512 \
            --save_recent_ollama_outputs 20
        echo "[$(date --iso-8601=seconds)] Completed fixed-LHI-threshold=$threshold $fd"
    done
done

echo "[$(date --iso-8601=seconds)] Completed LHI-threshold ablation"
