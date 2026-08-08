#!/usr/bin/env bash
set -euo pipefail

project_root="/home/lwh/Documents/Code/LLM-PredM"
python_bin="/home/lwh/anaconda3/envs/default/bin/python"
output_root="outputs/CMAPSS/RAG/agentic_evaluation_w10_theta_init"
fixed_script="run_lhi_threshold_ablation_050_plus.sh"

cd "$project_root"

echo "[$(date --iso-8601=seconds)] Waiting for fixed-threshold experiment to finish..."
while pgrep -af "$fixed_script" | grep -v "pgrep -af" >/dev/null; do
    sleep 30
done
echo "[$(date --iso-8601=seconds)] Fixed-threshold experiment is no longer running."

run_fd() {
    local theta="$1"
    local fd="$2"
    local theta_tag
    theta_tag="$(printf '%s' "$theta" | tr '.' 'p')"
    local lhi_dir
    if [[ "$fd" == "FD001" || "$fd" == "FD003" ]]; then
        lhi_dir="outputs/CMAPSS/cluster_20/lhi_fix"
    else
        lhi_dir="outputs/CMAPSS/history_condition_h20_fd002_fd004/lhi"
    fi

    echo "[$(date --iso-8601=seconds)] Starting adaptive initial theta=$theta $fd"
    "$python_bin" -m src.RAG_cmapss.joint_simulation \
        --fds "$fd" \
        --lhi_dir "$lhi_dir" \
        --output_dir "$output_root/theta_${theta_tag}/$fd" \
        --score_col lhi_rmse \
        --raw_score_col d_rmse \
        --lhi_col lhi_rmse \
        --lhi_trigger "$theta" \
        --evaluation_window 10 \
        --risk_policy_mode llm_only \
        --prompt_variant kg \
        --model qwen3.5:9b \
        --timeout 600 \
        --ollama_num_predict 512 \
        --save_recent_ollama_outputs 20
    echo "[$(date --iso-8601=seconds)] Completed adaptive initial theta=$theta $fd"
}

for theta in 0.50 0.75 1.0 1.25; do
    for fd in FD001 FD002 FD003 FD004; do
        run_fd "$theta" "$fd"
    done
done

echo "[$(date --iso-8601=seconds)] Completed adaptive initial-theta experiments."
