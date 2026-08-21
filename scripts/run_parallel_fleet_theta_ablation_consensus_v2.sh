#!/usr/bin/env bash
set -euo pipefail

# Same configuration as parallel_fleet_mixed_theta_0p25_consensus_v2,
# varying only the initial LHI trigger (theta).
project_root="/home/lwh/Documents/Code/LLM-PredM"
python_bin="/home/lwh/.conda/envs/llm/bin/python"
cd "$project_root"

for theta in 0.5 0.75 1.0 1.25; do
    theta_tag="$(printf '%s' "$theta" | tr '.' 'p')"
    output_root="outputs/CMAPSS/RAG/parallel_fleet_mixed_theta_${theta_tag}_consensus_v2"
    echo "[$(date --iso-8601=seconds)] Starting parallel fleet theta=$theta"
    "$python_bin" -m src.RAG_cmapss.parallel_fleet_simulation \
        --lhi_dirs outputs/CMAPSS/cluster_20/lhi_fix \
                  outputs/CMAPSS/history_condition_h20_fd002_fd004/lhi \
        --mixed_fleet \
        --fds FD001 FD002 FD003 FD004 \
        --output_dir "$output_root" \
        --score_col lhi_rmse \
        --raw_score_col d_rmse \
        --lhi_col lhi_rmse \
        --lhi_trigger "$theta" \
        --health_reference_cycles 50 \
        --prompt_variant kg \
        --decision_mode forecast_window \
        --model qwen3.5:9b \
        --timeout 600 \
        --ollama_num_predict 512 \
        --save_recent_ollama_outputs 20
    echo "[$(date --iso-8601=seconds)] Completed parallel fleet theta=$theta"
done

echo "[$(date --iso-8601=seconds)] Completed theta ablation: 0.5 0.75 1.0 1.25"
