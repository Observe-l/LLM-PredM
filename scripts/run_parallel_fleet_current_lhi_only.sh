#!/usr/bin/env bash
set -euo pipefail

# Full shared-clock current-LHI-only ablation.
# The existing forecast-window experiment remains available through
# scripts/run_parallel_fleet_w10.sh and the default decision_mode.
project_root="/home/lwh/Documents/Code/LLM-PredM"
python_bin="/home/lwh/.conda/envs/llm/bin/python"
output_root="outputs/CMAPSS/RAG/parallel_fleet_mixed_theta_0p25_current_lhi_only_consensus_v2"
cd "$project_root"

"$python_bin" -m src.RAG_cmapss.parallel_fleet_simulation \
    --lhi_dirs outputs/CMAPSS/cluster_20/lhi_fix \
              outputs/CMAPSS/history_condition_h20_fd002_fd004/lhi \
    --mixed_fleet \
    --fds FD001 FD002 FD003 FD004 \
    --output_dir "$output_root" \
    --score_col lhi_rmse \
    --raw_score_col d_rmse \
    --lhi_col lhi_rmse \
    --lhi_trigger 0.25 \
    --health_reference_cycles 50 \
    --prompt_variant kg \
    --decision_mode current_lhi_only \
    --model qwen3.5:9b \
    --timeout 600 \
    --ollama_num_predict 512 \
    --save_recent_ollama_outputs 20
