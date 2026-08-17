#!/usr/bin/env bash
set -euo pipefail

project_root="/home/lwh/Documents/Code/LLM-PredM"
python_bin="/home/lwh/anaconda3/envs/default/bin/python"
cd "$project_root"

exec "$python_bin" -m src.RAG_cmapss.joint_simulation \
    --fds FD001 \
    --lhi_dir outputs/CMAPSS/cluster_20/lhi_fix \
    --output_dir outputs/CMAPSS/RAG/agentic_evaluation_batch_w10/FD001 \
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
