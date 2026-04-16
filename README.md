# LLM-PredM

A multimodal LLM training prototype for predictive maintenance.

This repository contains a preprocessing pipeline for the raw CSV files and a Qwen LoRA fine-tuning workflow.

## Preprocess

```bash
conda run -n llm python -m src.llm_predm.preprocess \
  --dataset_dir dataset \
  --output_dir artifacts/qwen_predm \
  --window_len 1 \
  --stride 1 \
  --action_train_ratio 0.7
```

## Train

```bash
conda run -n llm python -m src.llm_predm.train \
  --artifacts_dir artifacts/qwen_predm \
  --base_model_path models/Qwen3-4B \
  --output_dir outputs/qwen3-4b_predm_lora \
  --load_in_4bit
```

## Model Design

- Text input: dataset/task description plus the current vehicle specification profile, encoded with Qwen's original embedding layer.
- Time-series input: sensor observations are processed by a custom patch embedder and then reprogrammed into the LLM hidden space through attention over text prototypes.
- Output head: the model does not decode through the original vocabulary head. Instead, it uses a custom `RUL regression head + maintenance action classification head`.

## Notes

- `ollama qwen3.5:4b` can be used locally for deployment and inference, but its underlying format is `GGUF`, so it cannot be fine-tuned directly with `transformers + peft`.
- For training, you must provide a Hugging Face-compatible Qwen checkpoint path. The resulting LoRA adapter and custom task heads will be saved under `output_dir`.
