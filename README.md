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

## Chronos-2 Zero-Shot C-MAPSS Workflow

The zero-shot C-MAPSS code lives under `src/zero_shot_cmapss/`. It uses
`amazon/chronos-2` to forecast selected sensors and then derives health indicators
from the forecast trajectories.

Targets:

```text
s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21
```

Operating-condition covariates:

```text
setting1, setting2, setting3
```

Covariate modes:

- `past_only`: operating conditions are observed only in the context window.
- `known_future`: operating conditions are also known over the forecast horizon.

The first Chronos run downloads `amazon/chronos-2` into the Hugging Face cache.
The forecasting script disables Hugging Face Xet by default because partial Xet
downloads can stall with range-resume errors on some networks.

### 1. Run Zero-Shot Forecasting

This command evaluates on `train_FDxxx.txt`, uses stride-5 non-overlapping
forecast blocks, and applies leakage-free context residual scaling:

```bash
/home/lwh/anaconda3/bin/conda run --no-capture-output -n default python -m src.zero_shot_cmapss.chronos2_cmapss_forecast \
  --data_dir dataset/CMAPSSData \
  --output_dir outputs/stride_5_robust \
  --eval_split train \
  --device cuda \
  --context_length 40 \
  --prediction_length 5 \
  --stride 5 \
  --normalization none \
  --target_transform context_robust \
  --covariate_modes past_only known_future \
  --local_files_only
```

Useful options:

- `--eval_split train|test`: choose C-MAPSS split for zero-shot evaluation.
- `--target_transform context_robust`: forecast `(sensor - last_context_value) / context_MAD`, then restore to sensor scale using the same context statistics. This avoids future leakage and helps FD001/FD003 slow-trend forecasting.
- `--normalization none`: recommended default. `zscore` is kept only for ablation/reproduction.
- `--plot_units FD001:4 FD004:3`: generate full-unit forecast plots for specific units.

Main outputs:

- `metrics.csv`: full-curve MAE, MSE, and RMSE by FD, covariate mode, and sensor.
- `forecasts.csv`: full forecast curves after aggregating overlapping horizon predictions by cycle.
- `window_forecasts.csv`: raw rolling-window predictions for every horizon step.
- `window_metrics.csv`: raw window-level MAE, MSE, and RMSE.
- `anomaly_scores.csv`: window-level normalized forecasting-error score.
- `sensor_anomaly_scores.csv`: sensor-level normalized forecasting-error score.
- `plots/`: grouped-by-sensor full-unit forecast plots.
- `run_config.json`: exact experiment configuration.

### 2. Plot Rolling Anomaly Trends

This post-processes `anomaly_scores.csv` and plots rolling mean/median trends over
forecast start cycle.

```bash
/home/lwh/anaconda3/bin/conda run -n default python -m src.zero_shot_cmapss.plot_anomaly_scores \
  --input_csv outputs/stride_5_robust/anomaly_scores.csv \
  --rolling_window 5 \
  --plot_units FD001:4 FD004:3
```

Outputs:

- `anomaly_plots/fd_level_anomaly_trends.png`
- `anomaly_plots/fd_level_anomaly_trends.csv`
- `anomaly_plots/unit_trends/*_anomaly_trend.png`

With `stride=5`, `--rolling_window 5` smooths roughly 25 cycles.

### 3. Compute CARD Health Indicator

CARD uses forecast-trajectory features and condition-aware historical references.
The script first trains a 6-cluster KMeans operating-regime classifier on
`FD002/train_FD002.txt` using `setting1,setting2,setting3`.

Expected regime-class validation:

- FD001: 1 class
- FD002: 6 classes
- FD003: 1 class
- FD004: 6 classes

Recommended CARD command:

```bash
/home/lwh/anaconda3/bin/conda run --no-capture-output -n default python -m src.zero_shot_cmapss.card_indicator \
  --data_dir dataset/CMAPSSData \
  --forecast_dir outputs/stride_5_robust \
  --output_dir outputs/stride_5_robust/card_knn \
  --regime_reference distribution \
  --reference_strategy knn \
  --knn_reference 30 \
  --plot_units FD001:1 FD002:1 FD003:1 FD004:1
```

CARD details:

- Forecast feature per window/sensor: `[mean(prediction), gamma * slope(prediction)]`.
- FD002 KMeans regimes define condition labels.
- `--regime_reference distribution` uses the horizon regime distribution.
- `--reference_strategy knn` selects nearest historical regime distributions from the same unit and sensor, avoiding the sparsity of exact regime-sequence matching.

Outputs:

- `regime_class_counts.csv`: verifies FD001/FD003 have 1 class and FD002/FD004 have 6 classes.
- `card_features.csv`: extracted forecast trajectory features and horizon regime descriptors.
- `card_scores.csv`: window-level CARD health indicator.
- `card_sensor_details.csv`: per-sensor CARD components and reference counts.
- `plots/*_card.png`: CARD curves for selected units.

## Model Design

- Text input: dataset/task description plus the current vehicle specification profile, encoded with Qwen's original embedding layer.
- Time-series input: sensor observations are processed by a custom patch embedder and then reprogrammed into the LLM hidden space through attention over text prototypes.
- Output head: the model does not decode through the original vocabulary head. Instead, it uses a custom `RUL regression head + maintenance action classification head`.

## Notes

- `ollama qwen3.5:4b` can be used locally for deployment and inference, but its underlying format is `GGUF`, so it cannot be fine-tuned directly with `transformers + peft`.
- For training, you must provide a Hugging Face-compatible Qwen checkpoint path. The resulting LoRA adapter and custom task heads will be saved under `output_dir`.
