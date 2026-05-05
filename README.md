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
  --context_length 0 \
  --prediction_length 5 \
  --target_transform context_robust \
  --covariate_modes past_only known_future \
  --local_files_only
```

Useful options:

- `--eval_split train|test`: choose C-MAPSS split for zero-shot evaluation.
- `--target_transform context_robust`: forecast `(sensor - last_context_value) / context_MAD`, then restore to sensor scale using the same context statistics. This avoids future leakage and helps FD001/FD003 slow-trend forecasting.
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

### 3. Evaluate Forecasting Metrics

This evaluates every rolling forecast round in `window_forecasts.csv` with two
cycle-level metrics:

- Forecast Error: min-max normalize `y_true` and `y_pred` with each window's
  past context range, then compute MAE and RMSE by sensor plus an `ALL` sensor
  aggregate.
- Condition-matched Forecast State Drift: use the first 50 cycles of each unit
  as healthy reference, match by the operating-condition keys from
  `plot_operating_condition_clusters.py`, then compute MAE and RMSE between the
  forecast state and the healthy condition-matched reference.

```bash
/home/lwh/anaconda3/bin/conda run --no-capture-output -n default python -m src.zero_shot_cmapss.forecasting_evaluation \
  --data_dir dataset/CMAPSSData \
  --forecast_dir outputs/roll_5 \
  --output_dir outputs/roll_5/forecasting_evaluation \
  --rolling_window 5 \
  --plot_units FD001:1 FD002:1 FD003:1 FD004:1
```

By default, the script evaluates all covariate modes present in
`window_forecasts.csv`. Use `--covariate_modes known_future` or
`--covariate_modes past_only` to restrict the comparison.

Outputs:

- `forecast_error_round_metrics.csv`: per-unit, per-round MAE/RMSE by sensor and `ALL`.
- `forecast_error_fd_level.csv`: FD-level median trend over units.
- `condition_matched_drift_round_metrics.csv`: per-unit, per-round condition-matched drift.
- `condition_matched_drift_fd_level.csv`: FD-level condition-matched drift trend.
- `healthy_condition_reference.csv`: healthy references by unit, condition, and sensor.
- `*_fd_level.png` and `unit_plots/*.png`: cycle-axis metric plots.

### 4. Compute Log-Ratio LHI

This computes a lightweight log-ratio health indicator from condition-matched
min-max drift. Min-max ranges are computed from each forecast window's past
context, matching `forecasting_evaluation.py` and avoiding future leakage.
Condition means are computed inside each engine only, using
`cycle <= --healthy_cycles`. The LHI baseline is calibrated from the initial
forecast drift immediately after the healthy reference interval.

```bash
/home/lwh/anaconda3/bin/conda run --no-capture-output -n default python -m src.zero_shot_cmapss.lhi_indicator \
  --forecast_dir outputs/cluster_20 \
  --output_dir outputs/cluster_20/lhi \
  --healthy_cycles 50 \
  --rolling_window 5 \
  --plot_units FD001:1 FD002:4 FD004:3
```

For each forecast window and sensor:

```text
y_pred_norm = (y_pred - min_past_context) / (max_past_context - min_past_context)
```

For each unit, sensor, and operating condition, the raw healthy mean is computed
from `cycle <= --healthy_cycles`, then transformed with the same forecast-window
past min-max range:

```text
mean_healthy_norm = (mean_healthy_raw - min_past_context) / (max_past_context - min_past_context)
```

The forecast drift is:

```text
D_RMSE = sqrt(mean_s (y_pred_norm(t,s) - mean_healthy_norm(condition(t),s))^2)
```

The red drift baseline in the plots is calibrated per unit and covariate mode:

```text
B_RMSE = mean(D_RMSE) over the first forecast block after --healthy_cycles
```

By default this means the first forecast block after `--healthy_cycles`. Use
`--baseline_cycles N` to average over the first `N` monitor target cycles
instead.

Then:

```text
LHI = log((D_RMSE + eps) / (B_RMSE + eps))
```

Outputs:

- `lhi_scores.csv`: per forecast cycle `D_MAE`, `D_RMSE`, `LHI_MAE`, and `LHI_RMSE`.
- `lhi_baselines.csv`: unit-specific `B_MAE` and `B_RMSE`.
- `baseline_forecast_points.csv`: forecast drift rows used to average `B`.
- `past_minmax_ranges.csv`: forecast-window past-context min-max ranges and usable flags.
- `unit_drift/*_drift.png`: drift curves with red `B_RMSE` baseline.
- `unit_lhi/*_lhi.png`: log-ratio LHI curves.

Batch plot any unit interval from existing `lhi_scores.csv`:

```bash
/home/lwh/anaconda3/bin/conda run --no-capture-output -n default python -m src.zero_shot_cmapss.plot_lhi_units \
  --lhi_dir outputs/cluster_20/lhi \
  --fd FD001 \
  --unit_start 1 \
  --unit_end 20
```

This writes one combined drift/LHI figure per unit. Use `--metric mae` to plot
MAE-based drift and LHI instead of RMSE.

### 5. Compute CARD Health Indicator

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

## Maintenance SFT Project

This repository also includes a separate maintenance-log supervised fine-tuning project under `src/llm_sft_maintenace`.

### Maintenance SFT Preprocess

```bash
conda run -n llm python -m src.llm_sft_maintenace.preprocess \
  --dataset_dir dataset \
  --output_dir artifacts/maintenance_sft
```

This pipeline uses:

- Training split: `dataset/synthetic_maintenance_log_train.csv`
- Validation split: `dataset/synthetic_maintenance_log_validation.csv`
- Test split: `dataset/synthetic_maintenance_log_test.csv`

The maintenance SFT input is split into:

- Text input: domain/task prompt plus `event_id`, `case_id`, `work_order_id`, `vehicle_id`, `system`, `subsystem`, `component`, and `maintenance_note`
- Numeric input: `spec_profile`, `event_time_step`, `predicted_rul`, and `sensor_readings`

The supervision targets are:

- `action_taken`
- `action_priority`

### Maintenance SFT Train

```bash
conda run -n llm python -m src.llm_sft_maintenace.train \
  --artifacts_dir artifacts/maintenance_sft \
  --base_model_path models/Qwen3-4B \
  --output_dir outputs/maintenance_sft_lora \
  --load_in_4bit
```

The maintenance SFT model uses:

- Qwen text embeddings for the textual fields
- A numeric feature projector for structured numeric inputs
- Two projection heads on top of the LLM hidden states:
  - maintenance action classification
  - action priority classification

Validation is run after each epoch, and test evaluation is run after training.

### Maintenance SFT Evaluate

You can evaluate a saved checkpoint on the test split and run a no-prompt ablation with:

```bash
conda run -n llm python -m src.llm_sft_maintenace.evaluate \
  --artifacts_dir artifacts/maintenance_sft \
  --checkpoint_dir outputs/maintenance_sft_lora/best \
  --split test \
  --load_in_4bit \
  --output_json outputs/maintenance_sft_lora/test_ablation.json
```

The evaluation script reports:

- full-prompt test performance
- no-prompt ablation performance
- action accuracy
- priority accuracy
- joint accuracy

## Notes

- `ollama qwen3.5:4b` can be used locally for deployment and inference, but its underlying format is `GGUF`, so it cannot be fine-tuned directly with `transformers + peft`.
- For training, you must provide a Hugging Face-compatible Qwen checkpoint path. The resulting LoRA adapter and custom task heads will be saved under `output_dir`.
