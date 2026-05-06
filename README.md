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

- `cluster_covariate`: first split each forecast window by operating-condition
  cluster, then pass setting1-3 as past and future covariates. This is the
  default experiment mode and replaces the old `known_future` name.
- `future_covariate`: do not split by operating-condition cluster; pass raw
  multivariate sensor history plus setting1-3 as past and future covariates.
- `no_covariate`: pass only the multivariate sensor history.

The first Chronos run downloads `amazon/chronos-2` into the Hugging Face cache.
The forecasting script disables Hugging Face Xet by default because partial Xet
downloads can stall with range-resume errors on some networks.

### 1. Run Zero-Shot Forecasting

This command evaluates on `train_FDxxx.txt` and creates one local database
row-set for every eligible forecast-start cycle:

```bash
/home/lwh/anaconda3/bin/conda run --no-capture-output -n default python -m src.zero_shot_cmapss.chronos2_cmapss_forecast \
  --data_dir dataset/CMAPSSData \
  --output_dir outputs/stride_1_h20 \
  --eval_split train \
  --device cuda \
  --context_length 0 \
  --prediction_length 20 \
  --forecast_start_cycle 20 \
  --stride 1 \
  --target_transform context_minmax \
  --covariate_modes cluster_covariate \
  --local_files_only
```

The generation script does not plot. With `--stride 1`, every eligible
forecast-start cycle gets a full `--prediction_length` horizon. For an engine
with 150 cycles, forecast starts are cycles 20-150. Horizon steps beyond the
observed sequence are saved with `y_true` empty, so LHI/decision signals can
still use the forecast state. MAE/MSE/RMSE metrics are computed separately using
only full-ground-truth, non-overlapping starts, equivalent to metric stride
`prediction_length`.

Useful options:

- `--eval_split train|test`: choose C-MAPSS split for zero-shot evaluation.
- `--target_transform context_minmax|none`: default `context_minmax` scales each sensor with the current forecast window's past-context min/max, then restores predictions to raw scale. Use `none` for a raw-scale ablation.
- `--forecast_start_cycle 20`: first forecast-start cycle.
- `--forecast_end_cycle N`: optional latest forecast-start cycle; by default the script uses the final observed cycle.

Main outputs:

- `metrics.csv`: MAE, MSE, and RMSE by FD, covariate mode, and sensor, computed from `metric_window_forecasts.csv`.
- `window_forecasts.csv`: raw rolling-window predictions for every forecast start and every horizon step. No aggregation is applied.
- `metric_window_forecasts.csv`: full-ground-truth, non-overlapping windows used for MAE/MSE/RMSE.
- `window_metrics.csv`: raw window-level MAE, MSE, and RMSE.
- `anomaly_scores.csv`: window-level normalized forecasting-error score.
- `sensor_anomaly_scores.csv`: sensor-level normalized forecasting-error score.
- `run_config.json`: exact experiment configuration.

Plot any unit interval from the generated forecast database:

```bash
/home/lwh/anaconda3/bin/conda run --no-capture-output -n default python -m src.zero_shot_cmapss.plot_forecast_units \
  --forecast_dir outputs/stride_1_h20 \
  --fd FD001 \
  --unit_start 1 \
  --unit_end 20
```

The plotting script reads `prediction_length` from `run_config.json` and plots
forecast starts at that stride, so the displayed horizons do not overlap.

### 2. Plot Rolling Anomaly Trends

This older helper post-processes `anomaly_scores.csv` and plots rolling
mean/median trends over forecast start cycle when available.

```bash
/home/lwh/anaconda3/bin/conda run -n default python -m src.zero_shot_cmapss.plot_anomaly_scores \
  --input_csv outputs/stride_1_h20/anomaly_scores.csv \
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
  --rolling_window 5
```

By default, the script evaluates all covariate modes present in
`window_forecasts.csv`. Use `--covariate_modes cluster_covariate`,
`--covariate_modes future_covariate`, or `--covariate_modes no_covariate` to
restrict the comparison.

Outputs:

- `forecast_error_round_metrics.csv`: per-unit, per-round MAE/RMSE by sensor and `ALL`.
- `forecast_error_metric_rows.csv`: rows used for Forecast Error after filtering to full-ground-truth, non-overlapping starts.
- `forecast_error_fd_level.csv`: FD-level median trend over units.
- `condition_matched_drift_round_metrics.csv`: per-unit, per-round condition-matched drift.
- `condition_matched_drift_fd_level.csv`: FD-level condition-matched drift trend.
- `healthy_condition_reference.csv`: healthy references by unit, condition, and sensor.

Plot any unit interval from the generated evaluation metrics:

```bash
/home/lwh/anaconda3/bin/conda run --no-capture-output -n default python -m src.zero_shot_cmapss.plot_forecasting_evaluation \
  --evaluation_dir outputs/roll_5/forecasting_evaluation \
  --fd FD001 \
  --unit_start 1 \
  --unit_end 20 \
  --plot_fd_level
```

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
  --top_k_sensors 5
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
instead. `--top_k_sensors` controls how many sensor-level drift contributors are
reported for each LHI point.

Then:

```text
LHI = log((D_RMSE + eps) / (B_RMSE + eps))
```

Outputs:

- `lhi_scores.csv`: per forecast cycle `D_MAE`, `D_RMSE`, `LHI_MAE`, `LHI_RMSE`,
  `top_drift_sensors`, and top sensor drift values.
- `top_drift_sensors.csv`: long-format ranked sensor contributors for each forecast cycle.
- `sensor_lhi_components.csv`: all sensor-level drift components before top-k filtering.
- `lhi_baselines.csv`: unit-specific `B_MAE` and `B_RMSE`.
- `baseline_forecast_points.csv`: forecast drift rows used to average `B`.
- `past_minmax_ranges.csv`: forecast-window past-context min-max ranges and usable flags.

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

### 5. Maintenance Decision Signals

This computes condition-matched drift decision signals, dynamic
thresholds, slopes, and log-ratio LHI tables. The generation script writes CSV
data only.

```bash
/home/lwh/anaconda3/bin/conda run --no-capture-output -n default python -m src.zero_shot_cmapss.maintenance_decision_analysis \
  --forecast_dir outputs/stride_1_h20 \
  --output_dir outputs/stride_1_h20/maintenance_decision \
  --covariate_mode cluster_covariate \
  --healthy_cycles 50 \
  --start_cycle 20
```

Plot any unit interval from the generated decision tables:

```bash
/home/lwh/anaconda3/bin/conda run --no-capture-output -n default python -m src.zero_shot_cmapss.plot_maintenance_decision \
  --decision_dir outputs/stride_1_h20/maintenance_decision \
  --fd FD001 \
  --unit_start 1 \
  --unit_end 20 \
  --plot_fd_level
```

### 6. Compute CARD Health Indicator

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
