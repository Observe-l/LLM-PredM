# C-MAPSS LLM-PredM Knowledge Graph

This directory contains a maintenance-oriented KG for the C-MAPSS zero-shot
forecasting workflow described in `documents/KG_design.md`.

The KG is intentionally not a direct RUL predictor. It translates forecast
evidence into explainable maintenance hypotheses:

```text
Forecast evidence
-> dataset-specific threshold hypothesis
-> sensor/component degradation hypothesis
-> time-aware maintenance action
-> engineer or breakdown feedback
-> reflection rule update
```

## Files

- `static_domain_triples.csv`: aeroengine structure, component characteristics,
  cross-section parameter dependencies, and sensor-pattern hypothesis support.
- `sensor_mapping.csv`: selected C-MAPSS sensors, measured physical quantities,
  physical quantity types, and component associations.
- `alias_mapping.csv`: bridge triples between expanded C-MAPSS sensor quantity
  names and DKAMFormer parameter nodes such as `P3`, `T3`, `Ps3`, `N2`, and
  `NRc`.
- `dataset_rules.csv`: FD001-FD004 degradation-mode assumptions, threshold
  hypotheses, action policies, and initial reflection-rule anchors.
- `hypothesis_action_rules.csv`: allowed maintenance actions and how degradation
  or risk hypotheses map to action types.
- `reflection_rules.csv`: seed reflection rules plus append-only dynamic rules
  updated from maintenance execution feedback or breakdown feedback.
- `dynamic_case_schema.json`: schema for per-inference `ForecastCase`,
  `ForecastSummary`, `ForecastObservation`, and `ActionHypothesis` nodes.

## Core Query Path

For top sensors `S7`, `S11`, and `S3` in `FD002`, the intended retrieval path is:

```text
FD002 -> allows_hypothesis -> HPC_related_degradation
S7 -> measures -> P30_total_pressure_at_HPC_outlet -> associated_with -> HPC
P30_total_pressure_at_HPC_outlet -> alias_of -> P3 -> supports_working_characteristic -> Ghpc
S11 -> measures -> Ps30_static_pressure_at_HPC_outlet -> associated_with -> HPC
Ps30_static_pressure_at_HPC_outlet -> alias_of -> Ps3
S3 -> measures -> T30_total_temperature_at_HPC_outlet -> associated_with -> HPC
T30_total_temperature_at_HPC_outlet -> alias_of -> T3 -> supports_working_characteristic -> Ghpc
HPC -> supports_hypothesis -> HPC_related_degradation
HPC_related_degradation -> suggests_action_type -> schedule_HPC_maintenance
```

The forecast score itself should be stored as observation evidence, not as a
hard health event. Thresholds are represented as dataset-specific hypotheses and
can be revised by reflection feedback.

## Dynamic Case Construction

For each zero-shot inference, create one `ForecastCase` node and attach:

- `ForecastSummary`: current score, peak score, trend, crossing cycles, horizon,
  persistent high-risk duration, first persistent pattern cycle, dominant
  component hypothesis, dominant top sensors, and peak-cycle top sensors.
- `ForecastObservation`: one node per top sensor or persistent sensor pattern.
- `ActionHypothesis`: the LLM-proposed action type and action time.

Recommended case id:

```text
ForecastCase_<FD>_Engine<unit_id>_Cycle<cutoff_cycle>
```

## Reflection Update Rule

Do not overwrite historical actions. Append a row to `reflection_rules.csv`:

```text
feedback -> revise action type -> adjust action time -> adjust threshold -> adjust component preference
```

Examples:

- Over-maintenance on weak evidence raises the relevant dataset threshold and
  revises future similar actions toward `schedule_monitoring`.
- Breakdown before scheduled maintenance lowers the threshold and moves future
  action time earlier.

## Retrieval Notes

Retrieval code should treat `disallows_action_type` as a hard negative policy.
It should not expose disallowed actions as candidate paths for the LLM. If a
future KG version uses numeric policy weights, filter candidate policy edges with
`edge_weight > 0`.

Combustion-side and turbine-side evidence, such as `S12`, `S4`, `S20`, and
`S21`, maps to `uncertain_component_degradation` in this action taxonomy unless
it appears together with strong FAN or HPC evidence.
