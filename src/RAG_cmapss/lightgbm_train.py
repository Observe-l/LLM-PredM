from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from .config import DEFAULT_KG_DIR
from .lightgbm_features import (
    FEATURE_COLUMNS,
    RISK_FEATURE_COLUMNS,
    filter_risk_training_rows,
    filter_update_training_rows,
    read_reflection_training_rows,
    risk_labels_from_reflection_rows,
    training_features_from_reflection_rows,
    update_labels_from_reflection_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Layer-3 LightGBM tools from reflection_rules.csv.")
    parser.add_argument("--reflection_rules", type=Path, default=DEFAULT_KG_DIR / "reflection_rules.csv")
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--min_rows", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_lightgbm_models(
        reflection_rules=args.reflection_rules,
        output_dir=args.output_dir,
        min_rows=args.min_rows,
    )
    print(json.dumps(summary, indent=2))


def train_lightgbm_models(
    reflection_rules: str | Path,
    output_dir: str | Path,
    min_rows: int = 4,
) -> dict[str, Any]:
    try:
        from lightgbm import LGBMClassifier
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "lightgbm is not installed in this Python environment. Install it to train models, "
            "or run the agent without model files to use the built-in conservative fallback."
        ) from exc

    reflection_rules = Path(reflection_rules)
    output_dir = Path(output_dir)
    rows = read_reflection_training_rows(reflection_rules)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"reflection_rows": len(rows), "models": {}}

    risk_rows = filter_risk_training_rows(rows)
    if len(risk_rows) >= min_rows and len(set(risk_labels_from_reflection_rows(risk_rows))) >= 2:
        model = _new_classifier(LGBMClassifier)
        x = training_features_from_reflection_rows(risk_rows, columns=RISK_FEATURE_COLUMNS)
        y = risk_labels_from_reflection_rows(risk_rows)
        model.fit(x, y)
        path = output_dir / "lightgbm_risk.pkl"
        _save_model(path, model, feature_columns=RISK_FEATURE_COLUMNS, feature_schema="risk_v2_no_feedback_or_action_result")
        summary["models"]["risk"] = {
            "path": str(path),
            "rows": len(risk_rows),
            "classes": list(map(int, model.classes_)),
            "feature_schema": "risk_v2_no_feedback_or_action_result",
            "excluded_features": [
                "action_to_peak_gap",
                "action_to_warning_gap",
                "action_to_persistence_gap",
                "previous_action_type_code",
                "feedback_label_code",
            ],
        }
    else:
        summary["models"]["risk"] = {"skipped": "not enough labeled rows or only one class", "rows": len(risk_rows)}

    for target, filename in [
        ("then_adjust_threshold", "lightgbm_update_threshold.pkl"),
        ("recommended_time_rule", "lightgbm_update_timing.pkl"),
    ]:
        update_rows = filter_update_training_rows(rows, target)
        labels = update_labels_from_reflection_rows(update_rows, target)
        if len(update_rows) >= min_rows and len(set(labels)) >= 2:
            model = _new_classifier(LGBMClassifier)
            x = training_features_from_reflection_rows(update_rows)
            model.fit(x, labels)
            path = output_dir / filename
            _save_model(path, model, feature_columns=FEATURE_COLUMNS, feature_schema=f"update_{target}_v1")
            summary["models"][target] = {"path": str(path), "rows": len(update_rows), "classes": list(model.classes_)}
        else:
            summary["models"][target] = {
                "skipped": "not enough labeled rows or only one class",
                "rows": len(update_rows),
            }

    summary_path = output_dir / "lightgbm_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _save_model(
    path: Path,
    model: Any,
    feature_columns: list[str],
    feature_schema: str,
) -> None:
    payload = {
        "model": model,
        "feature_columns": feature_columns,
        "feature_schema": feature_schema,
        "classes": list(model.classes_),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _new_classifier(cls: Any) -> Any:
    return cls(
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=1,
        min_data_in_bin=1,
        random_state=7,
        verbosity=-1,
    )


if __name__ == "__main__":
    main()
