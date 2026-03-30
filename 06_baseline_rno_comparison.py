from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts") / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Coursework3.RNO_1D_Skeleton import (
    ExperimentConfig,
    evaluate_checkpoint,
    pick_best_family,
    plot_residual_analysis,
    prepare_data,
    read_json,
    train_model,
    write_json,
)


ARTIFACT_ROOT = Path("artifacts")
TRAIN_PATH = Path("Coursework3/viscodata_3mat.mat")
SPLIT_SEED = 20260328
DEVICE = "cpu"
REFERENCE_FAMILY = "gru"
BASELINE_CORE = "baseline_rno"
BASELINE_RUN_NAME = "baseline_rno_from_best_gru"
FAMILIES = ["rnn", "gru", "lstm"]


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    ensure_directory(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def best_payload_for_family(family: str) -> dict:
    return read_json(ARTIFACT_ROOT / "optuna" / f"best_params_{family}.json")


def train_baseline_rno(data_bundle: dict) -> dict:
    reference_payload = best_payload_for_family(REFERENCE_FAMILY)
    base_config = ExperimentConfig(**json.loads(reference_payload["best_user_attrs"]["config"]))
    base_config.CORE_TYPE = BASELINE_CORE
    base_config.MODEL_TAG = BASELINE_RUN_NAME
    base_config.VERBOSE = False

    result = train_model(
        data_bundle=data_bundle,
        config=base_config,
        artifact_root=ARTIFACT_ROOT,
        run_name=BASELINE_RUN_NAME,
    )
    return {
        "config": base_config,
        "train_result": result,
        "reference_payload": reference_payload,
    }


def evaluate_named_checkpoint(name: str, checkpoint_path: Path, data_bundle: dict) -> dict:
    test_eval = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        x=data_bundle["x_test"],
        y=data_bundle["y_test"],
        output_normalizer=data_bundle["output_normalizer"],
        artifact_root=ARTIFACT_ROOT,
        run_name=f"{name}_test",
        y_true0=data_bundle["y_test"][:, 0],
    )
    prediction_arrays = np.load(test_eval["prediction_path"])
    residual_plot_path = plot_residual_analysis(
        y_true=prediction_arrays["y_true"],
        y_pred=prediction_arrays["y_pred"],
        output_normalizer=data_bundle["output_normalizer"],
        save_path=ARTIFACT_ROOT / "figures" / f"{name}_test_residuals.png",
    )
    return {
        **test_eval,
        "residual_plot_path": residual_plot_path,
    }


def make_training_curve_comparison(baseline_history_path: Path) -> Path:
    histories: list[tuple[str, Path, str]] = []
    for family in FAMILIES:
        payload = best_payload_for_family(family)
        trial_number = int(payload["best_trial_number"])
        histories.append(
            (
                family.upper(),
                ARTIFACT_ROOT / "logs" / f"optuna_{family}_trial_{trial_number:04d}_history.csv",
                family,
            )
        )
    histories.append(("Baseline RNO", baseline_history_path, "baseline"))

    colors = {
        "rnn": "#1f77b4",
        "gru": "#2ca02c",
        "lstm": "#d62728",
        "baseline": "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for label, history_path, key in histories:
        history = pd.read_csv(history_path)
        ax.plot(history["epoch"], history["val_loss"], linewidth=2.0, label=label, color=colors[key])
    ax.set_title("Validation-loss comparison: best models vs baseline RNO")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    return save_figure(fig, ARTIFACT_ROOT / "figures" / "06_best_models_validation_curve_comparison.png")


def make_test_metric_bar(comparison_df: pd.DataFrame, metric: str, ylabel: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    ax.bar(comparison_df["model"], comparison_df[metric], color=colors[: len(comparison_df)], alpha=0.9)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Test-set comparison: {ylabel}")
    ax.grid(axis="y", alpha=0.3)
    return save_figure(fig, ARTIFACT_ROOT / "figures" / filename)


def main() -> None:
    os.environ["COURSEWORK_DEVICE"] = DEVICE
    data_bundle = prepare_data(train_path=TRAIN_PATH, artifact_root=ARTIFACT_ROOT, split_seed=SPLIT_SEED)

    baseline_payload = train_baseline_rno(data_bundle=data_bundle)
    baseline_result = baseline_payload["train_result"]
    baseline_checkpoint = Path(baseline_result["checkpoint_path"])

    rows: list[dict[str, object]] = []
    evaluation_payloads: dict[str, dict] = {}

    for family in FAMILIES:
        payload = best_payload_for_family(family)
        checkpoint_path = Path(payload["best_user_attrs"]["checkpoint_path"])
        evaluation = evaluate_named_checkpoint(name=f"best_{family}", checkpoint_path=checkpoint_path, data_bundle=data_bundle)
        evaluation_payloads[family] = evaluation
        rows.append(
            {
                "model": family.upper(),
                "core_type": family,
                "source": "best_optuna",
                "best_val_loss": float(payload["best_value"]),
                "best_epoch": int(payload["best_user_attrs"]["best_epoch"]),
                "test_rmse": evaluation["metrics"]["rmse"],
                "test_mae": evaluation["metrics"]["mae"],
                "test_r2": evaluation["metrics"]["r2"],
                "test_nrmse": evaluation["metrics"]["nrmse"],
                "checkpoint_path": str(checkpoint_path),
            }
        )

    baseline_eval = evaluate_named_checkpoint(name="baseline_rno", checkpoint_path=baseline_checkpoint, data_bundle=data_bundle)
    evaluation_payloads["baseline_rno"] = baseline_eval
    rows.append(
        {
            "model": "Baseline RNO",
            "core_type": BASELINE_CORE,
            "source": f"matched_to_best_{REFERENCE_FAMILY}",
            "best_val_loss": float(baseline_result["best_val_loss"]),
            "best_epoch": int(baseline_result["best_epoch"]),
            "test_rmse": baseline_eval["metrics"]["rmse"],
            "test_mae": baseline_eval["metrics"]["mae"],
            "test_r2": baseline_eval["metrics"]["r2"],
            "test_nrmse": baseline_eval["metrics"]["nrmse"],
            "checkpoint_path": str(baseline_checkpoint),
        }
    )

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values("best_val_loss").reset_index(drop=True)
    comparison_path = ARTIFACT_ROOT / "reports" / "06_baseline_rno_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    curve_path = make_training_curve_comparison(baseline_history_path=Path(baseline_result["history_path"]))
    rmse_bar_path = make_test_metric_bar(
        comparison_df=comparison_df,
        metric="test_rmse",
        ylabel="Test RMSE",
        filename="06_test_rmse_comparison.png",
    )
    r2_bar_path = make_test_metric_bar(
        comparison_df=comparison_df,
        metric="test_r2",
        ylabel="Test R²",
        filename="06_test_r2_comparison.png",
    )

    summary = {
        "device": DEVICE,
        "reference_family": REFERENCE_FAMILY,
        "baseline_core": BASELINE_CORE,
        "baseline_checkpoint": str(baseline_checkpoint),
        "comparison_csv": str(comparison_path),
        "validation_curve_plot": str(curve_path),
        "rmse_bar_plot": str(rmse_bar_path),
        "r2_bar_plot": str(r2_bar_path),
    }
    summary_path = ARTIFACT_ROOT / "reports" / "06_baseline_rno_comparison_summary.json"
    write_json(summary_path, summary)
    print(comparison_df.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
