from __future__ import annotations

"""Sweep hidden size for the paper-style RNO with and without explicit strain-rate input."""

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts") / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from Coursework3.RNO_1D_Skeleton import (
    ExperimentConfig,
    evaluate_checkpoint,
    prepare_data,
    read_json,
    run_hidden_threshold_grid,
    write_json,
)


ARTIFACT_ROOT = Path("artifacts")
TRAIN_PATH = Path("Coursework3/viscodata_3mat.mat")
SPLIT_SEED = 20260328
DEVICE = "cpu"

REFERENCE_FAMILY = "gru"
REFERENCE_PARAMS_PATH = ARTIFACT_ROOT / "optuna" / f"best_params_{REFERENCE_FAMILY}.json"

BATCH_SIZE = 32
MIN_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 25
SWEEP_SEEDS = [20260341]
HIDDEN_GRID = [0, 1, 2, 3, 4, 5]
TOLERANCE_RATIO = 0.05

VARIANTS = [
    {
        "label": "paper_rno_no_rate",
        "use_rate": False,
        "run_prefix": "09_paper_rno_no_rate_h0to5",
        "color": "#1f77b4",
    },
    {
        "label": "paper_rno_with_rate",
        "use_rate": True,
        "run_prefix": "09_paper_rno_with_rate_h0to5",
        "color": "#d62728",
    },
]


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


def build_paper_config(use_rate: bool, seed: int, model_tag: str) -> ExperimentConfig:
    """Start from the tuned GRU recipe and swap only the paper-RNO-specific settings."""
    payload = read_json(REFERENCE_PARAMS_PATH)
    config = ExperimentConfig(**json.loads(payload["best_user_attrs"]["config"]))
    config.CORE_TYPE = "paper_rno"
    config.PAPER_USE_RATE_IN_STRESS = bool(use_rate)
    config.BATCH_SIZE = BATCH_SIZE
    config.MIN_EPOCHS = MIN_EPOCHS
    config.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE
    config.SEED = int(seed)
    config.MODEL_TAG = model_tag
    config.VERBOSE = False
    return config


def evaluate_sweep_results(
    results_df: pd.DataFrame,
    data_bundle: dict[str, Any],
    variant_label: str,
) -> pd.DataFrame:
    """Evaluate every hidden-size checkpoint on the test set for a second axis of comparison."""
    rows: list[dict[str, Any]] = []
    for row in results_df.sort_values(["n_hidden", "seed"]).itertuples(index=False):
        checkpoint_path = getattr(row, "checkpoint_path")
        n_hidden = int(getattr(row, "n_hidden"))
        seed = int(getattr(row, "seed"))
        eval_name = f"{variant_label}_h{n_hidden}_seed_{seed}_test"
        evaluation = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            x=data_bundle["x_test"],
            y=data_bundle["y_test"],
            output_normalizer=data_bundle["output_normalizer"],
            artifact_root=ARTIFACT_ROOT,
            run_name=eval_name,
            y_true0=data_bundle["y_test"][:, 0],
        )
        rows.append(
            {
                "variant": variant_label,
                "n_hidden": n_hidden,
                "seed": seed,
                "checkpoint_path": checkpoint_path,
                "test_rmse": float(evaluation["metrics"]["rmse"]),
                "test_mae": float(evaluation["metrics"]["mae"]),
                "test_r2": float(evaluation["metrics"]["r2"]),
                "test_nrmse": float(evaluation["metrics"]["nrmse"]),
                "test_relative_l2": float(evaluation["metrics"]["relative_l2"]),
            }
        )
    return pd.DataFrame(rows)


def plot_validation_overlay(grouped_frames: list[pd.DataFrame], labels: list[str], colors: list[str]) -> Path:
    """Overlay validation loss versus hidden size for the two paper-RNO variants."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for grouped, label, color in zip(grouped_frames, labels, colors):
        ax.plot(grouped["n_hidden"], grouped["mean_val_loss"], marker="o", linewidth=2.0, color=color, label=label)
    ax.set_title("paper_rno validation loss vs hidden size")
    ax.set_xlabel("n_hidden")
    ax.set_ylabel("Mean validation loss")
    ax.set_yscale("log")
    ax.set_xticks(HIDDEN_GRID)
    ax.grid(alpha=0.3)
    ax.legend()
    return save_figure(fig, ARTIFACT_ROOT / "figures" / "09_paper_rno_validation_loss_vs_hidden.png")


def plot_test_relative_l2_overlay(test_metrics_df: pd.DataFrame, colors_by_variant: dict[str, str]) -> Path:
    """Plot test relative L2 versus hidden size for the two paper-RNO variants."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    grouped = (
        test_metrics_df.groupby(["variant", "n_hidden"], as_index=False)
        .agg(mean_test_relative_l2=("test_relative_l2", "mean"))
        .sort_values(["variant", "n_hidden"])
    )
    for variant in grouped["variant"].unique():
        subset = grouped.loc[grouped["variant"] == variant]
        ax.plot(
            subset["n_hidden"],
            subset["mean_test_relative_l2"],
            marker="o",
            linewidth=2.0,
            color=colors_by_variant[variant],
            label=variant,
        )
    ax.set_title("paper_rno test relative L2 vs hidden size")
    ax.set_xlabel("n_hidden")
    ax.set_ylabel("Test relative L2")
    ax.set_yscale("log")
    ax.set_xticks(HIDDEN_GRID)
    ax.grid(alpha=0.3)
    ax.legend()
    return save_figure(fig, ARTIFACT_ROOT / "figures" / "09_paper_rno_test_relative_l2_vs_hidden.png")


def main() -> None:
    """Run both hidden sweeps and save combined validation and test summaries."""
    os.environ["COURSEWORK_DEVICE"] = DEVICE
    data_bundle = prepare_data(train_path=TRAIN_PATH, artifact_root=ARTIFACT_ROOT, split_seed=SPLIT_SEED)

    grouped_frames: list[pd.DataFrame] = []
    labels: list[str] = []
    colors: list[str] = []
    sweep_rows: list[pd.DataFrame] = []
    test_metric_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        label = str(variant["label"])
        use_rate = bool(variant["use_rate"])
        run_prefix = str(variant["run_prefix"])
        color = str(variant["color"])
        base_config = build_paper_config(use_rate=use_rate, seed=SWEEP_SEEDS[0], model_tag=run_prefix)
        print(
            f"[09] sweep variant={label} use_rate={use_rate} "
            f"hidden_grid={HIDDEN_GRID} seeds={SWEEP_SEEDS} batch={base_config.BATCH_SIZE}",
            flush=True,
        )
        sweep_result = run_hidden_threshold_grid(
            data_bundle=data_bundle,
            base_config=base_config,
            hidden_grid=HIDDEN_GRID,
            seeds=SWEEP_SEEDS,
            artifact_root=ARTIFACT_ROOT,
            run_prefix=run_prefix,
            tolerance_ratio=TOLERANCE_RATIO,
            reference_loss=None,
            verbose=True,
        )

        grouped = sweep_result["threshold_summary"]["grouped_results"].copy()
        grouped.insert(0, "variant", label)
        grouped_frames.append(grouped)
        labels.append(label)
        colors.append(color)

        results_df = sweep_result["results_df"].copy()
        results_df.insert(0, "variant", label)
        sweep_rows.append(results_df)

        test_metrics_df = evaluate_sweep_results(
            results_df=sweep_result["results_df"],
            data_bundle=data_bundle,
            variant_label=label,
        )
        test_metric_frames.append(test_metrics_df)

        summary_rows.append(
            {
                "variant": label,
                "paper_use_rate_in_stress": use_rate,
                "selected_hidden": sweep_result["threshold_summary"]["selected_hidden"],
                "minimum_hidden": sweep_result["threshold_summary"]["minimum_hidden"],
                "best_hidden": sweep_result["threshold_summary"]["best_hidden"],
                "best_loss": sweep_result["threshold_summary"]["best_loss"],
                "threshold_limit": sweep_result["threshold_summary"]["threshold_limit"],
                "threshold_found": sweep_result["threshold_summary"]["threshold_found"],
                "figure_path": str(sweep_result["figure_path"]),
                "grouped_results_path": str(sweep_result["grouped_results_path"]),
                "results_path": str(sweep_result["results_path"]),
            }
        )

    grouped_all = pd.concat(grouped_frames, ignore_index=True)
    grouped_all_path = ARTIFACT_ROOT / "reports" / "09_paper_rno_hidden_sweep_grouped_results.csv"
    grouped_all.to_csv(grouped_all_path, index=False)

    raw_all = pd.concat(sweep_rows, ignore_index=True)
    raw_all_path = ARTIFACT_ROOT / "reports" / "09_paper_rno_hidden_sweep_raw_results.csv"
    raw_all.to_csv(raw_all_path, index=False)

    test_metrics_all = pd.concat(test_metric_frames, ignore_index=True)
    test_metrics_all_path = ARTIFACT_ROOT / "reports" / "09_paper_rno_hidden_sweep_test_metrics.csv"
    test_metrics_all.to_csv(test_metrics_all_path, index=False)

    validation_overlay_path = plot_validation_overlay(grouped_frames=grouped_frames, labels=labels, colors=colors)
    test_relative_l2_overlay_path = plot_test_relative_l2_overlay(
        test_metrics_df=test_metrics_all,
        colors_by_variant={str(item["label"]): str(item["color"]) for item in VARIANTS},
    )

    summary = {
        "device": DEVICE,
        "reference_family": REFERENCE_FAMILY,
        "batch_size": BATCH_SIZE,
        "min_epochs": MIN_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "hidden_grid": HIDDEN_GRID,
        "sweep_seeds": SWEEP_SEEDS,
        "grouped_results_csv": str(grouped_all_path),
        "raw_results_csv": str(raw_all_path),
        "test_metrics_csv": str(test_metrics_all_path),
        "validation_overlay_figure": str(validation_overlay_path),
        "test_relative_l2_overlay_figure": str(test_relative_l2_overlay_path),
        "variant_summaries": summary_rows,
    }
    summary_path = ARTIFACT_ROOT / "reports" / "09_paper_rno_hidden_sweep_summary.json"
    write_json(summary_path, summary)

    print(grouped_all.to_string(index=False))
    print(test_metrics_all.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
