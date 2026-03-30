from __future__ import annotations

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
    run_hidden_threshold_grid_progressive,
    train_model,
    write_json,
)


ARTIFACT_ROOT = Path("artifacts")
TRAIN_PATH = Path("Coursework3/viscodata_3mat.mat")
SPLIT_SEED = 20260328
DEVICE = "cpu"

REFERENCE_FAMILY = "gru"
BASELINE_CORE = "baseline_rno"
REFERENCE_SEED = 20260340
NEW_BATCH_SIZE = 32
EXISTING_BATCH8_SUMMARY = ARTIFACT_ROOT / "logs" / "baseline_rno_from_best_gru_summary.json"
EXISTING_BATCH8_CONFIG = ARTIFACT_ROOT / "logs" / "baseline_rno_from_best_gru_config.json"
ACCEL_MIN_EPOCHS = 50
ACCEL_EARLY_STOPPING_PATIENCE = 25

HIDDEN_GRID = [0, 1, 2, 3, 4]
SWEEP_SEEDS = [20260328, 20260329, 20260330]
SWEEP_TOLERANCE_RATIO = 0.05
SWEEP_RUN_PREFIX = "07_baseline_rno_batch32_accel_hidden_grid_0to4"


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


def best_payload_for_family(family: str) -> dict[str, Any]:
    return read_json(ARTIFACT_ROOT / "optuna" / f"best_params_{family}.json")


def build_baseline_config(batch_size: int, seed: int, model_tag: str) -> ExperimentConfig:
    reference_payload = best_payload_for_family(REFERENCE_FAMILY)
    reference_config = ExperimentConfig(**json.loads(reference_payload["best_user_attrs"]["config"]))
    reference_config.CORE_TYPE = BASELINE_CORE
    reference_config.BATCH_SIZE = int(batch_size)
    reference_config.MIN_EPOCHS = ACCEL_MIN_EPOCHS
    reference_config.EARLY_STOPPING_PATIENCE = ACCEL_EARLY_STOPPING_PATIENCE
    reference_config.SEED = int(seed)
    reference_config.MODEL_TAG = model_tag
    reference_config.VERBOSE = False
    return reference_config


def train_and_evaluate(run_name: str, config: ExperimentConfig, data_bundle: dict[str, Any]) -> dict[str, Any]:
    print(
        f"[07] train run={run_name} core={config.CORE_TYPE} "
        f"batch={config.BATCH_SIZE} n_hidden={config.N_HIDDEN} seed={config.SEED}",
        flush=True,
    )
    train_result = train_model(
        data_bundle=data_bundle,
        config=config,
        artifact_root=ARTIFACT_ROOT,
        run_name=run_name,
    )
    checkpoint_path = Path(train_result["checkpoint_path"])
    evaluation = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        x=data_bundle["x_test"],
        y=data_bundle["y_test"],
        output_normalizer=data_bundle["output_normalizer"],
        artifact_root=ARTIFACT_ROOT,
        run_name=f"{run_name}_test",
        y_true0=data_bundle["y_test"][:, 0],
    )
    return {"train_result": train_result, "evaluation": evaluation, "config": config}


def evaluate_existing_batch8(data_bundle: dict[str, Any]) -> dict[str, Any]:
    summary = read_json(EXISTING_BATCH8_SUMMARY)
    config_payload = read_json(EXISTING_BATCH8_CONFIG)
    config = ExperimentConfig(**config_payload)
    evaluation = evaluate_checkpoint(
        checkpoint_path=summary["checkpoint_path"],
        x=data_bundle["x_test"],
        y=data_bundle["y_test"],
        output_normalizer=data_bundle["output_normalizer"],
        artifact_root=ARTIFACT_ROOT,
        run_name="baseline_rno_from_best_gru_existing_batch8_test",
        y_true0=data_bundle["y_test"][:, 0],
    )
    return {
        "train_result": summary,
        "evaluation": evaluation,
        "config": config,
    }


def plot_batch_comparison_validation_curves(history_paths: list[tuple[str, Path]]) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = {"batch8": "#1f77b4", "batch32": "#d62728"}
    for label, history_path in history_paths:
        history = pd.read_csv(history_path)
        ax.plot(
            history["epoch"],
            history["val_loss"],
            label=label,
            linewidth=2.0,
            color=colors["batch32" if "32" in label else "batch8"],
        )
    ax.set_title("Baseline RNO validation curves: batch size 8 vs 32")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    return save_figure(fig, ARTIFACT_ROOT / "figures" / "07_baseline_rno_batchsize_validation_curves.png")


def plot_batch_comparison_summary(comparison_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    labels = [f"batch {value}" for value in comparison_df["batch_size"].tolist()]
    colors = ["#1f77b4", "#d62728"]

    runtime_df = comparison_df.dropna(subset=["total_seconds"]).copy()
    if runtime_df.empty:
        axes[0].text(0.5, 0.5, "No runtime data available", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_xticks([])
    else:
        runtime_labels = [f"batch {value}" for value in runtime_df["batch_size"].tolist()]
        runtime_colors = colors[: len(runtime_df)]
        axes[0].bar(runtime_labels, runtime_df["total_seconds"], color=runtime_colors, alpha=0.9)
        if len(runtime_df) < len(comparison_df):
            axes[0].text(
                0.5,
                0.92,
                "batch 8 runtime unavailable in existing baseline logs",
                ha="center",
                va="top",
                fontsize=9,
                transform=axes[0].transAxes,
            )
    axes[0].set_title("Baseline RNO runtime")
    axes[0].set_ylabel("Training time (s)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, comparison_df["test_relative_l2"], color=colors, alpha=0.9)
    axes[1].set_title("Baseline RNO test relative L2")
    axes[1].set_ylabel("Relative L2 error")
    axes[1].grid(axis="y", alpha=0.3)

    return save_figure(fig, ARTIFACT_ROOT / "figures" / "07_baseline_rno_batchsize_runtime_relative_l2.png")


def plot_hidden_sweep_validation_overlay(run_prefix: str, results_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    grouped = (
        results_df.sort_values(["n_hidden", "best_val_loss"])
        .groupby("n_hidden", as_index=False)
        .first()
    )
    cmap = plt.get_cmap("viridis")
    hidden_values = grouped["n_hidden"].tolist()
    if len(hidden_values) == 1:
        color_positions = [0.55]
    else:
        color_positions = [index / (len(hidden_values) - 1) for index in range(len(hidden_values))]

    for color_position, (_, row) in zip(color_positions, grouped.iterrows()):
        history_path = Path(row["checkpoint_path"]).with_name(Path(row["checkpoint_path"]).stem + "_history.csv")
        history_path = ARTIFACT_ROOT / "logs" / history_path.name
        if not history_path.exists():
            continue
        history = pd.read_csv(history_path)
        ax.plot(
            history["epoch"],
            history["val_loss"],
            linewidth=2.0,
            color=cmap(color_position),
            label=f"h={int(row['n_hidden'])}",
        )

    ax.set_title("Baseline RNO batch-32 validation curves by hidden size")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    return save_figure(fig, ARTIFACT_ROOT / "figures" / f"{run_prefix}_validation_overlay.png")


def main() -> None:
    os.environ["COURSEWORK_DEVICE"] = DEVICE
    data_bundle = prepare_data(train_path=TRAIN_PATH, artifact_root=ARTIFACT_ROOT, split_seed=SPLIT_SEED)

    batch_rows: list[dict[str, Any]] = []
    history_paths: list[tuple[str, Path]] = []
    existing_batch8 = evaluate_existing_batch8(data_bundle=data_bundle)
    existing_train = existing_batch8["train_result"]
    existing_eval = existing_batch8["evaluation"]
    existing_config = existing_batch8["config"]
    history_paths.append((f"batch {existing_config.BATCH_SIZE}", Path(existing_train["history_path"])))
    batch_rows.append(
        {
            "batch_size": int(existing_config.BATCH_SIZE),
            "core_type": BASELINE_CORE,
            "seed": int(existing_config.SEED),
            "n_hidden": int(existing_config.N_HIDDEN),
            "best_val_loss": float(existing_train["best_val_loss"]),
            "best_epoch": int(existing_train["best_epoch"]),
            "total_seconds": float(existing_train["total_seconds"]) if "total_seconds" in existing_train else None,
            "test_rmse": float(existing_eval["metrics"]["rmse"]),
            "test_mae": float(existing_eval["metrics"]["mae"]),
            "test_r2": float(existing_eval["metrics"]["r2"]),
            "test_nrmse": float(existing_eval["metrics"]["nrmse"]),
            "test_relative_l2": float(existing_eval["metrics"]["relative_l2"]),
            "checkpoint_path": str(existing_train["checkpoint_path"]),
            "history_path": str(existing_train["history_path"]),
            "summary_path": str(EXISTING_BATCH8_SUMMARY),
            "source": "existing_batch8_baseline",
        }
    )

    batch_size = NEW_BATCH_SIZE
    run_name = f"baseline_rno_best_gru_batch{batch_size}_accel_replay"
    config = build_baseline_config(
        batch_size=batch_size,
        seed=REFERENCE_SEED,
        model_tag=run_name,
    )
    payload = train_and_evaluate(run_name=run_name, config=config, data_bundle=data_bundle)
    train_result = payload["train_result"]
    evaluation = payload["evaluation"]
    history_paths.append((f"batch {batch_size}", Path(train_result["history_path"])))
    batch_rows.append(
        {
            "batch_size": batch_size,
            "core_type": BASELINE_CORE,
            "seed": config.SEED,
            "n_hidden": config.N_HIDDEN,
            "best_val_loss": float(train_result["best_val_loss"]),
            "best_epoch": int(train_result["best_epoch"]),
            "total_seconds": float(train_result["total_seconds"]),
            "test_rmse": float(evaluation["metrics"]["rmse"]),
            "test_mae": float(evaluation["metrics"]["mae"]),
            "test_r2": float(evaluation["metrics"]["r2"]),
            "test_nrmse": float(evaluation["metrics"]["nrmse"]),
            "test_relative_l2": float(evaluation["metrics"]["relative_l2"]),
            "checkpoint_path": str(train_result["checkpoint_path"]),
            "history_path": str(train_result["history_path"]),
            "summary_path": str(train_result["summary_path"]),
            "source": "new_batch32_replay",
        }
    )

    batch_comparison_df = pd.DataFrame(batch_rows).sort_values("batch_size").reset_index(drop=True)
    batch_comparison_path = ARTIFACT_ROOT / "reports" / "07_baseline_rno_batchsize_comparison.csv"
    batch_comparison_df.to_csv(batch_comparison_path, index=False)

    batch_summary = {
        "device": DEVICE,
        "reference_family": REFERENCE_FAMILY,
        "baseline_core": BASELINE_CORE,
        "reference_seed": REFERENCE_SEED,
        "comparison_csv": str(batch_comparison_path),
        "validation_curve_plot": str(plot_batch_comparison_validation_curves(history_paths=history_paths)),
        "runtime_relative_l2_plot": str(plot_batch_comparison_summary(comparison_df=batch_comparison_df)),
        "speedup_batch32_over_batch8": (
            float(
                batch_comparison_df.loc[batch_comparison_df["batch_size"] == 8, "total_seconds"].iloc[0]
                / batch_comparison_df.loc[batch_comparison_df["batch_size"] == 32, "total_seconds"].iloc[0]
            )
            if pd.notna(batch_comparison_df.loc[batch_comparison_df["batch_size"] == 8, "total_seconds"].iloc[0])
            else None
        ),
        "relative_l2_ratio_batch32_over_batch8": float(
            batch_comparison_df.loc[batch_comparison_df["batch_size"] == 32, "test_relative_l2"].iloc[0]
            / batch_comparison_df.loc[batch_comparison_df["batch_size"] == 8, "test_relative_l2"].iloc[0]
        ),
    }
    batch_summary_path = ARTIFACT_ROOT / "reports" / "07_baseline_rno_batchsize_comparison_summary.json"
    write_json(batch_summary_path, batch_summary)

    batch32_config = build_baseline_config(
        batch_size=32,
        seed=SWEEP_SEEDS[0],
        model_tag=f"{SWEEP_RUN_PREFIX}_template",
    )
    print(
        f"[07] start hidden sweep core={batch32_config.CORE_TYPE} "
        f"batch={batch32_config.BATCH_SIZE} hidden_grid={HIDDEN_GRID} seeds={SWEEP_SEEDS}",
        flush=True,
    )
    sweep_result = run_hidden_threshold_grid_progressive(
        data_bundle=data_bundle,
        base_config=batch32_config,
        hidden_grid=HIDDEN_GRID,
        seeds=SWEEP_SEEDS,
        artifact_root=ARTIFACT_ROOT,
        run_prefix=SWEEP_RUN_PREFIX,
        tolerance_ratio=SWEEP_TOLERANCE_RATIO,
        reference_loss=None,
        verbose=True,
    )
    validation_overlay_path = plot_hidden_sweep_validation_overlay(
        run_prefix=SWEEP_RUN_PREFIX,
        results_df=sweep_result["results_df"],
    )

    sweep_summary = {
        "device": DEVICE,
        "reference_family": REFERENCE_FAMILY,
        "baseline_core": BASELINE_CORE,
        "batch_size": 32,
        "hidden_grid": HIDDEN_GRID,
        "sweep_seeds": SWEEP_SEEDS,
        "results_path": str(sweep_result["results_path"]),
        "trace_path": str(sweep_result["trace_path"]),
        "grouped_results_path": str(sweep_result["grouped_results_path"]),
        "threshold_json_path": str(sweep_result["threshold_json_path"]),
        "seed_progression_path": str(sweep_result["stage_summary_path"]),
        "threshold_figure_path": str(sweep_result["figure_path"]),
        "validation_overlay_path": str(validation_overlay_path),
        "selected_hidden": sweep_result["threshold_summary"]["selected_hidden"],
        "best_hidden": sweep_result["threshold_summary"]["best_hidden"],
        "threshold_found": sweep_result["threshold_summary"]["threshold_found"],
    }
    sweep_summary_path = ARTIFACT_ROOT / "reports" / "07_baseline_rno_batch32_hidden_sweep_summary.json"
    write_json(sweep_summary_path, sweep_summary)

    print(batch_comparison_df.to_string(index=False))
    print(json.dumps(batch_summary, indent=2))
    print(json.dumps(sweep_summary, indent=2))


if __name__ == "__main__":
    main()
