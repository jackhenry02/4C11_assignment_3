from __future__ import annotations

"""Test the paper-style RNO at h=0 with and without direct strain-rate input."""

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
    train_model,
    write_json,
)


ARTIFACT_ROOT = Path("artifacts")
TRAIN_PATH = Path("Coursework3/viscodata_3mat.mat")
SPLIT_SEED = 20260328
DEVICE = "cpu"

REFERENCE_FAMILY = "gru"
REFERENCE_PARAMS_PATH = ARTIFACT_ROOT / "optuna" / f"best_params_{REFERENCE_FAMILY}.json"
PAPER_SEED = 20260341
BATCH_SIZE = 32
MIN_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 25
N_HIDDEN = 0


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


def build_paper_config(use_rate: bool, model_tag: str) -> ExperimentConfig:
    """Reuse the tuned GRU recipe, but swap in the paper-style core and h=0 setting."""
    payload = read_json(REFERENCE_PARAMS_PATH)
    config = ExperimentConfig(**json.loads(payload["best_user_attrs"]["config"]))
    config.CORE_TYPE = "paper_rno"
    config.PAPER_USE_RATE_IN_STRESS = bool(use_rate)
    config.N_HIDDEN = N_HIDDEN
    config.BATCH_SIZE = BATCH_SIZE
    config.MIN_EPOCHS = MIN_EPOCHS
    config.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE
    config.SEED = PAPER_SEED
    config.MODEL_TAG = model_tag
    config.VERBOSE = False
    return config


def run_variant(data_bundle: dict[str, Any], use_rate: bool) -> dict[str, Any]:
    """Train and evaluate one h=0 paper-RNO variant."""
    variant_key = "with_rate" if use_rate else "no_rate"
    run_name = f"08_paper_rno_h0_{variant_key}"
    config = build_paper_config(use_rate=use_rate, model_tag=run_name)
    print(
        f"[08] train run={run_name} core={config.CORE_TYPE} "
        f"use_rate={config.PAPER_USE_RATE_IN_STRESS} batch={config.BATCH_SIZE} "
        f"n_hidden={config.N_HIDDEN} seed={config.SEED}",
        flush=True,
    )
    train_result = train_model(
        data_bundle=data_bundle,
        config=config,
        artifact_root=ARTIFACT_ROOT,
        run_name=run_name,
    )
    evaluation = evaluate_checkpoint(
        checkpoint_path=train_result["checkpoint_path"],
        x=data_bundle["x_test"],
        y=data_bundle["y_test"],
        output_normalizer=data_bundle["output_normalizer"],
        artifact_root=ARTIFACT_ROOT,
        run_name=f"{run_name}_test",
        y_true0=data_bundle["y_test"][:, 0],
    )
    return {
        "variant_key": variant_key,
        "config": config,
        "train_result": train_result,
        "evaluation": evaluation,
    }


def plot_comparison(results_df: pd.DataFrame) -> Path:
    """Summarise how the two h=0 variants differ in loss, test error, and runtime."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    labels = ["no rate", "with rate"]
    colors = ["#1f77b4", "#d62728"]

    axes[0].bar(labels, results_df["best_val_loss"], color=colors, alpha=0.9)
    axes[0].set_title("Validation loss")
    axes[0].set_ylabel("Best validation loss")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, results_df["test_relative_l2"], color=colors, alpha=0.9)
    axes[1].set_title("Test relative L2")
    axes[1].set_ylabel("Relative L2")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(labels, results_df["total_seconds"], color=colors, alpha=0.9)
    axes[2].set_title("Training time")
    axes[2].set_ylabel("Seconds")
    axes[2].grid(axis="y", alpha=0.3)

    return save_figure(fig, ARTIFACT_ROOT / "figures" / "08_paper_rno_h0_comparison.png")


def main() -> None:
    """Run both h=0 paper-RNO variants and save a compact comparison table and figure."""
    os.environ["COURSEWORK_DEVICE"] = DEVICE
    data_bundle = prepare_data(train_path=TRAIN_PATH, artifact_root=ARTIFACT_ROOT, split_seed=SPLIT_SEED)

    rows: list[dict[str, Any]] = []
    for use_rate in [False, True]:
        payload = run_variant(data_bundle=data_bundle, use_rate=use_rate)
        config = payload["config"]
        train_result = payload["train_result"]
        evaluation = payload["evaluation"]
        rows.append(
            {
                "variant": "paper_rno_with_rate" if use_rate else "paper_rno_no_rate",
                "paper_use_rate_in_stress": bool(use_rate),
                "batch_size": int(config.BATCH_SIZE),
                "min_epochs": int(config.MIN_EPOCHS),
                "early_stopping_patience": int(config.EARLY_STOPPING_PATIENCE),
                "n_hidden": int(config.N_HIDDEN),
                "seed": int(config.SEED),
                "best_val_loss": float(train_result["best_val_loss"]),
                "best_epoch": int(train_result["best_epoch"]),
                "n_epochs_completed": int(train_result["n_epochs_completed"]),
                "total_seconds": float(train_result["total_seconds"]),
                "test_rmse": float(evaluation["metrics"]["rmse"]),
                "test_mae": float(evaluation["metrics"]["mae"]),
                "test_r2": float(evaluation["metrics"]["r2"]),
                "test_nrmse": float(evaluation["metrics"]["nrmse"]),
                "test_relative_l2": float(evaluation["metrics"]["relative_l2"]),
                "checkpoint_path": str(train_result["checkpoint_path"]),
                "history_path": str(train_result["history_path"]),
                "summary_path": str(train_result["summary_path"]),
            }
        )

    results_df = pd.DataFrame(rows).sort_values("paper_use_rate_in_stress").reset_index(drop=True)
    comparison_path = ARTIFACT_ROOT / "reports" / "08_paper_rno_h0_comparison.csv"
    results_df.to_csv(comparison_path, index=False)
    figure_path = plot_comparison(results_df=results_df)

    summary = {
        "device": DEVICE,
        "reference_family": REFERENCE_FAMILY,
        "comparison_csv": str(comparison_path),
        "comparison_figure": str(figure_path),
        "batch_size": BATCH_SIZE,
        "min_epochs": MIN_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "n_hidden": N_HIDDEN,
        "seed": PAPER_SEED,
    }
    summary_path = ARTIFACT_ROOT / "reports" / "08_paper_rno_h0_comparison_summary.json"
    write_json(summary_path, summary)

    print(results_df.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
