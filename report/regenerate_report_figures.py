from __future__ import annotations

"""Regenerate report-only figure copies with larger fonts from saved outputs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Coursework3.RNO_1D_Skeleton import (
    plot_hidden_threshold,
    plot_sample_histories,
    prepare_data,
    read_json,
    run_hidden_state_analysis,
    run_inference_and_testing,
    run_trajectory_and_hysteresis_analysis,
)


REPORT_ROOT = Path(__file__).resolve().parent
FIGURE_DIR = REPORT_ROOT / "figures"
OPTUNA_FIGURE_DIR = FIGURE_DIR / "optuna"
ARTIFACT_ROOT = ROOT / "artifacts"

BEST_GRU_CHECKPOINT = ARTIFACT_ROOT / "checkpoints" / "optuna_gru_trial_0012.pt"


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def apply_report_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 16,
            "lines.linewidth": 2.2,
            "axes.grid": False,
        }
    )


def ensure_directories() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def regenerate_sample_histories() -> Path:
    data_bundle = prepare_data(train_path=ROOT / "Coursework3" / "viscodata_3mat.mat", artifact_root=REPORT_ROOT, split_seed=20260328)
    return plot_sample_histories(
        data_input_raw=data_bundle["data_input_raw"],
        data_output_raw=data_bundle["data_output_raw"],
        save_path=FIGURE_DIR / "01_sample_histories.png",
        sample_count=6,
    )


def regenerate_optuna_figures() -> list[Path]:
    optuna_module = import_module_from_path("optuna_report_plots_local", ROOT / "02b_optuna_report_plots.py")
    train_module = import_module_from_path("training_curve_plots_local", ROOT / "make_training_curve_plots.py")

    optuna_module.ARTIFACT_ROOT = ARTIFACT_ROOT
    optuna_module.OPTUNA_DIR = ARTIFACT_ROOT / "optuna"
    optuna_module.FIGURE_DIR = OPTUNA_FIGURE_DIR
    optuna_module.TABLE_DIR = REPORT_ROOT / "optuna"
    optuna_module.ensure_directory(optuna_module.FIGURE_DIR)

    family_payloads = [optuna_module.load_family_data(family) for family in optuna_module.FAMILIES]
    family_summary_df = pd.DataFrame(
        [
            {
                "family": payload["family"],
                "best_value": payload["summary"]["best_value"],
                "best_trial_elapsed_seconds": payload["summary"]["best_trial_elapsed_seconds"],
            }
            for payload in family_payloads
        ]
    )
    all_trials_df = pd.concat([payload["merged_df"] for payload in family_payloads], ignore_index=True)

    paths = [
        optuna_module.make_family_best_value_bar(family_summary_df),
        optuna_module.make_family_best_duration_bar(family_summary_df),
        optuna_module.make_family_value_distribution(all_trials_df),
    ]

    gru_payload = next(payload for payload in family_payloads if payload["family"] == "gru")
    paths.extend(
        [
            optuna_module.make_optimization_history("gru", gru_payload["trials_df"]),
            optuna_module.make_hidden_loss_scatter("gru", gru_payload["merged_df"]),
            optuna_module.make_param_importance_plot("gru", gru_payload["study"]),
        ]
    )

    train_module.ARTIFACT_ROOT = ARTIFACT_ROOT
    train_module.OPTUNA_DIR = ARTIFACT_ROOT / "optuna"
    train_module.LOG_DIR = ARTIFACT_ROOT / "logs"
    train_module.FIGURE_DIR = FIGURE_DIR
    train_module.OPTUNA_FIGURE_DIR = OPTUNA_FIGURE_DIR
    train_module.HIDDEN_FIGURE_DIR = FIGURE_DIR
    train_module.ensure_directory(OPTUNA_FIGURE_DIR)
    paths.extend(train_module.plot_best_optuna_curves())
    return paths


def regenerate_inference_figures() -> dict[str, Path]:
    run_inference_and_testing(
        checkpoint_path=BEST_GRU_CHECKPOINT,
        artifact_root=REPORT_ROOT,
        split_seed=20260328,
    )
    run_trajectory_and_hysteresis_analysis(
        checkpoint_path=BEST_GRU_CHECKPOINT,
        artifact_root=REPORT_ROOT,
        split_seed=20260328,
    )
    return {
        "residual": FIGURE_DIR / "04_residual_analysis.png",
        "stress_strain": FIGURE_DIR / "05_test_stress_strain_examples.png",
        "stress_time": FIGURE_DIR / "05_test_stress_time_examples.png",
        "hysteresis": FIGURE_DIR / "05_hysteresis_checks.png",
    }


def regenerate_hidden_threshold_figures() -> list[Path]:
    paths: list[Path] = []

    low_dim_results = pd.read_csv(ARTIFACT_ROOT / "final" / "hidden_threshold_low_dim_with_reference_results.csv")
    low_dim_threshold = read_json(ARTIFACT_ROOT / "final" / "hidden_threshold_low_dim_with_reference_threshold.json")
    grouped = pd.read_csv(ARTIFACT_ROOT / "final" / "hidden_threshold_low_dim_with_reference_grouped_results.csv")

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    raw_results = low_dim_results.loc[:, ["n_hidden", "best_val_loss"]].copy()
    jitter_rng = np.random.default_rng(0)
    jitter = jitter_rng.uniform(-0.3, 0.3, size=len(raw_results))
    ax.scatter(
        raw_results["n_hidden"] + jitter,
        raw_results["best_val_loss"],
        alpha=0.35,
        s=28,
        color="#7f7f7f",
        label="individual seed runs",
    )
    ax.errorbar(
        grouped["n_hidden"],
        grouped["mean_val_loss"],
        yerr=grouped["std_val_loss"].fillna(0.0),
        marker="o",
        capsize=4,
        color="#1f77b4",
        label="mean across seeds",
    )
    ax.axhline(low_dim_threshold["threshold_limit"], color="#d62728", linestyle="--", label="5% target threshold")
    ax.axvline(low_dim_threshold["selected_hidden"], color="#ff7f0e", linestyle=":", label="best tested hidden")
    ax.set_title("Hidden-state threshold analysis")
    ax.set_xlabel("n_hidden")
    ax.set_ylabel("Mean validation loss")
    ax.set_xticks(grouped["n_hidden"].tolist())
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    low_dim_path = FIGURE_DIR / "hidden_threshold_low_dim_with_reference_threshold.png"
    fig.tight_layout()
    fig.savefig(low_dim_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(low_dim_path)

    paper_no_rate_results = pd.read_csv(ARTIFACT_ROOT / "final" / "09_paper_rno_no_rate_h0to5_results.csv")
    paper_with_rate_results = pd.read_csv(ARTIFACT_ROOT / "final" / "09_paper_rno_with_rate_h0to5_results.csv")
    no_rate_path, _ = plot_hidden_threshold(
        results_df=paper_no_rate_results,
        save_path=FIGURE_DIR / "09_paper_rno_no_rate_h0to5_threshold.png",
        tolerance_ratio=0.05,
        reference_loss=None,
        require_plateau=None,
    )
    with_rate_path, _ = plot_hidden_threshold(
        results_df=paper_with_rate_results,
        save_path=FIGURE_DIR / "09_paper_rno_with_rate_h0to5_threshold.png",
        tolerance_ratio=0.05,
        reference_loss=None,
        require_plateau=None,
    )
    paths.extend([no_rate_path, with_rate_path])

    grouped_results = pd.read_csv(ARTIFACT_ROOT / "reports" / "09_paper_rno_hidden_sweep_grouped_results.csv")
    test_metrics = pd.read_csv(ARTIFACT_ROOT / "reports" / "09_paper_rno_hidden_sweep_test_metrics.csv")
    colors = {"paper_rno_no_rate": "#1f77b4", "paper_rno_with_rate": "#d62728"}

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for variant, subset in grouped_results.groupby("variant"):
        subset = subset.sort_values("n_hidden")
        ax.plot(subset["n_hidden"], subset["mean_val_loss"], marker="o", label=variant, color=colors[variant])
    ax.set_title("paper_rno validation loss vs hidden size")
    ax.set_xlabel("n_hidden")
    ax.set_ylabel("Mean validation loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    val_path = FIGURE_DIR / "09_paper_rno_validation_loss_vs_hidden.png"
    fig.tight_layout()
    fig.savefig(val_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(val_path)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    grouped_test = (
        test_metrics.groupby(["variant", "n_hidden"], as_index=False)
        .agg(mean_test_relative_l2=("test_relative_l2", "mean"))
        .sort_values(["variant", "n_hidden"])
    )
    for variant, subset in grouped_test.groupby("variant"):
        ax.plot(subset["n_hidden"], subset["mean_test_relative_l2"], marker="o", label=variant, color=colors[variant])
    ax.set_title("paper_rno test relative L2 vs hidden size")
    ax.set_xlabel("n_hidden")
    ax.set_ylabel("Test relative L2")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    test_path = FIGURE_DIR / "09_paper_rno_test_relative_l2_vs_hidden.png"
    fig.tight_layout()
    fig.savefig(test_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(test_path)

    comparison_df = pd.read_csv(ARTIFACT_ROOT / "reports" / "08_paper_rno_h0_comparison.csv")
    labels = ["no rate", "with rate"]
    colors_bar = ["#1f77b4", "#d62728"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
    axes[0].bar(labels, comparison_df["best_val_loss"], color=colors_bar, alpha=0.9)
    axes[0].set_title("Validation loss")
    axes[0].set_ylabel("Best validation loss")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(labels, comparison_df["test_relative_l2"], color=colors_bar, alpha=0.9)
    axes[1].set_title("Test relative L2")
    axes[1].set_ylabel("Relative L2")
    axes[1].grid(axis="y", alpha=0.3)
    axes[2].bar(labels, comparison_df["total_seconds"], color=colors_bar, alpha=0.9)
    axes[2].set_title("Training time")
    axes[2].set_ylabel("Seconds")
    axes[2].grid(axis="y", alpha=0.3)
    ablation_path = FIGURE_DIR / "08_paper_rno_h0_comparison.png"
    fig.tight_layout()
    fig.savefig(ablation_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(ablation_path)

    return paths


def regenerate_hidden_state_figures() -> dict[str, Path]:
    model_specs = pd.read_csv(ARTIFACT_ROOT / "reports" / "11_hidden_state_model_specs.csv").to_dict("records")
    result = run_hidden_state_analysis(
        model_specs=model_specs,
        artifact_root=REPORT_ROOT,
        split_seed=20260328,
        preferred_sample_class="cyclic_or_reversing",
    )
    return {
        "correlation": Path(result["correlation_plot_path"]),
        "pca_variance": Path(result["pca_variance_plot_path"]),
        "pca_trajectories": Path(result["pca_trajectory_plot_path"]),
        "time_traces": Path(result["time_trace_plot_path"]),
    }


def regenerate_validation_curve_comparison() -> Path:
    comparison_df = pd.read_csv(ARTIFACT_ROOT / "reports" / "06_baseline_rno_comparison.csv")
    history_paths = {
        "GRU": ARTIFACT_ROOT / "logs" / "optuna_gru_trial_0012_history.csv",
        "Baseline RNO": ARTIFACT_ROOT / "logs" / "baseline_rno_from_best_gru_history.csv",
        "RNN": ARTIFACT_ROOT / "logs" / "optuna_rnn_trial_0011_history.csv",
        "LSTM": ARTIFACT_ROOT / "logs" / "optuna_lstm_trial_0019_history.csv",
    }
    colors = {"GRU": "#2ca02c", "Baseline RNO": "#9467bd", "RNN": "#1f77b4", "LSTM": "#d62728"}
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for label, path in history_paths.items():
        history = pd.read_csv(path)
        ax.plot(history["epoch"], history["val_loss"], label=label, color=colors[label])
    ax.set_title("Validation curves for the best trained models")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    save_path = FIGURE_DIR / "06_best_models_validation_curve_comparison.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main() -> None:
    apply_report_style()
    ensure_directories()

    generated = {
        "sample_histories": str(regenerate_sample_histories()),
        "optuna_figures": [str(path) for path in regenerate_optuna_figures()],
        "inference_figures": {key: str(path) for key, path in regenerate_inference_figures().items()},
        "hidden_threshold_figures": [str(path) for path in regenerate_hidden_threshold_figures()],
        "hidden_state_figures": {key: str(path) for key, path in regenerate_hidden_state_figures().items()},
        "validation_curve_comparison": str(regenerate_validation_curve_comparison()),
    }

    summary_path = REPORT_ROOT / "report_figure_style_summary.json"
    summary_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    print(json.dumps(generated, indent=2))


if __name__ == "__main__":
    main()
