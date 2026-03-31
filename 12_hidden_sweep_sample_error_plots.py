from __future__ import annotations

"""Plot average and worst single-case test errors against hidden size for the completed sweeps."""

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts") / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Coursework3.RNO_1D_Skeleton import (
    evaluate_checkpoint,
    prepare_data,
    summarize_sample_prediction_metrics,
    write_json,
)


ARTIFACT_ROOT = Path("artifacts")
TRAIN_PATH = Path("Coursework3/viscodata_3mat.mat")
SPLIT_SEED = 20260328
DEVICE = "cpu"
TOLERANCE_RATIO = 0.05

SWEEPS = [
    {
        "label": "gru_low_dim",
        "display_name": "GRU low-dimensional sweep",
        "results_csv": ARTIFACT_ROOT / "final" / "hidden_threshold_low_dim_with_reference_results.csv",
        "color_mean": "#1f77b4",
        "color_worst": "#0d3b66",
    },
    {
        "label": "paper_rno_no_rate",
        "display_name": "paper RNO without rate",
        "results_csv": ARTIFACT_ROOT / "final" / "09_paper_rno_no_rate_h0to5_results.csv",
        "color_mean": "#d62728",
        "color_worst": "#7f1d1d",
    },
    {
        "label": "paper_rno_with_rate",
        "display_name": "paper RNO with rate",
        "results_csv": ARTIFACT_ROOT / "final" / "09_paper_rno_with_rate_h0to5_results.csv",
        "color_mean": "#2ca02c",
        "color_worst": "#14532d",
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


def prediction_path_for_run(run_name: str) -> Path:
    return ARTIFACT_ROOT / "predictions" / f"{run_name}_predictions.npz"


def get_prediction_path(
    checkpoint_path: str | Path,
    data_bundle: dict[str, Any],
    run_name: str,
) -> Path:
    """Reuse an existing prediction file when present, otherwise evaluate the checkpoint once."""
    prediction_path = prediction_path_for_run(run_name)
    if prediction_path.exists():
        return prediction_path
    evaluation = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        x=data_bundle["x_test"],
        y=data_bundle["y_test"],
        output_normalizer=data_bundle["output_normalizer"],
        artifact_root=ARTIFACT_ROOT,
        run_name=run_name,
        y_true0=data_bundle["y_test"][:, 0],
    )
    return Path(evaluation["prediction_path"])


def summarise_sweep(
    sweep: dict[str, Any],
    data_bundle: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute per-checkpoint average and worst single-case errors for one hidden-size sweep."""
    results_df = pd.read_csv(sweep["results_csv"]).sort_values(["n_hidden", "seed"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    for row in results_df.itertuples(index=False):
        n_hidden = int(row.n_hidden)
        seed = int(row.seed)
        run_name = f"12_{sweep['label']}_h{n_hidden}_seed_{seed}"
        prediction_path = get_prediction_path(
            checkpoint_path=row.checkpoint_path,
            data_bundle=data_bundle,
            run_name=run_name,
        )
        payload = np.load(prediction_path)
        sample_metrics = summarize_sample_prediction_metrics(
            y_true_raw=payload["y_true_raw"],
            y_pred_raw=payload["y_pred_raw"],
        )
        worst_row = sample_metrics.loc[sample_metrics["sample_relative_l2"].idxmax()]
        rows.append(
            {
                "sweep_label": sweep["label"],
                "sweep_display_name": sweep["display_name"],
                "n_hidden": n_hidden,
                "seed": seed,
                "best_val_loss": float(row.best_val_loss),
                "best_epoch": int(row.best_epoch),
                "checkpoint_path": str(row.checkpoint_path),
                "prediction_path": str(prediction_path),
                "mean_sample_relative_l2": float(sample_metrics["sample_relative_l2"].mean()),
                "median_sample_relative_l2": float(sample_metrics["sample_relative_l2"].median()),
                "worst_sample_relative_l2": float(worst_row["sample_relative_l2"]),
                "worst_sample_rmse": float(worst_row["sample_rmse"]),
                "worst_sample_index": int(worst_row["sample_index"]),
            }
        )

    per_run_df = pd.DataFrame(rows).sort_values(["n_hidden", "seed"]).reset_index(drop=True)
    grouped_df = (
        per_run_df.groupby(["sweep_label", "sweep_display_name", "n_hidden"], as_index=False)
        .agg(
            mean_of_mean_sample_relative_l2=("mean_sample_relative_l2", "mean"),
            mean_of_median_sample_relative_l2=("median_sample_relative_l2", "mean"),
            mean_of_worst_sample_relative_l2=("worst_sample_relative_l2", "mean"),
            std_of_worst_sample_relative_l2=("worst_sample_relative_l2", "std"),
            n_runs=("seed", "count"),
        )
        .sort_values("n_hidden")
        .reset_index(drop=True)
    )

    best_worst = float(grouped_df["mean_of_worst_sample_relative_l2"].min())
    worst_threshold_limit = (1.0 + TOLERANCE_RATIO) * best_worst
    acceptable = grouped_df.loc[
        grouped_df["mean_of_worst_sample_relative_l2"] <= worst_threshold_limit, "n_hidden"
    ].tolist()
    selected_hidden = int(min(acceptable)) if acceptable else None

    summary = {
        "sweep_label": sweep["label"],
        "sweep_display_name": sweep["display_name"],
        "results_csv": str(sweep["results_csv"]),
        "best_worst_sample_relative_l2": best_worst,
        "worst_threshold_limit": worst_threshold_limit,
        "selected_hidden_by_worst_case": selected_hidden,
        "acceptable_hidden_by_worst_case": [int(value) for value in acceptable],
        "tolerance_ratio": TOLERANCE_RATIO,
    }
    return per_run_df, grouped_df, summary


def plot_grouped_summary(grouped_df: pd.DataFrame, summaries: list[dict[str, Any]]) -> Path:
    """Make one subplot per sweep with both average and worst-case test errors."""
    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(15.0, 4.8), sharey=False)
    if len(SWEEPS) == 1:
        axes = [axes]

    summary_by_label = {item["sweep_label"]: item for item in summaries}

    for ax, sweep in zip(axes, SWEEPS):
        subset = grouped_df.loc[grouped_df["sweep_label"] == sweep["label"]].sort_values("n_hidden")
        summary = summary_by_label[sweep["label"]]

        ax.plot(
            subset["n_hidden"],
            subset["mean_of_mean_sample_relative_l2"],
            marker="o",
            linewidth=2.2,
            color=sweep["color_mean"],
            label="mean test-case rel. $L^2$",
        )
        ax.plot(
            subset["n_hidden"],
            subset["mean_of_worst_sample_relative_l2"],
            marker="s",
            linewidth=2.2,
            color=sweep["color_worst"],
            label="worst single test-case rel. $L^2$",
        )

        if summary["selected_hidden_by_worst_case"] is not None:
            ax.axvline(
                summary["selected_hidden_by_worst_case"],
                color="black",
                linestyle="--",
                linewidth=1.3,
                alpha=0.65,
            )

        ax.set_title(
            f"{sweep['display_name']}\nworst-case 5% select: "
            f"{summary['selected_hidden_by_worst_case']}"
        )
        ax.set_xlabel("n_hidden")
        ax.set_ylabel("Relative $L^2$")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.set_xticks(sorted(subset["n_hidden"].unique()))
        ax.legend(fontsize=8, loc="best")

    return save_figure(fig, ARTIFACT_ROOT / "figures" / "12_hidden_sweep_mean_vs_worst_test_case_errors.png")


def main() -> None:
    """Compute per-sample hidden-sweep test errors and save the comparison plots."""
    os.environ["COURSEWORK_DEVICE"] = DEVICE
    data_bundle = prepare_data(train_path=TRAIN_PATH, artifact_root=ARTIFACT_ROOT, split_seed=SPLIT_SEED)

    per_run_frames: list[pd.DataFrame] = []
    grouped_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for sweep in SWEEPS:
        print(
            f"[12] summarise sweep={sweep['label']} results_csv={sweep['results_csv']}",
            flush=True,
        )
        per_run_df, grouped_df, summary = summarise_sweep(sweep=sweep, data_bundle=data_bundle)
        per_run_frames.append(per_run_df)
        grouped_frames.append(grouped_df)
        summaries.append(summary)

    per_run_all = pd.concat(per_run_frames, ignore_index=True)
    grouped_all = pd.concat(grouped_frames, ignore_index=True)

    per_run_csv = ARTIFACT_ROOT / "reports" / "12_hidden_sweep_sample_error_per_run.csv"
    grouped_csv = ARTIFACT_ROOT / "reports" / "12_hidden_sweep_sample_error_grouped.csv"
    summary_json = ARTIFACT_ROOT / "reports" / "12_hidden_sweep_sample_error_summary.json"

    per_run_all.to_csv(per_run_csv, index=False)
    grouped_all.to_csv(grouped_csv, index=False)
    figure_path = plot_grouped_summary(grouped_df=grouped_all, summaries=summaries)

    write_json(
        summary_json,
        {
            "device": DEVICE,
            "per_run_csv": str(per_run_csv),
            "grouped_csv": str(grouped_csv),
            "figure_path": str(figure_path),
            "sweep_summaries": summaries,
        },
    )

    print(grouped_all.to_string(index=False))
    print(json.dumps({"figure_path": str(figure_path), "summary_json": str(summary_json)}, indent=2))


if __name__ == "__main__":
    main()
