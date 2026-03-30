from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts") / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd


ARTIFACT_ROOT = Path("artifacts")
OPTUNA_DIR = ARTIFACT_ROOT / "optuna"
FIGURE_DIR = ARTIFACT_ROOT / "figures" / "optuna"
TABLE_DIR = ARTIFACT_ROOT / "optuna"
FAMILIES = ["rnn", "gru", "lstm"]
TOP_K = 5
RANDOM_SEED = 20260330

PARAM_COLUMNS = {
    "params_BATCH_SIZE": "batch_size",
    "params_FEATURE_DEPTH": "feature_depth",
    "params_FEATURE_WIDTH": "feature_width",
    "params_GRAD_CLIP_VALUE": "grad_clip_value",
    "params_LEARNING_RATE": "learning_rate",
    "params_N_HIDDEN": "n_hidden",
    "params_READOUT_DEPTH": "readout_depth",
    "params_READOUT_WIDTH": "readout_width",
    "params_WEIGHT_DECAY": "weight_decay",
}

FAMILY_COLORS = {
    "rnn": "#1f77b4",
    "gru": "#2ca02c",
    "lstm": "#d62728",
}


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_study_for_family(family: str) -> optuna.Study:
    summary = load_json(OPTUNA_DIR / f"study_summary_{family}.json")
    storage_path = OPTUNA_DIR / f"study_{family}.db"
    return optuna.load_study(
        study_name=summary["study_name"],
        storage=f"sqlite:///{storage_path.resolve()}",
    )


def load_family_data(family: str) -> dict[str, object]:
    trials_df = pd.read_csv(OPTUNA_DIR / f"trials_{family}.csv")
    timing_df = pd.read_csv(OPTUNA_DIR / f"timing_{family}.csv")
    summary = load_json(OPTUNA_DIR / f"study_summary_{family}.json")
    best_params_payload = load_json(OPTUNA_DIR / f"best_params_{family}.json")
    study = load_study_for_family(family)

    trials_df = trials_df.rename(columns=PARAM_COLUMNS).copy()
    trials_df["family"] = family
    trials_df["elapsed_minutes"] = pd.to_numeric(
        trials_df["user_attrs_elapsed_seconds"], errors="coerce"
    ) / 60.0
    trials_df["best_so_far"] = trials_df["value"].cummin()

    timing_df = timing_df.copy()
    timing_df["elapsed_minutes"] = pd.to_numeric(
        timing_df["elapsed_seconds"], errors="coerce"
    ) / 60.0

    merged = trials_df.merge(
        timing_df,
        left_on="number",
        right_on="trial_number",
        how="left",
        suffixes=("", "_timing"),
    )
    merged["family"] = family

    return {
        "family": family,
        "study": study,
        "trials_df": trials_df,
        "timing_df": timing_df,
        "merged_df": merged,
        "summary": summary,
        "best_params_payload": best_params_payload,
    }


def format_loss(value: float) -> str:
    return f"{value:.2e}"


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    ensure_directory(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_family_best_value_bar(family_summary_df: pd.DataFrame) -> Path:
    ordered = family_summary_df.sort_values("best_value").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        ordered["family"],
        ordered["best_value"],
        color=[FAMILY_COLORS[family] for family in ordered["family"]],
        alpha=0.9,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Best validation loss")
    ax.set_title("Best validation loss by recurrent core")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, ordered["best_value"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            format_loss(float(value)),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return save_figure(fig, FIGURE_DIR / "02_family_best_validation_loss.png")


def make_family_best_duration_bar(family_summary_df: pd.DataFrame) -> Path:
    ordered = family_summary_df.sort_values("best_trial_elapsed_seconds").reset_index(drop=True)
    minutes = ordered["best_trial_elapsed_seconds"] / 60.0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        ordered["family"],
        minutes,
        color=[FAMILY_COLORS[family] for family in ordered["family"]],
        alpha=0.9,
    )
    ax.set_ylabel("Best-trial runtime [min]")
    ax.set_title("Runtime of the best trial by recurrent core")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, minutes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return save_figure(fig, FIGURE_DIR / "02_family_best_trial_runtime.png")


def make_family_value_distribution(all_trials_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ordered_families = ["rnn", "gru", "lstm"]
    data = [
        all_trials_df.loc[all_trials_df["family"] == family, "value"].to_numpy()
        for family in ordered_families
    ]
    box = ax.boxplot(data, tick_labels=ordered_families, patch_artist=True, showfliers=True)
    for patch, family in zip(box["boxes"], ordered_families):
        patch.set_facecolor(FAMILY_COLORS[family])
        patch.set_alpha(0.55)
    for family_index, family in enumerate(ordered_families, start=1):
        family_df = all_trials_df.loc[all_trials_df["family"] == family]
        jitter = np.random.default_rng(RANDOM_SEED + family_index).uniform(-0.12, 0.12, size=len(family_df))
        ax.scatter(
            np.full(len(family_df), family_index) + jitter,
            family_df["value"],
            s=18,
            alpha=0.45,
            color=FAMILY_COLORS[family],
        )
    ax.set_yscale("log")
    ax.set_ylabel("Validation loss")
    ax.set_title("Distribution of Optuna trial losses")
    ax.grid(axis="y", alpha=0.3)
    return save_figure(fig, FIGURE_DIR / "02_family_trial_loss_distribution.png")


def make_family_runtime_distribution(all_trials_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ordered_families = ["rnn", "gru", "lstm"]
    data = [
        all_trials_df.loc[all_trials_df["family"] == family, "elapsed_minutes"].to_numpy()
        for family in ordered_families
    ]
    box = ax.boxplot(data, tick_labels=ordered_families, patch_artist=True, showfliers=True)
    for patch, family in zip(box["boxes"], ordered_families):
        patch.set_facecolor(FAMILY_COLORS[family])
        patch.set_alpha(0.55)
    for family_index, family in enumerate(ordered_families, start=1):
        family_df = all_trials_df.loc[all_trials_df["family"] == family]
        jitter = np.random.default_rng(RANDOM_SEED + 20 + family_index).uniform(-0.12, 0.12, size=len(family_df))
        ax.scatter(
            np.full(len(family_df), family_index) + jitter,
            family_df["elapsed_minutes"],
            s=18,
            alpha=0.45,
            color=FAMILY_COLORS[family],
        )
    ax.set_ylabel("Trial runtime [min]")
    ax.set_title("Distribution of Optuna trial runtimes")
    ax.grid(axis="y", alpha=0.3)
    return save_figure(fig, FIGURE_DIR / "02_family_trial_runtime_distribution.png")


def make_value_runtime_tradeoff(all_trials_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        family_df = all_trials_df.loc[all_trials_df["family"] == family].copy()
        scatter = ax.scatter(
            family_df["elapsed_minutes"],
            family_df["value"],
            c=family_df["n_hidden"],
            cmap="viridis",
            s=48,
            alpha=0.85,
            edgecolors="none",
        )
        best_idx = family_df["value"].idxmin()
        best_row = family_df.loc[best_idx]
        ax.scatter(
            [best_row["elapsed_minutes"]],
            [best_row["value"]],
            marker="*",
            s=200,
            color="#ff7f0e",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )
        ax.set_title(family.upper())
        ax.set_xlabel("Runtime [min]")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("n_hidden")
    axes[0].set_ylabel("Validation loss")
    fig.suptitle("Accuracy-runtime trade-off across Optuna trials", y=1.02)
    return save_figure(fig, FIGURE_DIR / "02_value_runtime_tradeoff.png")


def make_optimization_history(family: str, trials_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    ax.scatter(trials_df["number"], trials_df["value"], s=32, alpha=0.75, color=FAMILY_COLORS[family], label="trial value")
    ax.plot(trials_df["number"], trials_df["best_so_far"], color="black", linewidth=1.6, label="best so far")
    best_row = trials_df.loc[trials_df["value"].idxmin()]
    ax.scatter(
        [best_row["number"]],
        [best_row["value"]],
        marker="*",
        s=220,
        color="#ff7f0e",
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
        label="best trial",
    )
    ax.set_title(f"{family.upper()} optimization history")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    return save_figure(fig, FIGURE_DIR / f"02_{family}_optimization_history.png")


def make_hidden_loss_scatter(family: str, trials_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    batch_sizes = sorted(trials_df["batch_size"].dropna().unique().tolist())
    markers = {8: "o", 16: "s", 32: "^"}
    for batch_size in batch_sizes:
        subset = trials_df.loc[trials_df["batch_size"] == batch_size]
        ax.scatter(
            subset["n_hidden"],
            subset["value"],
            s=60,
            alpha=0.8,
            label=f"batch={int(batch_size)}",
            marker=markers.get(int(batch_size), "o"),
        )
    ax.set_title(f"{family.upper()} loss vs hidden dimension")
    ax.set_xlabel("n_hidden")
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    return save_figure(fig, FIGURE_DIR / f"02_{family}_loss_vs_hidden.png")


def make_runtime_hidden_scatter(family: str, trials_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    batch_sizes = sorted(trials_df["batch_size"].dropna().unique().tolist())
    markers = {8: "o", 16: "s", 32: "^"}
    for batch_size in batch_sizes:
        subset = trials_df.loc[trials_df["batch_size"] == batch_size]
        ax.scatter(
            subset["n_hidden"],
            subset["elapsed_minutes"],
            s=60,
            alpha=0.8,
            label=f"batch={int(batch_size)}",
            marker=markers.get(int(batch_size), "o"),
        )
    ax.set_title(f"{family.upper()} runtime vs hidden dimension")
    ax.set_xlabel("n_hidden")
    ax.set_ylabel("Runtime [min]")
    ax.grid(alpha=0.3)
    ax.legend()
    return save_figure(fig, FIGURE_DIR / f"02_{family}_runtime_vs_hidden.png")


def make_slice_grid(family: str, trials_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_specs = [
        ("learning_rate", "Learning rate", True),
        ("weight_decay", "Weight decay", True),
        ("readout_width", "Readout width", False),
        ("n_hidden", "n_hidden", False),
    ]
    for ax, (column, label, use_log_x) in zip(axes.flat, plot_specs):
        ax.scatter(trials_df[column], trials_df["value"], c=trials_df["batch_size"], cmap="plasma", s=52, alpha=0.85)
        if use_log_x:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Validation loss")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{family.upper()} hyperparameter slices", y=1.01)
    return save_figure(fig, FIGURE_DIR / f"02_{family}_slice_grid.png")


def make_param_importance_plot(family: str, study: optuna.Study) -> Path:
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        importances = {}

    if not importances:
        importances = {
            "N_HIDDEN": np.nan,
            "BATCH_SIZE": np.nan,
            "LEARNING_RATE": np.nan,
            "READOUT_WIDTH": np.nan,
            "WEIGHT_DECAY": np.nan,
        }

    importance_df = pd.DataFrame(
        {"parameter": list(importances.keys()), "importance": list(importances.values())}
    ).sort_values("importance", ascending=True)
    importance_df.to_csv(TABLE_DIR / f"param_importance_{family}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    ax.barh(importance_df["parameter"], importance_df["importance"], color=FAMILY_COLORS[family], alpha=0.85)
    ax.set_xlabel("Optuna importance")
    ax.set_title(f"{family.upper()} parameter importance")
    ax.grid(axis="x", alpha=0.3)
    return save_figure(fig, FIGURE_DIR / f"02_{family}_param_importance.png")


def build_summary_tables(all_trials_df: pd.DataFrame, family_summary_df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []

    overall_summary = (
        all_trials_df.groupby("family", as_index=False)
        .agg(
            n_trials=("number", "count"),
            min_loss=("value", "min"),
            median_loss=("value", "median"),
            mean_loss=("value", "mean"),
            mean_runtime_minutes=("elapsed_minutes", "mean"),
            median_runtime_minutes=("elapsed_minutes", "median"),
            min_runtime_minutes=("elapsed_minutes", "min"),
            max_runtime_minutes=("elapsed_minutes", "max"),
        )
        .sort_values("min_loss")
    )
    overall_summary_path = TABLE_DIR / "family_summary_report.csv"
    overall_summary.to_csv(overall_summary_path, index=False)
    paths.append(overall_summary_path)

    top_trials = (
        all_trials_df.sort_values(["value", "elapsed_minutes"])
        .groupby("family", group_keys=False)
        .head(TOP_K)
        .loc[
            :,
            [
                "family",
                "number",
                "value",
                "elapsed_minutes",
                "n_hidden",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "readout_width",
                "readout_depth",
                "feature_width",
                "feature_depth",
                "user_attrs_best_epoch",
            ],
        ]
    )
    top_trials_path = TABLE_DIR / "top_trials_report.csv"
    top_trials.to_csv(top_trials_path, index=False)
    paths.append(top_trials_path)

    family_summary_path = TABLE_DIR / "family_comparison_report.csv"
    family_summary_df.to_csv(family_summary_path, index=False)
    paths.append(family_summary_path)

    return paths


def main() -> None:
    ensure_directory(FIGURE_DIR)
    ensure_directory(TABLE_DIR)

    family_summary_df = pd.read_csv(OPTUNA_DIR / "family_comparison.csv")
    family_payloads = [load_family_data(family) for family in FAMILIES]
    all_trials_df = pd.concat([payload["merged_df"] for payload in family_payloads], ignore_index=True)

    figure_paths = [
        make_family_best_value_bar(family_summary_df=family_summary_df),
        make_family_best_duration_bar(family_summary_df=family_summary_df),
        make_family_value_distribution(all_trials_df=all_trials_df),
        make_family_runtime_distribution(all_trials_df=all_trials_df),
        make_value_runtime_tradeoff(all_trials_df=all_trials_df),
    ]

    for payload in family_payloads:
        family = payload["family"]
        merged_df = payload["merged_df"]
        trials_df = payload["trials_df"]
        study = payload["study"]
        figure_paths.extend(
            [
                make_optimization_history(family=family, trials_df=trials_df),
                make_hidden_loss_scatter(family=family, trials_df=merged_df),
                make_runtime_hidden_scatter(family=family, trials_df=merged_df),
                make_slice_grid(family=family, trials_df=merged_df),
                make_param_importance_plot(family=family, study=study),
            ]
        )

    table_paths = build_summary_tables(all_trials_df=all_trials_df, family_summary_df=family_summary_df)

    manifest_path = TABLE_DIR / "optuna_report_plot_manifest.json"
    manifest_payload = {
        "figure_paths": [str(path) for path in figure_paths],
        "table_paths": [str(path) for path in table_paths],
        "families": FAMILIES,
        "top_k": TOP_K,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    print(json.dumps(manifest_payload, indent=2))


if __name__ == "__main__":
    main()
