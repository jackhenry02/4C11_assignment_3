from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts") / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARTIFACT_ROOT = Path("artifacts")
OPTUNA_DIR = ARTIFACT_ROOT / "optuna"
LOG_DIR = ARTIFACT_ROOT / "logs"
FIGURE_DIR = ARTIFACT_ROOT / "figures"
OPTUNA_FIGURE_DIR = FIGURE_DIR / "optuna"
HIDDEN_FIGURE_DIR = FIGURE_DIR
FAMILIES = ["rnn", "gru", "lstm"]
FAMILY_COLORS = {
    "rnn": "#1f77b4",
    "gru": "#2ca02c",
    "lstm": "#d62728",
}
HIDDEN_COLOR_MAP = plt.colormaps["viridis"]


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def history_path_for_best_optuna_family(family: str) -> tuple[Path, dict]:
    payload = load_json(OPTUNA_DIR / f"best_params_{family}.json")
    best_trial_number = int(payload["best_trial_number"])
    history_path = LOG_DIR / f"optuna_{family}_trial_{best_trial_number:04d}_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file for {family}: {history_path}")
    return history_path, payload


def plot_best_optuna_curves() -> list[Path]:
    ensure_directory(OPTUNA_FIGURE_DIR)
    saved_paths: list[Path] = []
    combined_fig, combined_axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, family in zip(combined_axes, FAMILIES):
        history_path, payload = history_path_for_best_optuna_family(family)
        history = pd.read_csv(history_path)
        best_epoch = int(payload["best_user_attrs"]["best_epoch"])
        best_value = float(payload["best_value"])

        ax.plot(history["epoch"], history["train_loss"], color=FAMILY_COLORS[family], alpha=0.35, linewidth=1.5, label="train")
        ax.plot(history["epoch"], history["val_loss"], color=FAMILY_COLORS[family], linewidth=2.0, label="validation")
        ax.scatter(
            [best_epoch],
            [best_value],
            marker="*",
            s=180,
            color="#ff7f0e",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
            label="best epoch",
        )
        ax.set_title(f"{family.upper()} best Optuna run")
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        if ax is combined_axes[0]:
            ax.set_ylabel("Loss")
        ax.legend()

        family_fig, family_ax = plt.subplots(figsize=(7.2, 4.5))
        family_ax.plot(history["epoch"], history["train_loss"], color=FAMILY_COLORS[family], alpha=0.35, linewidth=1.6, label="train")
        family_ax.plot(history["epoch"], history["val_loss"], color=FAMILY_COLORS[family], linewidth=2.1, label="validation")
        family_ax.scatter(
            [best_epoch],
            [best_value],
            marker="*",
            s=220,
            color="#ff7f0e",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
            label=f"best epoch = {best_epoch}",
        )
        family_ax.set_title(f"{family.upper()} best Optuna training curve")
        family_ax.set_xlabel("Epoch")
        family_ax.set_ylabel("Loss")
        family_ax.set_yscale("log")
        family_ax.grid(alpha=0.3)
        family_ax.legend()
        family_path = OPTUNA_FIGURE_DIR / f"02_{family}_best_training_curve.png"
        family_fig.tight_layout()
        family_fig.savefig(family_path, dpi=220, bbox_inches="tight")
        plt.close(family_fig)
        saved_paths.append(family_path)

    combined_fig.tight_layout()
    combined_path = OPTUNA_FIGURE_DIR / "02_best_optuna_training_curves.png"
    combined_fig.savefig(combined_path, dpi=220, bbox_inches="tight")
    plt.close(combined_fig)
    saved_paths.append(combined_path)
    return saved_paths


def parse_hidden_history_filename(path: Path) -> tuple[int, int] | None:
    match = re.search(r"_h(\d+)_seed_(\d+)_history\.csv$", path.name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def collect_hidden_histories() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    patterns = [
        ("hidden_threshold_low_dim", LOG_DIR.glob("hidden_threshold_low_dim_*_history.csv")),
        ("hidden_threshold_adaptive", LOG_DIR.glob("hidden_threshold_adaptive_*_history.csv")),
    ]
    for run_prefix, paths in patterns:
        for history_path in paths:
            parsed = parse_hidden_history_filename(history_path)
            if parsed is None:
                continue
            n_hidden, seed = parsed
            summary_path = history_path.with_name(history_path.name.replace("_history.csv", "_summary.json"))
            best_val_loss = np.nan
            best_epoch = np.nan
            device = ""
            if summary_path.exists():
                summary = load_json(summary_path)
                best_val_loss = float(summary["best_val_loss"])
                best_epoch = float(summary["best_epoch"])
                device = str(summary.get("device", ""))
            rows.append(
                {
                    "run_prefix": run_prefix,
                    "history_path": history_path,
                    "summary_path": summary_path,
                    "n_hidden": n_hidden,
                    "seed": seed,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "device": device,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_prefix",
                "history_path",
                "summary_path",
                "n_hidden",
                "seed",
                "best_val_loss",
                "best_epoch",
                "device",
            ]
        )
    histories = pd.DataFrame(rows)
    histories = histories.sort_values(
        ["n_hidden", "best_val_loss", "run_prefix", "seed"],
        na_position="last",
    ).reset_index(drop=True)
    return histories


def plot_hidden_validation_overlay() -> Path | None:
    ensure_directory(HIDDEN_FIGURE_DIR)
    history_index = collect_hidden_histories()
    if history_index.empty:
        return None

    best_per_hidden = (
        history_index.sort_values(["n_hidden", "best_val_loss"], na_position="last")
        .groupby("n_hidden", as_index=False)
        .first()
        .sort_values("n_hidden")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    n_hidden_values = best_per_hidden["n_hidden"].tolist()
    color_positions = np.linspace(0.15, 0.9, num=max(len(n_hidden_values), 2))

    for color_position, (_, row) in zip(color_positions, best_per_hidden.iterrows()):
        history = pd.read_csv(row["history_path"])
        n_hidden = int(row["n_hidden"])
        seed = int(row["seed"])
        color = HIDDEN_COLOR_MAP(float(color_position))
        label = f"h={n_hidden}, seed={seed}"
        if row["device"]:
            label += f", {row['device']}"
        ax.plot(
            history["epoch"],
            history["val_loss"],
            linewidth=2.0,
            color=color,
            label=label,
        )
        if np.isfinite(row["best_epoch"]) and np.isfinite(row["best_val_loss"]):
            ax.scatter(
                [float(row["best_epoch"])],
                [float(row["best_val_loss"])],
                s=70,
                color=color,
                edgecolors="black",
                linewidths=0.4,
                zorder=5,
            )

    ax.set_title("Validation-loss overlay for hidden-size sweep")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    figure_path = HIDDEN_FIGURE_DIR / "03_hidden_sweep_validation_overlay.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def main() -> None:
    ensure_directory(OPTUNA_FIGURE_DIR)
    ensure_directory(HIDDEN_FIGURE_DIR)

    optuna_paths = plot_best_optuna_curves()
    hidden_overlay_path = plot_hidden_validation_overlay()

    manifest = {
        "optuna_training_curve_paths": [str(path) for path in optuna_paths],
        "hidden_overlay_path": str(hidden_overlay_path) if hidden_overlay_path is not None else None,
    }
    manifest_path = ARTIFACT_ROOT / "optuna" / "training_curve_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
