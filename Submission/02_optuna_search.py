from __future__ import annotations

"""Run the main Optuna architecture comparison across RNN, GRU, and LSTM cores."""

import json
import os
import time
from pathlib import Path

import optuna
import pandas as pd

from Coursework3.RNO_1D_Skeleton import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_SPLIT_SEED,
    ExperimentConfig,
    ensure_artifact_tree,
    prepare_data,
    train_model,
    write_json,
)


ARTIFACT_ROOT = Path(DEFAULT_ARTIFACT_ROOT)
TRAIN_PATH = Path("Coursework3/viscodata_3mat.mat")
SEED = DEFAULT_SPLIT_SEED
DEVICE = os.environ.get("COURSEWORK_OPTUNA_DEVICE", "cpu").strip().lower() or "cpu"
FAMILIES = ["rnn", "gru", "lstm"]
N_TRIALS = 30
TIMEOUT_SECONDS = None
STUDY_DIRECTION = "minimize"
EPOCHS = 220
MIN_EPOCHS = 90
EARLY_STOPPING_PATIENCE = 60
LR_FACTOR = 0.5
LR_PATIENCE = 15
NUM_WORKERS = 0
VERBOSE = False
PRINT_EVERY_EPOCHS = 10


def trial_to_config(trial: optuna.Trial, core_type: str, model_tag: str) -> ExperimentConfig:
    """Translate one Optuna suggestion into the shared training-config object."""
    return ExperimentConfig(
        CORE_TYPE=core_type,
        EPOCHS=EPOCHS,
        BATCH_SIZE=trial.suggest_categorical("BATCH_SIZE", [8, 16, 32]),
        LEARNING_RATE=trial.suggest_float("LEARNING_RATE", 1e-4, 5e-3, log=True),
        WEIGHT_DECAY=trial.suggest_float("WEIGHT_DECAY", 1e-6, 1e-2, log=True),
        GRAD_CLIP_VALUE=trial.suggest_categorical("GRAD_CLIP_VALUE", [0.5, 1.0, 2.0, 5.0]),
        LR_FACTOR=LR_FACTOR,
        LR_PATIENCE=LR_PATIENCE,
        EARLY_STOPPING_PATIENCE=EARLY_STOPPING_PATIENCE,
        MIN_EPOCHS=MIN_EPOCHS,
        N_HIDDEN=trial.suggest_int("N_HIDDEN", 4, 48, step=4),
        READOUT_WIDTH=trial.suggest_categorical("READOUT_WIDTH", [64, 96, 128, 192]),
        READOUT_DEPTH=trial.suggest_int("READOUT_DEPTH", 2, 4),
        FEATURE_WIDTH=trial.suggest_categorical("FEATURE_WIDTH", [16, 32, 64, 96]),
        FEATURE_DEPTH=trial.suggest_int("FEATURE_DEPTH", 1, 3),
        SHUFFLE=True,
        NUM_WORKERS=NUM_WORKERS,
        SEED=SEED + trial.number,
        MODEL_TAG=model_tag,
        USE_TRUE_INITIAL_OUTPUT=True,
        VERBOSE=VERBOSE,
        PRINT_EVERY_EPOCHS=PRINT_EVERY_EPOCHS,
    )


def export_study_outputs(study: optuna.Study, family: str, artifact_root: str | Path = ARTIFACT_ROOT) -> dict[str, Path]:
    """Persist the study tables after each trial so long runs remain recoverable."""
    directories = ensure_artifact_tree(artifact_root)
    optuna_dir = directories["optuna"]

    trials_df = study.trials_dataframe()
    trials_path = optuna_dir / f"trials_{family}.csv"
    trials_df.to_csv(trials_path, index=False)

    timing_rows = []
    for trial in study.trials:
        duration_seconds = trial.duration.total_seconds() if trial.duration is not None else None
        elapsed_seconds = trial.user_attrs.get("elapsed_seconds")
        timing_rows.append(
            {
                "family": family,
                "trial_number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "duration_seconds": duration_seconds,
                "elapsed_seconds": elapsed_seconds,
                "best_epoch": trial.user_attrs.get("best_epoch"),
                "n_hidden": trial.params.get("N_HIDDEN"),
                "batch_size": trial.params.get("BATCH_SIZE"),
                "learning_rate": trial.params.get("LEARNING_RATE"),
                "weight_decay": trial.params.get("WEIGHT_DECAY"),
                "readout_width": trial.params.get("READOUT_WIDTH"),
                "readout_depth": trial.params.get("READOUT_DEPTH"),
                "feature_width": trial.params.get("FEATURE_WIDTH"),
                "feature_depth": trial.params.get("FEATURE_DEPTH"),
                "checkpoint_path": trial.user_attrs.get("checkpoint_path"),
            }
        )
    timing_df = pd.DataFrame(timing_rows)
    timing_path = optuna_dir / f"timing_{family}.csv"
    timing_df.to_csv(timing_path, index=False)

    best_trial = study.best_trial
    best_params_payload = {
        "core_type": family,
        "study_name": study.study_name,
        "best_value": float(study.best_value),
        "best_trial_number": int(best_trial.number),
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs,
    }
    best_params_path = optuna_dir / f"best_params_{family}.json"
    write_json(best_params_path, best_params_payload)

    summary_path = optuna_dir / f"study_summary_{family}.json"
    write_json(
        summary_path,
        {
                "n_trials": len(study.trials),
                "best_value": float(study.best_value),
                "best_trial_number": int(best_trial.number),
                "best_trial_elapsed_seconds": best_trial.user_attrs.get("elapsed_seconds"),
                "study_name": study.study_name,
                "storage": study._storage.url if hasattr(study._storage, "url") else "sqlite",
            },
        )
    return {
        "trials_path": trials_path,
        "timing_path": timing_path,
        "best_params_path": best_params_path,
        "summary_path": summary_path,
    }


def run_family_study(
    family: str,
    data_bundle: dict,
    artifact_root: str | Path = ARTIFACT_ROOT,
) -> dict[str, Path]:
    """Optimise one recurrent-core family while continuously exporting progress to disk."""
    directories = ensure_artifact_tree(artifact_root)
    study_name = f"coursework3_{family}"
    storage_path = directories["optuna"] / f"study_{family}.db"
    storage = f"sqlite:///{storage_path}"

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=STUDY_DIRECTION,
        sampler=sampler,
    )
    completed_trials = sum(1 for trial in study.trials if trial.state.is_finished())
    remaining_trials = max(N_TRIALS - completed_trials, 0)

    def objective(trial: optuna.Trial) -> float:
        model_tag = f"optuna_{family}_trial_{trial.number:04d}"
        # Each trial is a full independent training run under one sampled hyperparameter set.
        config = trial_to_config(trial=trial, core_type=family, model_tag=model_tag)
        print(
            f"[optuna] family={family} trial={trial.number + 1} "
            f"n_hidden={config.N_HIDDEN} batch={config.BATCH_SIZE} "
            f"lr={config.LEARNING_RATE:.3e} wd={config.WEIGHT_DECAY:.3e} "
            f"readout=({config.READOUT_DEPTH},{config.READOUT_WIDTH}) "
            f"feature=({config.FEATURE_DEPTH},{config.FEATURE_WIDTH})",
            flush=True,
        )
        trial_start = time.perf_counter()
        result = train_model(
            data_bundle=data_bundle,
            config=config,
            artifact_root=artifact_root,
            run_name=model_tag,
        )
        elapsed = time.perf_counter() - trial_start
        trial.set_user_attr("checkpoint_path", result["checkpoint_path"])
        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("config", json.dumps(config.__dict__, sort_keys=True))
        trial.set_user_attr("elapsed_seconds", elapsed)
        print(
            f"[optuna] family={family} trial={trial.number + 1} complete "
            f"best_val={result['best_val_loss']:.6e} best_epoch={result['best_epoch']} "
            f"elapsed_s={elapsed:.1f}",
            flush=True,
        )
        return float(result["best_val_loss"])

    def callback(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
        # Export after every finished trial so the run can be inspected or resumed safely.
        export_study_outputs(study=study, family=family, artifact_root=artifact_root)
        print(
            f"[optuna] family={family} progress completed_trials={len(study.trials)} "
            f"current_best={study.best_value:.6e} best_trial={study.best_trial.number + 1}",
            flush=True,
        )

    print(
        f"[optuna] starting family={family} device={DEVICE} target_trials={N_TRIALS} "
        f"completed_trials={completed_trials} remaining_trials={remaining_trials} "
        f"epochs={EPOCHS} min_epochs={MIN_EPOCHS}",
        flush=True,
    )
    family_start = time.perf_counter()
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, timeout=TIMEOUT_SECONDS, gc_after_trial=True, callbacks=[callback])
    family_elapsed = time.perf_counter() - family_start
    paths = export_study_outputs(study=study, family=family, artifact_root=artifact_root)
    paths["storage_path"] = storage_path
    print(
        f"[optuna] finished family={family} best_value={study.best_value:.6e} "
        f"best_trial={study.best_trial.number + 1} elapsed_s={family_elapsed:.1f}",
        flush=True,
    )
    return paths


def main() -> None:
    """Run the family studies sequentially and save a compact comparison table."""
    os.environ["COURSEWORK_DEVICE"] = DEVICE
    data_bundle = prepare_data(train_path=TRAIN_PATH, artifact_root=ARTIFACT_ROOT, split_seed=SEED)
    result_rows = []
    for family in FAMILIES:
        outputs = run_family_study(family=family, data_bundle=data_bundle, artifact_root=ARTIFACT_ROOT)
        best_params = json.loads(Path(outputs["best_params_path"]).read_text(encoding="utf-8"))
        result_rows.append(
            {
                "family": family,
                "best_value": best_params["best_value"],
                "best_trial_number": best_params["best_trial_number"],
                "best_params_path": str(outputs["best_params_path"]),
                "storage_path": str(outputs["storage_path"]),
                "trials_path": str(outputs["trials_path"]),
                "timing_path": str(outputs["timing_path"]),
                "best_trial_elapsed_seconds": best_params["best_user_attrs"].get("elapsed_seconds"),
            }
        )

    summary = pd.DataFrame(result_rows).sort_values("best_value")
    summary_path = ensure_artifact_tree(ARTIFACT_ROOT)["optuna"] / "family_comparison.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved family comparison to {summary_path}")


if __name__ == "__main__":
    main()
