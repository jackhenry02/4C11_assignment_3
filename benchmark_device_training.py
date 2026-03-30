from __future__ import annotations

"""Replay one representative training configuration on CPU and MPS for runtime benchmarking."""

import json
import os
import time
from pathlib import Path

import pandas as pd

from Coursework3.RNO_1D_Skeleton import (
    DEFAULT_SPLIT_SEED,
    ExperimentConfig,
    ensure_artifact_tree,
    prepare_data,
    select_device,
    train_model,
    write_json,
)


ARTIFACT_ROOT = Path("artifacts_benchmark")
TRAIN_PATH = Path("Coursework3/viscodata_3mat.mat")
SEED = DEFAULT_SPLIT_SEED
DEVICES = [item.strip().lower() for item in os.environ.get("BENCHMARK_DEVICES", "cpu,mps").split(",") if item.strip()]
BEST_PARAMS_PATH = Path("artifacts/optuna/best_params_rnn.json")
RUN_TAG = os.environ.get("BENCHMARK_RUN_TAG", "").strip()

CORE_TYPE = "rnn"
EPOCHS = 220
MIN_EPOCHS = 90
EARLY_STOPPING_PATIENCE = 60
LR_FACTOR = 0.5
LR_PATIENCE = 15
NUM_WORKERS = 0
SHUFFLE = True
VERBOSE = False
PRINT_EVERY_EPOCHS = 10
TRAIN_ON_TRAIN_PLUS_VAL = False
FIXED_EPOCHS = int(os.environ["BENCHMARK_FIXED_EPOCHS"]) if os.environ.get("BENCHMARK_FIXED_EPOCHS") else None
BATCH_SIZE_OVERRIDE = int(os.environ["BENCHMARK_BATCH_SIZE"]) if os.environ.get("BENCHMARK_BATCH_SIZE") else None

FALLBACK_CONFIG = {
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 1.8177929193890178e-03,
    "WEIGHT_DECAY": 1.8123914365651553e-05,
    "GRAD_CLIP_VALUE": 1.0,
    "N_HIDDEN": 48,
    "READOUT_WIDTH": 96,
    "READOUT_DEPTH": 4,
    "FEATURE_WIDTH": 32,
    "FEATURE_DEPTH": 1,
}


def load_reference_config() -> tuple[ExperimentConfig, dict[str, str | int | float]]:
    """Use the saved best RNN config when available, otherwise fall back to a fixed recipe."""
    payload: dict[str, object] = {}
    if BEST_PARAMS_PATH.exists():
        payload = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
        best_user_attrs = payload.get("best_user_attrs", {})
        config_json = best_user_attrs.get("config") if isinstance(best_user_attrs, dict) else None
        if isinstance(config_json, str):
            config_dict = json.loads(config_json)
            config = ExperimentConfig(**config_dict)
            if BATCH_SIZE_OVERRIDE is not None:
                config.BATCH_SIZE = BATCH_SIZE_OVERRIDE
            meta = {
                "source": str(BEST_PARAMS_PATH),
                "best_trial_number": int(payload.get("best_trial_number", -1)),
                "best_value": float(payload.get("best_value", float("nan"))),
            }
            return config, meta

    config = ExperimentConfig(
        CORE_TYPE=CORE_TYPE,
        EPOCHS=EPOCHS,
        BATCH_SIZE=int(FALLBACK_CONFIG["BATCH_SIZE"]),
        LEARNING_RATE=float(FALLBACK_CONFIG["LEARNING_RATE"]),
        WEIGHT_DECAY=float(FALLBACK_CONFIG["WEIGHT_DECAY"]),
        GRAD_CLIP_VALUE=float(FALLBACK_CONFIG["GRAD_CLIP_VALUE"]),
        LR_FACTOR=LR_FACTOR,
        LR_PATIENCE=LR_PATIENCE,
        EARLY_STOPPING_PATIENCE=EARLY_STOPPING_PATIENCE,
        MIN_EPOCHS=MIN_EPOCHS,
        N_HIDDEN=int(FALLBACK_CONFIG["N_HIDDEN"]),
        READOUT_WIDTH=int(FALLBACK_CONFIG["READOUT_WIDTH"]),
        READOUT_DEPTH=int(FALLBACK_CONFIG["READOUT_DEPTH"]),
        FEATURE_WIDTH=int(FALLBACK_CONFIG["FEATURE_WIDTH"]),
        FEATURE_DEPTH=int(FALLBACK_CONFIG["FEATURE_DEPTH"]),
        SHUFFLE=SHUFFLE,
        NUM_WORKERS=NUM_WORKERS,
        SEED=SEED,
        MODEL_TAG="benchmark_reference",
        USE_TRUE_INITIAL_OUTPUT=True,
        VERBOSE=VERBOSE,
        PRINT_EVERY_EPOCHS=PRINT_EVERY_EPOCHS,
    )
    if BATCH_SIZE_OVERRIDE is not None:
        config.BATCH_SIZE = BATCH_SIZE_OVERRIDE
    meta = {
        "source": "fallback_config",
        "best_trial_number": -1,
        "best_value": float("nan"),
    }
    return config, meta


def benchmark_device(device_name: str, data_bundle: dict, base_config: ExperimentConfig) -> dict[str, object]:
    """Train the same configuration on one requested device and record timing and best loss."""
    os.environ["COURSEWORK_DEVICE"] = device_name
    start = time.perf_counter()
    try:
        selected_device = str(select_device())
    except Exception as exc:
        return {
            "requested_device": device_name,
            "selected_device": None,
            "status": "unavailable",
            "error": str(exc),
        }

    config = ExperimentConfig(**base_config.__dict__)
    model_tag_parts = [base_config.MODEL_TAG]
    run_name_parts = ["benchmark", config.CORE_TYPE]
    if RUN_TAG:
        model_tag_parts.append(RUN_TAG)
        run_name_parts.append(RUN_TAG)
    model_tag_parts.append(device_name)
    run_name_parts.append(device_name)
    config.MODEL_TAG = "_".join(model_tag_parts)
    run_name = "_".join(run_name_parts)

    result = train_model(
        data_bundle=data_bundle,
        config=config,
        artifact_root=ARTIFACT_ROOT,
        run_name=run_name,
        train_on_train_plus_val=TRAIN_ON_TRAIN_PLUS_VAL,
        fixed_epochs=FIXED_EPOCHS,
    )
    total_seconds = time.perf_counter() - start

    return {
        "requested_device": device_name,
        "selected_device": selected_device,
        "status": "complete",
        "error": None,
        "elapsed_seconds": total_seconds,
        "epochs_completed": int(result["n_epochs_completed"]),
        "best_epoch": int(result["best_epoch"]),
        "best_val_loss": float(result["best_val_loss"]),
        "checkpoint_path": str(result["checkpoint_path"]),
        "history_path": str(result["history_path"]),
        "summary_path": str(result["summary_path"]),
    }


def main() -> None:
    """Run the device comparison and save a machine-readable summary."""
    directories = ensure_artifact_tree(ARTIFACT_ROOT)
    benchmark_dir = directories["logs"]

    base_config, meta = load_reference_config()
    data_bundle = prepare_data(train_path=TRAIN_PATH, artifact_root=ARTIFACT_ROOT, split_seed=SEED)

    rows: list[dict[str, object]] = []
    for device_name in DEVICES:
        rows.append(benchmark_device(device_name=device_name, data_bundle=data_bundle, base_config=base_config))

    results_df = pd.DataFrame(rows)
    suffix = f"_{RUN_TAG}" if RUN_TAG else ""
    results_path = benchmark_dir / f"device_benchmark_results{suffix}.csv"
    results_df.to_csv(results_path, index=False)

    payload = {
        "benchmark_artifact_root": str(ARTIFACT_ROOT),
        "reference_config_source": meta,
        "reference_config": base_config.__dict__,
        "devices": DEVICES,
        "run_tag": RUN_TAG,
        "results_path": str(results_path),
    }

    if {"cpu", "mps"}.issubset(set(results_df["requested_device"])) and (results_df["status"] == "complete").all():
        cpu_elapsed = float(results_df.loc[results_df["requested_device"] == "cpu", "elapsed_seconds"].iloc[0])
        mps_elapsed = float(results_df.loc[results_df["requested_device"] == "mps", "elapsed_seconds"].iloc[0])
        payload["speed_ratio_mps_over_cpu"] = mps_elapsed / cpu_elapsed if cpu_elapsed > 0 else None

    summary_path = benchmark_dir / f"device_benchmark_summary{suffix}.json"
    write_json(summary_path, payload)

    print(results_df.to_string(index=False))
    print(f"\nSaved benchmark results to {results_path}")
    print(f"Saved benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
