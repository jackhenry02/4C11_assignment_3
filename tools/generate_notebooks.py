from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def notebook_metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }


def write_notebook(path: Path, cells: list) -> None:
    notebook = nbf.v4.new_notebook(cells=cells, metadata=notebook_metadata())
    path.write_text(nbf.writes(notebook), encoding="utf-8")


def build_00_environment_and_mps() -> None:
    cells = [
        nbf.v4.new_markdown_cell(
            "# 00 Environment And MPS\n\n"
            "This notebook validates the runtime, checks the dataset, and runs MPS smoke tests for the exact operator workflow."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n\n"
            "ARTIFACT_ROOT = Path('artifacts')\n"
            "TRAIN_PATH = Path('Coursework3/viscodata_3mat.mat')\n"
            "SEED = 20260328\n"
            "MPS_TEST_BATCH = 4\n"
            "MPS_TEST_SEQ_LEN = 1001\n"
            "MPS_TEST_HIDDEN = 16\n"
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n\n"
            "from Coursework3.RNO_1D_Skeleton import run_environment_and_mps_checks\n\n"
            "environment_results = run_environment_and_mps_checks(\n"
            "    TRAIN_PATH=TRAIN_PATH,\n"
            "    ARTIFACT_ROOT=ARTIFACT_ROOT,\n"
            "    SEED=SEED,\n"
            "    MPS_TEST_BATCH=MPS_TEST_BATCH,\n"
            "    MPS_TEST_SEQ_LEN=MPS_TEST_SEQ_LEN,\n"
            "    MPS_TEST_HIDDEN=MPS_TEST_HIDDEN,\n"
            ")\n\n"
            "summary_df = pd.DataFrame(environment_results['summary_rows'])\n"
            "smoke_df = pd.DataFrame(environment_results['smoke_rows'])\n"
            "summary_df"
        ),
        nbf.v4.new_code_cell("smoke_df"),
        nbf.v4.new_code_cell("environment_results"),
    ]
    write_notebook(ROOT / "00_environment_and_mps.ipynb", cells)


def build_01_eda_and_preprocessing() -> None:
    cells = [
        nbf.v4.new_markdown_cell(
            "# 01 EDA And Preprocessing\n\n"
            "This notebook explores the dataset, compares normalization choices, runs lightweight time-series diagnostics, and saves split/normalization artifacts."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n\n"
            "ARTIFACT_ROOT = Path('artifacts')\n"
            "TRAIN_PATH = Path('Coursework3/viscodata_3mat.mat')\n"
            "F_FIELD = 'epsi_tol'\n"
            "SIG_FIELD = 'sigma_tol'\n"
            "SEED = 20260328\n"
            "Ntotal = 400\n"
            "TRAIN_RATIO = 0.70\n"
            "VAL_RATIO = 0.15\n"
            "train_size = int(round(Ntotal * TRAIN_RATIO))\n"
            "val_size = int(round(Ntotal * VAL_RATIO))\n"
            "test_start = train_size + val_size\n"
            "ACF_LAGS = 60\n"
            "SCATTER_SAMPLES = 20000\n"
            "REPRESENTATIVE_SAMPLE_COUNT = 6\n"
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "from Coursework3.RNO_1D_Skeleton import run_eda_and_preprocessing\n\n"
            "eda_results = run_eda_and_preprocessing(\n"
            "    TRAIN_PATH=TRAIN_PATH,\n"
            "    ARTIFACT_ROOT=ARTIFACT_ROOT,\n"
            "    F_FIELD=F_FIELD,\n"
            "    SIG_FIELD=SIG_FIELD,\n"
            "    SEED=SEED,\n"
            "    ACF_LAGS=ACF_LAGS,\n"
            "    SCATTER_SAMPLES=SCATTER_SAMPLES,\n"
            "    REPRESENTATIVE_SAMPLE_COUNT=REPRESENTATIVE_SAMPLE_COUNT,\n"
            ")\n\n"
            "summary_df = pd.DataFrame(eda_results['summary_rows'])\n"
            "split_df = pd.DataFrame(eda_results['split_rows'])\n"
            "normalization_df = pd.DataFrame(eda_results['normalization_rows'])\n"
            "stationarity_df = pd.DataFrame(eda_results['stationarity_rows'])\n\n"
            "display(summary_df)\n"
            "display(split_df)\n"
            "display(normalization_df)\n"
            "display(stationarity_df)\n"
        ),
        nbf.v4.new_code_cell(
            "for figure_path in eda_results['figure_paths']:\n"
            "    display(Image(filename=str(figure_path)))"
        ),
        nbf.v4.new_code_cell("eda_results"),
    ]
    write_notebook(ROOT / "01_eda_and_preprocessing.ipynb", cells)


def build_03_final_training_and_hidden_threshold() -> None:
    cells = [
        nbf.v4.new_markdown_cell(
            "# 03 Final Training And Hidden Threshold\n\n"
            "Load the best Optuna family, freeze the non-hidden hyperparameters, run a cached low-dimensional hidden-size sweep on CPU, compare it against the existing `h=4` reference run, and optionally retrain the final chosen model."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n\n"
            "ARTIFACT_ROOT = Path('artifacts')\n"
            "TRAIN_PATH = Path('Coursework3/viscodata_3mat.mat')\n"
            "DEVICE = 'cpu'\n"
            "SEED = 20260328\n"
            "SWEEP_SEEDS = [20260328, 20260329, 20260330]\n"
            "HIDDEN_GRID = [0, 1, 2, 3]\n"
            "REFERENCE_RESULTS_PATH = ARTIFACT_ROOT / 'final' / 'hidden_threshold_adaptive_results.csv'\n"
            "REFERENCE_HIDDEN = 4\n"
            "TOLERANCE_RATIO = 0.05\n"
            "RUN_PREFIX = 'hidden_threshold_low_dim'\n"
            "HIDDEN_VERBOSE = True\n"
            "RUN_FINAL_RETRAIN = False\n"
            "FINAL_RUN_NAME = 'final_model'\n"
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "import os\n"
            "from pathlib import Path\n\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "from IPython.display import Image, display\n\n"
            "from Coursework3.RNO_1D_Skeleton import (\n"
            "    ExperimentConfig,\n"
            "    pick_best_family,\n"
            "    plot_hidden_threshold,\n"
            "    prepare_data,\n"
            "    retrain_final_model,\n"
            "    run_hidden_threshold_grid_progressive,\n"
            "    write_json,\n"
            ")\n\n"
            "os.environ['COURSEWORK_DEVICE'] = DEVICE\n\n"
            "data_bundle = prepare_data(train_path=TRAIN_PATH, artifact_root=ARTIFACT_ROOT, split_seed=SEED)\n"
            "best_param_paths = sorted(Path(ARTIFACT_ROOT / 'optuna').glob('best_params_*.json'))\n"
            "core_type, best_payload, best_path = pick_best_family(best_param_paths)\n"
            "base_config = ExperimentConfig(**json.loads(best_payload['best_user_attrs']['config']))\n"
            "base_config.CORE_TYPE = core_type\n"
            "base_config.MODEL_TAG = 'hidden_threshold_low_dim'\n\n"
            "reference_loss = float(best_payload['best_value'])\n\n"
            "threshold_results = run_hidden_threshold_grid_progressive(\n"
            "    data_bundle=data_bundle,\n"
            "    base_config=base_config,\n"
            "    hidden_grid=HIDDEN_GRID,\n"
            "    seeds=SWEEP_SEEDS,\n"
            "    artifact_root=ARTIFACT_ROOT,\n"
            "    run_prefix=RUN_PREFIX,\n"
            "    tolerance_ratio=TOLERANCE_RATIO,\n"
            "    reference_loss=reference_loss,\n"
            "    verbose=HIDDEN_VERBOSE,\n"
            ")\n\n"
            "results_df = pd.DataFrame(threshold_results['results_df']).copy()\n"
            "reference_rows = pd.DataFrame()\n"
            "if REFERENCE_RESULTS_PATH.exists():\n"
            "    reference_rows = pd.read_csv(REFERENCE_RESULTS_PATH)\n"
            "    reference_rows = reference_rows.loc[reference_rows['n_hidden'] == REFERENCE_HIDDEN].copy()\n"
            "    if not reference_rows.empty:\n"
            "        if 'device' not in reference_rows.columns:\n"
            "            reference_rows['device'] = 'mps'\n"
            "        reference_rows['reference_tag'] = 'reused_h4_reference'\n\n"
            "results_with_reference = pd.concat([results_df, reference_rows], ignore_index=True)\n"
            "if not results_with_reference.empty:\n"
            "    results_with_reference = (\n"
            "        results_with_reference\n"
            "        .sort_values(['n_hidden', 'seed', 'best_val_loss'])\n"
            "        .drop_duplicates(subset=['n_hidden', 'seed'], keep='first')\n"
            "        .reset_index(drop=True)\n"
            "    )\n\n"
            "comparison_results_path = ARTIFACT_ROOT / 'final' / f'{RUN_PREFIX}_with_reference_results.csv'\n"
            "results_with_reference.to_csv(comparison_results_path, index=False)\n"
            "comparison_figure_path, comparison_threshold_summary = plot_hidden_threshold(\n"
            "    results_df=results_with_reference,\n"
            "    save_path=ARTIFACT_ROOT / 'figures' / f'{RUN_PREFIX}_with_reference_threshold.png',\n"
            "    tolerance_ratio=TOLERANCE_RATIO,\n"
            "    reference_loss=reference_loss,\n"
            "    require_plateau=False,\n"
            ")\n\n"
            "comparison_grouped = comparison_threshold_summary['grouped_results'].copy()\n"
            "baseline_row = comparison_grouped.loc[comparison_grouped['n_hidden'] == REFERENCE_HIDDEN]\n"
            "baseline_loss = float(baseline_row['mean_val_loss'].iloc[0]) if not baseline_row.empty else np.nan\n"
            "comparison_grouped['delta_from_prev'] = comparison_grouped['mean_val_loss'].diff()\n"
            "comparison_grouped['ratio_to_prev'] = comparison_grouped['mean_val_loss'] / comparison_grouped['mean_val_loss'].shift(1)\n"
            "comparison_grouped['ratio_to_h4'] = (\n"
            "    comparison_grouped['mean_val_loss'] / baseline_loss if np.isfinite(baseline_loss) else np.nan\n"
            ")\n"
            "comparison_grouped_path = ARTIFACT_ROOT / 'final' / f'{RUN_PREFIX}_with_reference_grouped_results.csv'\n"
            "comparison_grouped.to_csv(comparison_grouped_path, index=False)\n"
            "comparison_summary_path = ARTIFACT_ROOT / 'final' / f'{RUN_PREFIX}_with_reference_threshold.json'\n"
            "write_json(\n"
            "    comparison_summary_path,\n"
            "    {\n"
            "        'device': DEVICE,\n"
            "        'core_type': core_type,\n"
            "        'hidden_grid': HIDDEN_GRID,\n"
            "        'reference_hidden': REFERENCE_HIDDEN,\n"
            "        'reference_results_path': str(REFERENCE_RESULTS_PATH),\n"
            "        'comparison_results_path': str(comparison_results_path),\n"
            "        'comparison_grouped_path': str(comparison_grouped_path),\n"
            "        'comparison_figure_path': str(comparison_figure_path),\n"
            "        'threshold_limit': comparison_threshold_summary['threshold_limit'],\n"
            "        'selected_hidden': comparison_threshold_summary['selected_hidden'],\n"
            "        'threshold_found': comparison_threshold_summary['threshold_found'],\n"
            "        'acceptable_hidden': comparison_threshold_summary['acceptable_hidden'],\n"
            "    },\n"
            ")\n\n"
            "selected_hidden = comparison_threshold_summary['selected_hidden']\n"
            "final_result = None\n"
            "if RUN_FINAL_RETRAIN and selected_hidden is not None:\n"
            "    fixed_epochs = int(max(base_config.MIN_EPOCHS, round(float(comparison_grouped.loc[comparison_grouped['n_hidden'] == selected_hidden, 'median_best_epoch'].iloc[0]))))\n"
            "    final_result = retrain_final_model(\n"
            "        data_bundle=data_bundle,\n"
            "        base_config=base_config,\n"
            "        n_hidden=selected_hidden,\n"
            "        fixed_epochs=fixed_epochs,\n"
            "        artifact_root=ARTIFACT_ROOT,\n"
            "        run_name=FINAL_RUN_NAME,\n"
            "    )\n\n"
            "results_with_reference"
        ),
        nbf.v4.new_code_cell(
            "pd.DataFrame(threshold_results['stage_rows'])"
        ),
        nbf.v4.new_code_cell(
            "display(comparison_grouped)\n"
            "display(Image(filename=str(comparison_figure_path)))\n"
            "comparison_threshold_summary"
        ),
        nbf.v4.new_code_cell(
            "final_result"
        ),
    ]
    write_notebook(ROOT / "03_final_training_and_hidden_threshold.ipynb", cells)


def build_04_inference_and_testing() -> None:
    cells = [
        nbf.v4.new_markdown_cell(
            "# 04 Inference And Testing\n\n"
            "Evaluate the final checkpoint on the untouched test set and run qualitative unseen-load inference."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n\n"
            "ARTIFACT_ROOT = Path('artifacts')\n"
            "SEED = 20260328\n"
            "USE_OPTUNA_BEST = True\n"
            "CHECKPOINT_PATH = ARTIFACT_ROOT / 'checkpoints' / 'final_model.pt'\n\n"
            "if USE_OPTUNA_BEST:\n"
            "    best_param_paths = sorted((ARTIFACT_ROOT / 'optuna').glob('best_params_*.json'))\n"
            "    best_payload = None\n"
            "    for path in best_param_paths:\n"
            "        payload = json.loads(path.read_text())\n"
            "        if best_payload is None or payload['best_value'] < best_payload['best_value']:\n"
            "            best_payload = payload\n"
            "    if best_payload is None:\n"
            "        raise FileNotFoundError('No Optuna best-parameter files found.')\n"
            "    CHECKPOINT_PATH = Path(best_payload['best_user_attrs']['checkpoint_path'])\n"
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "from Coursework3.RNO_1D_Skeleton import run_inference_and_testing\n\n"
            "inference_results = run_inference_and_testing(\n"
            "    checkpoint_path=CHECKPOINT_PATH,\n"
            "    artifact_root=ARTIFACT_ROOT,\n"
            "    split_seed=SEED,\n"
            ")\n\n"
            "pd.DataFrame([inference_results['metrics']])"
        ),
        nbf.v4.new_code_cell(
            "for figure_path in [inference_results['residual_plot_path'], inference_results['unseen_plot_path']]:\n"
            "    display(Image(filename=str(figure_path)))"
        ),
        nbf.v4.new_code_cell("inference_results"),
    ]
    write_notebook(ROOT / "04_inference_and_testing.ipynb", cells)


def build_05_trajectory_and_hysteresis() -> None:
    cells = [
        nbf.v4.new_markdown_cell(
            "# 05 Trajectory And Hysteresis Analysis\n\n"
            "Compare true and predicted stress-strain trajectories on the test set for the best, median, and worst samples, and inspect hysteresis under cyclic loading."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n\n"
            "ARTIFACT_ROOT = Path('artifacts')\n"
            "SEED = 20260328\n"
            "USE_OPTUNA_BEST = True\n"
            "CHECKPOINT_PATH = ARTIFACT_ROOT / 'checkpoints' / 'final_model.pt'\n\n"
            "if USE_OPTUNA_BEST:\n"
            "    best_param_paths = sorted((ARTIFACT_ROOT / 'optuna').glob('best_params_*.json'))\n"
            "    best_payload = None\n"
            "    for path in best_param_paths:\n"
            "        payload = json.loads(path.read_text())\n"
            "        if best_payload is None or payload['best_value'] < best_payload['best_value']:\n"
            "            best_payload = payload\n"
            "    if best_payload is None:\n"
            "        raise FileNotFoundError('No Optuna best-parameter files found.')\n"
            "    CHECKPOINT_PATH = Path(best_payload['best_user_attrs']['checkpoint_path'])\n"
        ),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n\n"
            "from Coursework3.RNO_1D_Skeleton import run_trajectory_and_hysteresis_analysis\n\n"
            "analysis_results = run_trajectory_and_hysteresis_analysis(\n"
            "    checkpoint_path=CHECKPOINT_PATH,\n"
            "    artifact_root=ARTIFACT_ROOT,\n"
            "    split_seed=SEED,\n"
            ")\n\n"
            "analysis_results['example_df']"
        ),
        nbf.v4.new_code_cell(
            "for figure_path in [analysis_results['stress_strain_plot_path'], analysis_results['stress_time_plot_path'], analysis_results['hysteresis_plot_path']]:\n"
            "    display(Image(filename=str(figure_path)))"
        ),
        nbf.v4.new_code_cell("analysis_results"),
    ]
    write_notebook(ROOT / "05_trajectory_and_hysteresis_analysis.ipynb", cells)


def main() -> None:
    build_00_environment_and_mps()
    build_01_eda_and_preprocessing()
    build_03_final_training_and_hidden_threshold()
    build_04_inference_and_testing()
    build_05_trajectory_and_hysteresis()


if __name__ == "__main__":
    main()
