# Coursework Plan: Skeleton-Style Recurrent Operator Workflow With Tuned Hidden Size

## Summary
- Implement the full coursework as a skeleton-style local package plus notebooks, keeping the original variable names and code feel wherever practical.
- Use four main notebooks and one Optuna script:
  - `00_environment_and_mps.ipynb`
  - `01_eda_and_preprocessing.ipynb`
  - `02_optuna_search.py`
  - `03_final_training_and_hidden_threshold.ipynb`
  - `04_inference_and_testing.ipynb`
- Keep the recurrent-operator framing central, with three recurrent-core variants inside the same `RNO`-style interface: simple RNN, GRU, and LSTM.
- Use a fixed `70/15/15` train/validation/test split, full sequence length `1001`, fixed seeds, train-only normalization, saved checkpoints, and aggressive result persistence.
- Revise the earlier plan so `n_hidden` is included in Optuna tuning for model-family comparison, then do a separate explicit hidden-threshold sweep for part (c).

## Key Implementation Changes
- Create one reusable coursework module that preserves the skeleton naming/style:
  - keep `MatReader`, `RNO`, `TRAIN_PATH`, `F_FIELD`, `SIG_FIELD`, `Ntotal`, `train_size`, `test_start`, `loss_func`, `layer_input`, `layer_hidden`, `net`, `optimizer`, `scheduler`, `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, `y_test`
  - add only the minimum extra names needed for validation and device handling
- Keep the model interface skeleton-like:
  - input sequence: macroscopic strain history
  - output sequence: macroscopic stress history
  - explicit time stepping and hidden-state carry
  - per-step features remain strain, previous strain, and discrete strain rate
  - recurrent core chosen by a `core_type` setting inside the same overall operator structure
- Standardize training behavior across all families:
  - train-set min-max normalization to `[-1, 1]`
  - normalized `MSE` as `loss_func`
  - report original-scale `RMSE`, `MAE`, `R²`, normalized RMSE, and residual diagnostics
  - `AdamW`, `ReduceLROnPlateau`, gradient clipping, checkpoint-on-improvement, and long-patience early stopping
  - full seed control for `random`, `numpy`, and `torch`
- Device policy:
  - implement auto-selection as `mps -> cuda -> cpu`
  - add a dedicated environment notebook that validates whether MPS actually works in this kernel before any long run
  - keep all training code device-safe and able to fall back cleanly if MPS remains unavailable
- Optuna policy:
  - `02_optuna_search.py` runs one capped study per family with equal budget
  - tune `n_hidden` jointly with learning rate, weight decay, batch size, feedforward width/depth, and gradient-clip value
  - store studies in SQLite, save all trial results to CSV, and save best params per family to JSON
  - do not use notebooks as the source of truth for long Optuna runs
- Part (c) policy:
  - after selecting the winning family from Optuna, freeze all non-hidden hyperparameters to that family’s best trial values
  - run an explicit hidden-size sweep in `03_final_training_and_hidden_threshold.ipynb`
  - evaluate each hidden size over multiple seeds
  - define the minimum sufficient hidden size as the smallest `n_hidden` within `5%` of the best mean validation error and on the stable plateau
  - retrain that final chosen hidden size once on `train + validation`, then evaluate once on test
- Result persistence:
  - save split indices
  - save normalization metadata
  - save checkpoints for best epochs
  - save epoch logs and metrics to CSV/JSON
  - save Optuna study DB plus per-trial exports
  - save predictions/residual arrays for later plotting
  - save figures to a stable results directory so nothing depends on rerunning everything

## Notebook And Script Breakdown
- `00_environment_and_mps.ipynb`
  - print interpreter, PyTorch version, device availability, seeds, and selected device
  - validate MPS support for the exact ops used by the coursework
  - check data file presence and shape
  - set project-local writable cache locations for Matplotlib if needed
- `01_eda_and_preprocessing.ipynb`
  - inspect dataset integrity and sample trajectories
  - compare raw, min-max, and z-score scaling
  - inspect distributions, rates, representative histories, and strain/stress relationships
  - run ACF/PACF and limited stationarity checks as exploratory diagnostics
  - save the chosen split indices and normalization parameters
- `02_optuna_search.py`
  - load saved split and normalization artifacts
  - run per-family Optuna studies
  - save SQLite studies, best-params JSON, and trial tables
  - save intermediate summaries so overnight runs are recoverable
- `03_final_training_and_hidden_threshold.ipynb`
  - load best family and best non-hidden hyperparameters from saved Optuna outputs
  - train the selected family cleanly with full logging
  - run the hidden-size sweep for part (c)
  - generate the threshold plot and select the minimum sufficient hidden size
- `04_inference_and_testing.ipynb`
  - evaluate on untouched test data
  - generate residual analysis and comparison plots
  - run hand-crafted unseen strain histories
  - save report-ready figures and tables

## Dependencies And Public Interfaces
- Add the minimal extra dependencies needed for the requested workflow:
  - `statsmodels` for PACF and residual autocorrelation
  - `nbformat`, `nbconvert`, and `nbclient` so `.ipynb` notebooks are executable from the venv
- Keep the main public training interface simple:
  - one config object or top-of-file ALL_CAPS settings per notebook/script
  - one model-construction entrypoint taking `core_type` and `n_hidden`
  - one training entrypoint returning saved metrics/checkpoint paths
- Keep the Optuna outputs stable and explicit:
  - `best_params_<family>.json`
  - `trials_<family>.csv`
  - `study_<family>.db`

## Test Plan
- Verify the loader reads `epsi_tol` and `sigma_tol` with shape `(400, 1001)` and that the saved split is deterministic.
- Verify normalization and inverse-normalization are fit on train only and reproduce original values correctly.
- Verify RNN-, GRU-, and LSTM-core variants each complete a forward pass, backward pass, checkpoint save/load, and full-sequence inference on the selected device.
- Verify the MPS validation notebook checks the exact training ops before any heavy run.
- Verify `02_optuna_search.py` can resume from saved SQLite state and that trial exports and best-params JSON files are produced.
- Verify the hidden-size sweep notebook reproduces a threshold plot with mean and variability across seeds.
- Verify the final inference notebook reloads saved artifacts and regenerates the held-out metrics and unseen-load plots without retraining.

## Assumptions And Defaults
- The report narrative stays centered on recurrent neural operators; GRU and LSTM are presented as gated recurrent-core variants of the same operator structure.
- Full sequence length `1001` is used throughout to preserve the long-memory setting and support the vanishing-gradient discussion.
- Model-family choice and hidden-threshold choice are based on validation only; the test set is touched only at the end.
- Stationarity analysis is exploratory rather than a hard modeling assumption.
- The first execution phase after implementation will run only `00_environment_and_mps.ipynb` and `01_eda_and_preprocessing.ipynb`; the Optuna script remains deferred until you explicitly allow the overnight run.
