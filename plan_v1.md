# Coursework Plan: Skeleton-Style Recurrent Operator Workflow With MPS Validation

## Summary
- Keep the implementation close to the provided skeleton: explicit time stepping, simple PyTorch modules, readable code, and notebook-first workflow.
- Use a shared local Python module for reusable code plus five notebooks: `00` environment/MPS validation, `01` EDA/preprocessing, `02` Optuna search, `03` final training + hidden-threshold analysis, `04` inference + unseen-load testing.
- Compare three operator variants that share the same outer structure and training pipeline, differing only in recurrent core: simple RNN, GRU, and LSTM.
- Use a fixed sample-level `70/15/15` train/validation/test split, full sequence length `1001` throughout, fixed seeds, train-only normalization, and validation-only model selection.
- Treat MPS as a prerequisite to troubleshoot first, because the current runtime is Apple Silicon/macOS 15.4.1 with PyTorch 2.11.0 and `mps.is_built() = True` but `mps.is_available() = False`.

## Implementation Changes
- Build one reusable coursework module derived from the skeleton that exposes:
  - dataset loading
  - deterministic split generation
  - normalization/inverse normalization
  - model construction by `core_type in {"rnn","gru","lstm"}`
  - training/evaluation loops
  - checkpoint save/load
  - unseen strain-history generators
- Preserve the skeleton’s style:
  - explicit per-time-step recurrence
  - manual hidden-state initialization
  - simple feedforward readout around the recurrent state
  - no high-level training framework
- Standardize model I/O across all variants:
  - input sequence: macro strain history `epsilon(t)`
  - per-step features: `epsilon_t`, `epsilon_{t-1}`, and strain rate `(epsilon_t - epsilon_{t-1}) / dt`
  - output sequence: macro stress history `sigma(t)`
  - only the recurrent core changes between RNN/GRU/LSTM variants
- Use train-set min-max scaling to `[-1, 1]` for both strain and stress. Show alternatives in EDA, but lock the training pipeline to min-max for consistency with the skeleton and symmetric data range.
- Use normalized-space `MSE` for optimization. Report original-scale `RMSE`, `MAE`, `R²`, normalized RMSE, and residual diagnostics.
- Use `AdamW`, `ReduceLROnPlateau`, gradient clipping, best-checkpoint saving on every validation improvement, and separate long-patience early stopping.
- Keep hidden size out of Optuna. Use a fixed reference hidden size of `16` during model-family comparison so part (b) and part (c) stay separate.
- Run Optuna as a capped, equal-budget study per family on train/validation only, stored in SQLite for dashboard use. Tune shared hyperparameters only: learning rate, weight decay, batch size, feedforward width/depth, dropout if used, and gradient-clip value.
- Define the hidden-variable threshold for part (c) using the chosen family and fixed tuned training recipe:
  - sweep hidden sizes over a predeclared grid
  - use at least three seeds per hidden size
  - choose the smallest hidden size whose mean validation error is within `5%` of the best mean validation error and remains on that plateau for larger sizes
  - retrain that chosen size once on `train + validation`, then evaluate once on test
- Unseen-load testing stays qualitative because there is no reference solver in the repo:
  - monotonic ramp
  - ramp-hold-relaxation
  - triangular cyclic load
  - sinusoid with changed amplitude/frequency
  - discuss physical plausibility, rate sensitivity, hysteresis, smoothness, and symmetry

## MPS Validation And Device Policy
- Add a dedicated `00` notebook that runs before any heavy training and answers one question only: can this exact kernel use MPS safely for this coursework workflow?
- In that notebook, check:
  - `torch.backends.mps.is_built()`
  - `torch.backends.mps.is_available()`
  - forward/backward/optimizer-step smoke tests on MPS for `Linear`, `SELU`, `MSELoss`, gradient clipping, `AdamW`, `RNN`, `GRU`, and `LSTM`
  - one short explicit operator-style loop on MPS
- If all smoke tests pass in the notebook kernel, run training on MPS.
- If MPS is still unavailable, stop and fix the environment before long runs. The plan should include common checks only: kernel interpreter matches `.venv`, PyTorch is the arm64 wheel, no Rosetta mismatch, and notebook/kernel restart after reinstall.
- Keep runtime device selection implemented as `mps -> cuda -> cpu`, but the workflow assumes MPS should be working locally before Optuna or final sweeps begin.
- Add one small environment cell at the top of every training notebook that prints the selected device and seed.
- Set `MPLCONFIGDIR` to a writable project-local temp/cache path inside notebooks/scripts to avoid Matplotlib cache warnings on macOS.

## Test Plan
- Verify the loader reads `epsi_tol` and `sigma_tol` with shape `(400, 1001)` and that deterministic splits reproduce exactly.
- Verify normalization/inverse-normalization are fit on train only and reconstruct original values correctly.
- Verify each model variant can run one forward pass, one training epoch, checkpoint save/load, and full-sequence inference on the selected device.
- Verify the MPS smoke tests pass before heavy training on Mac; if they fail, do not proceed with MPS runs.
- Verify Optuna studies are reproducible from seed/config, stored in SQLite, and comparable across families.
- Verify the hidden-size sweep produces a threshold plot with mean and variability across seeds.
- Verify the final inference notebook reloads the chosen checkpoint, reproduces held-out test metrics, and generates all unseen-load plots.

## Assumptions And Defaults
- The coursework narrative remains centered on a recurrent neural operator; GRU/LSTM are framed as gated recurrent-core variants of the same operator architecture.
- Full sequence length `1001` is used everywhere to preserve the long-memory setting and support the vanishing-gradient discussion.
- Model-family choice and hidden-threshold choice are made from validation only; the test set is touched once at the end.
- `statsmodels` will be added for PACF and residual autocorrelation diagnostics.
- Stationarity analysis is exploratory only, since these are driven constitutive-response sequences rather than stationary free-running series.
