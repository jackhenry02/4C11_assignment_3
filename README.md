# 4C11 Assignment 3

Skeleton-style recurrent neural operator workflow for the viscoelastic coursework dataset.

## Main Entry Points

- `00_environment_and_mps.ipynb`: runtime and MPS validation
- `01_eda_and_preprocessing.ipynb`: EDA, split generation, normalization, saved figures
- `02_optuna_search.py`: per-family Optuna search for `rnn`, `gru`, and `lstm`
- `03_final_training_and_hidden_threshold.ipynb`: hidden-size sweep and final retraining
- `04_inference_and_testing.ipynb`: held-out test evaluation and unseen-load inference

## Core Module

- `Coursework3/RNO_1D_Skeleton.py`: reusable implementation that keeps the skeleton naming and overall code shape

## Saved Outputs

Generated artifacts are written under `artifacts/` and ignored by git.
