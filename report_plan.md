# LaTeX Report Plan: Investigative Search-To-Physics Narrative

## Summary
The report will be a long-form LaTeX article with an introduction but no formal abstract by default. It will stay loosely aligned with coursework parts `(a)`, `(b)`, and `(c)`, but the main narrative will be:

1. define the macroscopic constitutive problem and answer `(a)`,
2. show the architecture search across `RNN`, `GRU`, and `LSTM`,
3. establish `GRU` as the best predictive model,
4. then treat part `(c)` as a structured investigation into hidden/internal variables using progressively more physics-shaped models,
5. finish with qualitative behavior and hidden-state geometry.

The core report thesis will be:
- `GRU` is the best predictive surrogate on this dataset,
- the threshold in hidden variables is not sharply exposed by every architecture,
- the operator-style follow-up reveals why: explicit strain-rate input can make `h=0` surprisingly strong, while the no-rate paper-style model exposes the clearest failure of zero latent memory.

## Report Structure
### 1. Introduction
- Open with the macroscopic constitutive-learning problem for a 3-phase viscoelastic composite.
- Answer part `(a)` explicitly here:
  - input = macroscopic strain history, or recurrently `(epsilon_n, epsilon_{n-1}, dot epsilon_n)`
  - output = macroscopic stress history
- State what the dataset contains and does not contain:
  - only macroscopic strain/stress histories
  - not microscopic fields or direct microscopic parameter labels
- Include the brief erratum interpretation:
  - the viscous term is taken as strain-rate based, not displacement-rate based
- End with a short roadmap sentence:
  - data-driven search first, physics-shaped follow-up second

### 2. Problem Setup And Data
- Present the unit-cell setting and the micro-to-macro motivation.
- Describe the dataset:
  - `400` samples
  - `1001` time points per sample
  - deterministic `70/15/15` split
- Describe preprocessing:
  - train-only min-max normalization
  - full histories retained
- Keep this section brief and notation-focused.

### 3. Method I: Cell-Based Recurrent Models
- Present `RNN`, `GRU`, and `LSTM` as one methodology family.
- Explain the shared constitutive interface:
  - stepwise features built from current strain, previous strain, and discrete strain rate
  - hidden state used as learned memory
- Describe training:
  - normalized MSE
  - AdamW
  - ReduceLROnPlateau
  - early stopping
  - checkpoint on best validation loss
- Explain Optuna clearly and cite it.
- State exactly what Optuna tuned.

### 4. Method II: Operator-Inspired Recurrent Models
- Give this a separate methods section.
- Introduce:
  - baseline RNO: close to the coursework skeleton
  - paper-inspired RNO: closer to Liu et al.
- Explain the conceptual contrast with Method I:
  - gated statistical memory vs explicit hidden-state evolution
- For the paper-inspired model, define both cases:
  - no explicit strain rate in the stress readout
  - with explicit strain rate in the stress readout
- Cite the Liu paper here as the motivation for the internal-variable framing and later dimensionality-reduction discussion.

### 5. Results I: Architecture Search And Best Predictive Model
- Lead with the Optuna comparison across `RNN`, `GRU`, and `LSTM`.
- Show that `GRU` is best on validation loss and final test metrics.
- Include one compact quantitative comparison table.
- Discuss briefly why `LSTM` did not outperform `GRU` in practice.
- Mention CPU vs MPS only in passing:
  - final reported runs are CPU-based because the explicit stepwise loop was faster there in practice.

### 6. Results II: Investigating The Hidden-Variable Threshold
This section should be a visible progression, not a single final plot.

#### 6.1 Initial Hidden Sweep On The Best Cell-Based Model
- Show that the low-dimensional `GRU` sweep did not reveal a dramatic cliff.
- Use this as the motivation for trying more physically interpretable architectures.

#### 6.2 Baseline RNO Follow-Up
- Show the baseline RNO as a skeleton-faithful comparator.
- Use it to argue that architecture choice affects how visible the threshold is.

#### 6.3 Paper-Style `h=0` Ablation
- Present the strongest conceptual result:
  - `paper_rno_no_rate, h=0` fails badly
  - `paper_rno_with_rate, h=0` is much stronger
- Interpret this as evidence that explicit rate input carries a large fraction of the constitutive information.

#### 6.4 Full Paper-RNO Hidden Sweeps
- Show both no-rate and with-rate hidden-size curves.
- Main message:
  - no-rate gives the clearest threshold-like behavior
  - with-rate is much more accurate, but the hidden-size transition is smoother

#### 6.5 Final Part (c) Answer
- State the conclusion carefully:
  - no single sharp universal threshold was found across all architectures
  - the clearest operator-style no-rate study suggests a minimum hidden dimension of about `3`
  - once explicit rate information is included, the dependence on hidden size becomes much less abrupt
- Phrase this as an architecture- and input-dependent conclusion, not an overgeneralized physical truth.

### 7. Results III: Qualitative Behaviour And Latent-State Interpretation
- Use the best/median/worst stress-time and stress-strain plots to show predictive behavior directly.
- Use the hysteresis figure for cyclic loading.
- Then bring in the hidden-state geometry analysis:
  - PCA explained variance
  - optionally one correlation heatmap
- Main discussion point:
  - high-performing models often use large but highly redundant latent spaces
  - many hidden channels collapse effectively onto about two dominant directions
- Use this as supporting evidence only, not the sole proof for part `(c)`.

### 8. Discussion And Conclusion
- Synthesize the findings rather than restating plots.
- Final discussion points:
  - `GRU` is the strongest predictive model
  - operator-style models are more useful for interrogating internal variables
  - explicit strain-rate input is a major reason why some `h=0` cases still perform well
  - hidden-state dimensionality is often effectively lower than the raw hidden size
- End by answering `(a)`, `(b)`, and `(c)` explicitly in prose.

## Figures And Tables
### Main Figures To Include
Use a selective set, with subfigures where possible:

1. Dataset/sample histories:
- [01_sample_histories.png](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/figures/01_sample_histories.png)

2. Architecture search:
- [02_family_best_validation_loss.png](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/figures/optuna/02_family_best_validation_loss.png)
- [02_best_optuna_training_curves.png](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/figures/optuna/02_best_optuna_training_curves.png)

3. Threshold investigation progression:
- [hidden_threshold_low_dim_with_reference_threshold.png](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/figures/hidden_threshold_low_dim_with_reference_threshold.png)
- [08_paper_rno_h0_comparison.png](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/figures/08_paper_rno_h0_comparison.png)
- [09_paper_rno_validation_loss_vs_hidden.png](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/figures/09_paper_rno_validation_loss_vs_hidden.png)
- [09_paper_rno_test_relative_l2_vs_hidden.png](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/figures/09_paper_rno_test_relative_l2_vs_hidden.png)

4. Qualitative behavior:
- [05_test_stress_time_examples.png](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/figures/05_test_stress_time_examples.png)
- [05_test_stress_strain_examples.png](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/figures/05_test_stress_strain_examples.png)
- [05_hysteresis_checks.png](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/figures/05_hysteresis_checks.png)

5. Hidden-state interpretation:
- [11_hidden_state_pca_explained_variance.png](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/figures/11_hidden_state_pca_explained_variance.png)

### Optional Main-Body Figure
Include only if it helps the discussion stay concrete:
- [11_hidden_state_correlation_heatmaps.png](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/figures/11_hidden_state_correlation_heatmaps.png)

### Tables To Build
- Dataset and split summary table
- Best-model comparison table using:
  - [family_comparison.csv](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/optuna/family_comparison.csv)
  - [06_baseline_rno_comparison.csv](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/reports/06_baseline_rno_comparison.csv)
- Paper `h=0` ablation table from:
  - [08_paper_rno_h0_comparison.csv](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/reports/08_paper_rno_h0_comparison.csv)
- Hidden-state geometry table from:
  - [11_hidden_state_geometry_summary.csv](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/artifacts/reports/11_hidden_state_geometry_summary.csv)

### Omit From Main Body
- CPU/MPS benchmark figures
- loading-case figures from `10`
- most Optuna slice/runtime/importance plots
- residual plots for every model
- intermediate stage plots from progressive sweeps

## LaTeX Setup And Citation Plan
- Use:
  - `report.tex`
  - `references.bib`
  - standard `article` class unless an existing course template is imposed later
- Use BibTeX and numeric citations.
- Required references:
  - coursework brief
  - Optuna paper
  - Liu et al. paper
  - canonical `RNN`, `GRU`, and `LSTM` citations if those architectures are described explicitly
- The introduction and methods should cite the theory only where it helps structure the argument, not as a literature review for its own sake.

## Writing Rules
- No formal abstract by default.
- Use an introduction instead of a generic “Background” dump.
- Keep the report selective and narrative-driven, not notebook-like.
- Use negative or inconclusive threshold results explicitly as evidence:
  - not as mistakes
  - but as part of the scientific progression
- Do not claim that PCA or correlation proves the true number of physical internal variables.
- Use the hidden-state geometry as supporting interpretation only.

## Assumptions And Defaults
- The report is allowed to be long and detailed, rather than compressed to a short page target.
- The main framing is “architecture search first, physics-grounded follow-up second.”
- The operator-style models remain a full second methods section.
- Hidden-state/PCA analysis is part of the main discussion.
- Loading-case analysis from `10` stays out of the main body.
- CPU vs MPS appears only as a brief implementation note.
- Part `(c)` is presented as a structured investigation whose conclusion depends on architecture and explicit rate input, not as a single universal threshold.
