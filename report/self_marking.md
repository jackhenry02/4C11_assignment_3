# Self Marking

## Initial Draft: `report_initial.tex`

### Provisional Mark

**86\%**

### Why It Was Not Higher

- The main narrative was strong, but the operational definition of the hidden-variable threshold was implicit rather than stated cleanly.
- The recurrent-cell subsection did not explicitly cite the canonical RNN, GRU, and LSTM sources.
- The part (c) discussion gave the conclusions, but the threshold studies were not yet summarised in one table.
- The report used a good set of figures, but some of the supplementary work remained implicit rather than visible on the page.
- The appendix material that helps show the full depth of investigation was missing.

### Improvements Chosen

1. Add an explicit subsection defining how ``fully learned'' was operationalised in the hidden-size studies.
2. Add a compact numerical table for the paper-style $h=0$ ablation.
3. Add a compact threshold-summary table across the main hidden-size studies.
4. Add citations for the standard recurrent architectures.
5. Add a short appendix with extra optimisation and hidden-state figures so the full investigation is visible.

---

## Revised Draft: `report_revised.tex`

### Provisional Mark

**92\%**

### Why The Mark Increased

- The threshold criterion is now explicitly defined and therefore much easier to defend.
- The part (c) section now reads as a controlled sequence of experiments rather than a set of isolated plots.
- The $h=0$ ablation and threshold-summary tables make the key conclusions faster to verify.
- The appendix now shows more of the work without cluttering the main narrative.
- The methods section is better anchored to the relevant literature.

### Remaining Risks

- The report still relies on one-seed paper-RNO sweeps, which should be acknowledged if questioned.
- The exact hidden-variable count remains architecture dependent, so the part (c) conclusion must stay carefully worded.
- The environment used here did not contain a TeX compiler, so the `.tex` files were not compiled locally in this session.
