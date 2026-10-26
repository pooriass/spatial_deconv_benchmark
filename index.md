---
title: A held-out synthetic-spot benchmark for spatial deconvolution
---

# A held-out synthetic-spot benchmark for spatial deconvolution

This page describes a test of how well a deconvolution method (here DestVI) recovers cell *states* on your own Visium tissue. Run it before you build an analysis on those states. Code is in [`code/`](code/). The page holds no data and no results from any specific dataset.

## The question it answers

Published benchmarks rank methods by average performance. They cannot tell you whether one state in *your* tissue can be separated from its neighbours, given *your* sequencing depth and *your* reference. The answer depends on how similar the states are, how many cells each state has, and how deep the spots are. This benchmark measures it directly, using a ground truth you control.

## The idea in brief

Hide some patients from your single-cell reference and train the reference model on the rest. Then build fake Visium spots from the hidden cells, so the cell content of every spot is known. Run the spatial model on the fake spots and compare its output with the truth, state by state.

## Steps

1. **Split by patient.** Hold out one fifth of the patients (at least one). If any state you want to score has fewer than 25 cells on either side of the split, the script falls back to a 50/50 split inside each patient and records this in the output. The fallback is a weaker test, since the same patients appear in training and testing.
2. **Train the reference model** (CondSCVI) on the training cells only. The fine states are the labels it learns, and a broad cell-type column serves as the coarse label.
3. **Build fake spots from the held-out cells.** Each spot sums the counts of 2 to 10 cells. A binomial draw then thins the counts to a target depth, taken from a normal distribution whose mean and SD you set to match your real sections (floor of 500 UMI). The procedure follows the synthspot idea. Spots come in 5 regions of 50 to 200 spots each, and every region has its own composition. The truth for a spot is the fraction of its *cells* in each state.
4. **Two spot regimes.** In the *overlap* regime, region compositions come from a flat Dirichlet, so spots mix many states. In the *dominant* regime, one randomly chosen state takes about three quarters of the weight in a region, so spots are mostly one state. A state has to work in both.
5. **Fit DestVI** on each regime and read out the per-state proportions, row-normalised.
6. **Repeat per setting.** A setting is one gene space (for example a probe-based panel or a poly-A panel) combined with one label resolution (fine or coarse). The code accepts any number of settings.
7. **Score** the predictions (next section).

## Metrics

Every metric is computed per state, per setting and per regime. The decision uses the worse of the two regimes.

| Metric | What it tells you |
|---|---|
| Pearson r, Spearman rho | Does the predicted amount of a state rise and fall with the true amount across spots? Both ignore scale and offset. |
| Lin's concordance correlation (CCC) | Same question as Pearson, but a wrong scale or offset also lowers the score. |
| Calibration slope and intercept | How the prediction scales with the truth. The truth is a *cell* fraction and the prediction comes from expression, so a slope away from 1 can partly reflect mRNA content. |
| RMSE and MAE, against a floor | The floor is a prediction of the state's mean abundance in every spot. |
| AUROC, AUPRC | Can the method separate spots that contain the state from spots that do not? |
| False-positive weight | Mean predicted weight in spots where the state is truly absent, set against the state's own mean abundance. |
| Leakage matrix and identifiability | Correlation of every true state with every predicted state. A state is *identifiable* if its own predicted column is its best match. The comparison happens inside one row, so scale cancels. |
| Within-compartment share agreement | Inside a compartment (for example all fibroblast states), does the *fraction* belonging to each state match the truth? |
| Within-spot Jensen-Shannon divergence (JSD), top-1 accuracy | Does the method get the whole mix inside a spot right? The floor is a prediction of the global mean mix. |

## Decision rule

Fix the cutoff before you look at the results. In the code it is `CUT` (default 0.20). A state passes in a setting if, in both regimes, it meets all five conditions:

1. It is identifiable (the diagonal is the row maximum of the leakage matrix).
2. Worst-case Pearson r and worst-case Lin's CCC are both at least `CUT`.
3. RMSE beats the floor.
4. False-positive weight is below the state's own mean abundance.
5. The within-compartment mix (JSD) beats the floor.

`PASS_absolute_only` applies criteria 1 to 4. It asks a narrower question: can the *amount* of this state per spot be recovered, even if the split between sibling states cannot? AUPRC and share agreement are reported but do not enter the pass rule.

If a compartment has only one state at a setting, criterion 5 cannot be evaluated and the state cannot reach `PASS`. Use `PASS_absolute_only` there.

**The cutoff is a judgement call.** No published deconvolution benchmark sets a pass/fail line. They rank methods on several metrics together. Choose `CUT` for your own question, write it down first, and report how sensitive your conclusion is to it. Identifiability is the easier criterion to defend, since it does not depend on scale.

## What a result can and cannot tell you

- **The truth is synthetic.** The fake spots come from the same reference family the model learned from, in the same gene space. They carry no probe chemistry effects and no RNA diffusion between spots, so the test is optimistic. If your fake spots are shallower than real spots, the test turns pessimistic. Compare the median UMI of the fake spots (the script prints it) with your real data. The net direction of the error is unknown.
- **A failure is informative.** A state that cannot be recovered under these friendly conditions will not be recovered on real tissue. A pass only shows that the state clears the first bar.
- **Across-spot correlation and within-spot divergence answer different questions.** Correlation asks whether the pattern of a state across the tissue follows the truth. The divergence asks whether the mix inside each spot is right. Report both.
- **Metrics depend on the variance of the constructed spots.** If you change the number of regions or the Dirichlet settings, the correlations move.
- **Spot-based methods only.** DestVI cannot run on single cells, so the design does not apply to imaging data.

## How to run it

Install the packages in [`code/requirements.txt`](code/requirements.txt). The file pins `scvi-tools==1.5.1` because the script patches two bugs in that version's `DestVI.from_rna_model`: a missing `prior` key, and a numpy-versus-tensor error in the VampPrior step. The installed package itself is left unmodified.

**Input.** For each gene space, create a folder `$DECONV_ROOT/export/<TAG>/` with these files:

- `counts.mtx.gz`: raw counts, genes by cells, Matrix Market format
- `genes.txt`, `cells.txt`: one name per line, in matrix order
- `meta.csv`: index = cell names. Columns are `patient`, `cell_type` (broad label), `state` (fine states), and optionally `state_coarse` (coarse states).

State names should start with a compartment prefix (default `CAF_`, `DUCTAL_`). Change it with `IN_SCOPE_PREFIX` and `PREFIXES`.

**Environment variables for `resolvability.py`.**

| Variable | Meaning | Default |
|---|---|---|
| `TAG` | name of the gene-space folder under `export/` | required |
| `STATE_COL` | column of `meta.csv` that holds the states to test | `state` |
| `SUF` | suffix for output names: empty for fine, `_coarse` for coarse | empty |
| `VIS_MEAN`, `VIS_SD` | mean and SD of UMI per spot in your real data | required |
| `SC_EPOCHS`, `ST_EPOCHS` | maximum epochs for the reference model and for DestVI | 300, 2000 |
| `RESOLV_CAP` | maximum cells per state kept from the reference, for speed | 3000 |
| `BATCH`, `NTHREADS` | batch size and CPU threads | 1024, 10 |
| `IN_SCOPE_PREFIX` | comma-separated state prefixes to score | `CAF_,DUCTAL_` |
| `DECONV_ROOT` | working directory | `.` |

See [`code/run_example.sh`](code/run_example.sh) for a complete driver. `score.py` reads `RES`, `OUTDIR`, `CONFIGS`, `PREFIXES` and `CUT`.

**Test the scorer without training anything.** `tests/make_toy_inputs.py` writes fake prediction and truth tables. Run it, then run `score.py` with `RES=toy_results CONFIGS="A:,A:_coarse"`. This checks the scoring code and the file formats only and says nothing about DestVI.

**Outputs.** `results/` holds one prediction table and one truth table per setting and regime, plus a small metadata file. `scores/` holds `decision_table.csv`, `metrics_per_state.csv`, `metrics_per_spot.csv`, `jsd_by_group.csv` and the leakage matrices.

## Seeds and runtime

The script fixes its seeds: numpy generator 0, scvi seed 0, spot generator seed 1. DestVI training on CPU dominates the runtime. Set `RESOLV_CAP` to limit the size of the reference. When you change only the spot settings, the script reuses a saved reference model if it finds one, so you skip retraining.
