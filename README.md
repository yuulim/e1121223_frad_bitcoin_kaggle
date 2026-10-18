# BT4012 reproducible fraud detection

Open `e1121223.ipynb`, reload it from disk, restart the kernel, and Run All.
The version banner must say **DIVERSE-GRAPH-V11 | 92 individual + 12 ensemble Optuna trials**.

## Fresh setup

Use Python 3.10 or 3.11 and the pinned `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name bt4012 --display-name BT4012
```

Select that kernel. Keep the notebook with `train.csv`, `test.csv`, `txs_edgelist.csv`, and
`sample_submission.csv`, or edit `DATA_DIR`. Every result is computed in the current run.
No previous output, model, study, external dataset, or downloaded solution is required.
The notebook is self-contained; the tests are optional and are not needed for training.

## V11 changes

1. Add a two-stage graph-risk classifier using chronologically out-of-fold teacher predictions.
2. Add Extra Trees and Random Forest, expand XGBoost, and explicitly test model contributions
   to an ensemble even when their standalone scores are lower.
3. Expand and redistribute fixed-count hyperparameter searches while retaining window selection,
   three-seed final checks, reference recipes, duplicate removal and replacement margins.

The previous 64 same-family joint trials are reduced to 12 to fund broader experiments.
This revision does not introduce a new holdout or change the chronological scoring objective.
It implements suggestions 1, 2 and 3 only; subgroup specialists and a revised validation split
are not part of this revision.

## Search budget

```python
SEARCH_TRIALS = {
    "lgb_local": 24, "lgb_broad": 8, "xgb": 24,
    "extra_trees": 16, "random_forest": 12, "graph_risk": 8,
}
ENSEMBLE_SEARCH_TRIALS = {"focused": 8, "broad": 4}
FINAL_SEEDS = [42, 137, 2026]
```

There are **104 Optuna trials**, plus declared reference configurations. Joint trials fit two
components per fold. Defaults are fixed counts, with no wall-clock timeout. Runtime on the full
competition data has not been measured; it can exceed the previous 67-minute run.

| Study | Main choices |
|---|---|
| Local LightGBM | 7-63 leaves; 64-768 trees; learning rate 0.015-0.08 |
| Broad LightGBM | Five feature sets; 15-127 leaves; fixed trees or inner early stopping |
| XGBoost | Depth 2-6; 200/400/800/1200 trees; learning rate 0.015-0.08 |
| Extra Trees | 400-1000 trees; minimum leaf 1-30; feature fraction 0.3-1.0 |
| Random Forest | 300-800 trees; minimum leaf 1-30; feature fraction 0.3-1.0 |
| Graph-risk LightGBM | 7/15/31 leaves; 96/132/256/512 trees; learning rate 0.015-0.08 |

Forest depth choices are unlimited/8/16/24, with optional class balancing. They can use raw,
structural, or raw-plus-period-rank features. A median imputer is fitted on each training partition.
Random Forest bootstraps observations; Extra Trees uses randomized thresholds without bootstrapping.
Both support recency sample weights. Forest tree count and boosting iterations are different concepts.

Individual TPE studies use `min(10, max(2, count // 3))` startup trials. The 24-trial XGBoost study
therefore includes adaptive TPE search, unlike the previous four startup-only trials.
Set an individual family count to zero to disable its study. The original XGBoost reference remains.
Set both joint counts to zero to disable joint tuning. Disabling the window experiment also requires
zero joint counts. All changed settings are recorded in the run configuration.

## Neighboring predicted risk

Transactions are graph nodes; edges indicate directed connections, not amounts or wallet ownership.
The new branch derives 13 features: the teacher's own predicted risk, and for each incoming/outgoing
neighborhood, mean risk, maximum risk, fraction at or above 0.5, observed count, observed fraction,
and weighted two-hop mean risk. These are features for a second LightGBM classifier alongside the
original 165 features. No neighbor label is attached as an input feature.

Training features use chronological cross-fitting:

- The first teacher trains on periods 1-3 and predicts periods 4-7.
- The next trains through period 7 and predicts periods 8-11, and so on through period 35.
- Each teacher uses the literal recent-reference parameters, up to 20 periods of history,
  half-life 32, and 132 trees. The teacher is not tuned against its prediction block.
- A second-stage fit uses only labeled rows at or before its own cutoff, with available OOF features.
  Periods 1-3 are excluded from stage two because no earlier teacher predictions are available.
- For validation or test, a teacher trained through the applicable cutoff creates the new features.
  Final test features use a teacher trained through period 35.

`GRAPH_MIN_HISTORY`, `GRAPH_OOF_BLOCK_PERIODS`, `GRAPH_TEACHER_WINDOW`, and
`GRAPH_HIGH_RISK_THRESHOLD` declare this policy. OOF caches are built within each run and may cover
all training periods, but fitting always selects only its allowed historical rows. Featureless graph
nodes retain topology; they are not assigned fabricated zero-risk scores. Missing means/maxima remain
NaN. Attribute sources use labeled training rows and test rows, matching the existing observation policy.
The 0.5 threshold defines a feature only; Kaggle submissions remain continuous probabilities.

The notebook checks that no connected component spans known periods before computing global graph
features. This is batch prediction on the supplied graph, not a claim of real-time availability.

## Window selection, recency and ensembles

The fixed-parameter screen retains windows **16, 20, 24** and half-life 32. Each gets the final three
seeds. Compare equal-weight pairs 16/20, 16/24, 20/24, then lock the best pair for joint parameter tuning.
Individual boosting searches still offer all history and 12/16/20/24/28/32 periods. New forest and graph-risk
studies offer all history and 16/20/24 periods. These choices are prior experiment-informed hypotheses.

`RECENCY_MODE` controls new individual models and broad joint trials:

| Mode | Half-life |
|---|---|
| search | None, 4, 8, 12, 16, 24, 32 |
| recent | 4, 8, 12 |
| fixed | Positive `FIXED_HALF_LIFE` |
| uniform | None (equal weights) |

Weight = `2 ** ((period - newest_training_period) / half_life)`, anchored within each fit.
Original references, the controlled window screen, the graph teacher, and focused joint tuning retain
their declared decay settings independently. Stage-two graph-risk models follow the selected recency mode.

Seed checks retain the top five unique individual models, all required references/windows, each family's
best standalone model, and each family's best complement to the fixed 16/20 reference. Complement
screening tests added weights **0.10, 0.25, 0.50**. This allows a lower-AUC model to qualify when its combined
predictions improve the score. Scores use averaged probabilities, not averages of component AUCs.

After seed checks, those models stay eligible outside the top-eight general blend cap. Explicit
complement proposals combine each with the reference and with the best checked joint ensemble. Base
recipes and standalone models remain eligible, so an unhelpful family can receive zero final weight.
Identical models and effective recipes are deduplicated; different windows are not merged merely
because their predictions happen to match on early folds.

## Validation, safeguards and limitations

Cutoffs remain 10, 14, 18, 21, 25, 28, with validation horizons 14, 14, 14, 14, 10, 7 periods.
The objective remains weighted mean window ROC-AUC minus 0.05 times window-AUC standard deviation.
Fold weights are 0.2/0.2/0.2/0.2/0.1/0.1. Applicable early stopping uses only an inner training tail.

Blends must improve on the best screened single by 0.0001. Replacing the fixed 16/20 reference also
requires a selection-score gain of at least 0.0001. This is a tie policy, not a statistical guarantee.
The original baseline, recent reference and 16/20 reference are fitted afresh. The latter corresponds
to the user-reported 0.95102 recipe. That reported Kaggle score never enters the objective.

Development windows overlap and have been reused in previous experiments. Neither these scores nor
three-seed checks provide an unbiased test estimate. Broader tuning can overfit validation. Added model
families and graph features can fail to improve Kaggle. No guaranteed ROC-AUC increase is claimed.

## Reproducibility and output

Studies are sequential and seeded, with fixed counts; final seeds are 42/137/2026. Keep the same CSVs,
code, settings and pinned environment to repeat a run. Different libraries/hardware can change floating
point results and adaptive trial paths. No historical artifacts are read to produce predictions.

Every run saves `outputs/fresh_search_<timestamp>/` plus a sibling ZIP. Existing output folders can stay.
The folder includes notebook source, configuration, package versions, input/source hashes, trials,
validation diagnostics, locked ensemble recipe, model artifacts and aligned submission CSVs.

Key files:

- `submission.csv`: selected ensemble.
- `submission_window_reference.csv`: freshly generated fixed 16/20 reference, unless identical to an
  already exported role; consult `submission_manifest.json` for deduplicated filenames.
- `submission_best_single.csv`, `submission_baseline.csv`, `submission_recent_reference.csv`: alternatives,
  subject to the same exact-prediction file deduplication.
- `complement_screening.csv`: single-seed model contributions to the reference.
- `diversity_blend_comparison.csv`: seed-averaged diverse blend comparisons.
- `graph_teacher_audit.json`: strict training/prediction boundaries for every teacher.
- `candidate_predictions/graph_risk_oof_seed*.npz`: OOF risk feature caches, with row positions and columns.
- `optuna_*_trials.json/csv`, `candidate_specs.json`, `locked_strategy.json`: search and final recipe.
- `ensemble_comparison_all.csv`, `ensemble_comparison.csv`, `candidate_aliases.json`, `ensemble_aliases.json`:
  complete proposals and canonical selection records.

Graph-risk model artifacts carry the teacher and precomputed risk features keyed by this dataset's row
positions. The notebook's `predict_model` combines these with raw features. They are batch artifacts for
the supplied graph. Scoring a different graph requires recomputing its graph features; the pickle alone
is not a streaming deployment service. Input CSVs are excluded from the ZIP.

## Verification

`python tests/test_fresh_run.py` runs reduced real fitting and Optuna searches twice in separate empty
folders using tiny synthetic CSVs. It checks all model families, prediction alignment, deterministic
results, window selection, canonical recipes, replacement margins, final refitting and artifact reload.
Additional checks cover directed risk aggregation, missing-neighbor behavior and invariance of earlier
fits when future labels change. These tests do not train on competition CSVs or measure full runtime.
The Windows sandbox test uses single-threaded forests because joblib's thread-pool named pipes are blocked;
normal notebook fitting uses the configured thread count. Forest prediction accumulation is serial
to avoid thread-order differences in probability averages.
