# Fraudulent Bitcoin Transaction Detection

This project predicts the probability that a Bitcoin transaction is illicit. Transactions form a directed
graph, and the model must generalize from training periods **1-35** to test periods **36-49**.
The complete workflow is implemented in [e1121223.ipynb](e1121223.ipynb), covering data preparation,
feature engineering, model search, chronological validation and submission generation.

## Setup

Use Python 3.10 or 3.11 and the pinned `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name bt4012 --display-name BT4012
```

Select the **BT4012** kernel in VS Code or Jupyter and use the repository folder as the working directory.
The PowerShell commands above are for Windows; on macOS/Linux, use `.venv/bin/python` for the virtual
environment's interpreter. Restart the kernel and **Run All**.

The notebook requires all four CSVs. Place them beside the notebook, or edit `DATA_DIR`:

| File | Purpose |
|---|---|
| `train.csv` | Transaction attributes, periods and labels, including unknown labels |
| `test.csv` | Transactions requiring predictions |
| `txs_edgelist.csv` | Directed connections between transactions |
| `sample_submission.csv` | Output columns and row ordering |

Obtain these files through the competition's authorized data access; they do not need to be included
in the GitHub repository. Keep the notebook filename `e1121223.ipynb`: it reads its own source to record
the run. If the version check fails, reload the notebook from disk before restarting its kernel.

The notebook is self-contained and fits all models from the CSVs. Saved models, studies, output folders
and external data are not required. Tests are optional and are not needed for training.

## Workflow

1. Validate the supplied data and exclude unknown labels from supervised fitting.
2. Build transaction, graph-structure and within-period rank features.
3. Generate chronological out-of-fold risk features for a two-stage LightGBM classifier.
4. Compare LightGBM, XGBoost, Extra Trees and Random Forest using temporal validation and Optuna.
5. Evaluate training windows, three-seed predictions and complementary model blends.
6. Select a model or ensemble, refit it through period 35 and export aligned test probabilities.

## Data preprocessing and features

The notebook checks column schemas, identifiers, label values, time ranges and submission alignment.
Unknown labels are excluded from supervised fitting rather than converted to licit. Transaction IDs
and submission indices are not predictors. The 165 numeric attributes are already normalized;
infinities become NaN. Boosting models handle missing values, while forest models use median imputers
fitted only on their training partitions. No synthetic transactions are generated.

Candidate feature sets include raw attributes, the first 93 attributes, graph structure, neighborhood
aggregates and within-period percentile ranks. Structural features describe incoming/outgoing degrees,
component sizes, source/sink roles and walk statistics. Attribute aggregates include directional means,
deviations and two-hop information. Every graph endpoint contributes topology, including nodes without
attributes. Labeled training rows and test rows supply neighbor attributes to make observation coverage
more comparable; unavailable values are not treated as zero.

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
components per fold. Defaults are fixed counts, with no wall-clock timeout. A full run on the competition data took approximately **119 minutes before archiving** on the author's
device. Runtime varies with hardware and sampled configurations.

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

Optuna's individual TPE studies use `min(10, max(2, count // 3))` initial trials before adaptive search.
Set a family count to zero to disable its study. Separately declared reference models are still evaluated.
Set both joint counts to zero to disable joint tuning. Disabling the window experiment also requires
zero joint counts. All settings are recorded in the run configuration.

## Neighboring predicted risk

Transactions are graph nodes; edges indicate directed connections, not amounts or wallet ownership.
The graph-risk branch derives 13 features: the teacher's own predicted risk, and for each incoming/outgoing
neighborhood, mean risk, maximum risk, fraction at or above 0.5, observed count, observed fraction,
and weighted two-hop mean risk. These are features for a second LightGBM classifier alongside the
original 165 features. No neighbor label is attached as an input feature.

Training features use chronological cross-fitting:

- The first teacher trains on periods 1-3 and predicts periods 4-7.
- The next trains through period 7 and predicts periods 8-11, and so on through period 35.
- Each teacher uses the fixed LightGBM parameters in `recent_reference_params`, up to 20 periods of history,
  half-life 32, and 132 trees. The teacher is not tuned against its prediction block.
- A second-stage fit uses only labeled rows at or before its own cutoff, with available OOF features.
  Periods 1-3 are excluded from stage two because no earlier teacher predictions are available.
- For validation or test, a teacher trained through the applicable cutoff creates the risk features.
  Final test features use a teacher trained through period 35.

`GRAPH_MIN_HISTORY`, `GRAPH_OOF_BLOCK_PERIODS`, `GRAPH_TEACHER_WINDOW`, and
`GRAPH_HIGH_RISK_THRESHOLD` declare this policy. OOF caches are built within each run and may cover
all training periods, but fitting always selects only its allowed historical rows. Featureless graph
nodes retain topology; they are not assigned fabricated zero-risk scores. Missing means/maxima remain
NaN. Attribute sources use labeled training rows and test rows, following the same attribute-availability policy across periods.
The 0.5 threshold defines a feature only; Kaggle submissions remain continuous probabilities.

The notebook checks that no connected component spans known periods before computing global graph
features. This is batch prediction on the supplied graph, not a claim of real-time availability.

## Window selection, recency and ensembles

The fixed-parameter screen compares windows **16, 20, 24** and half-life 32. Each gets the final three
seeds. Compare equal-weight pairs 16/20, 16/24, 20/24, then lock the best pair for joint parameter tuning.
Individual boosting searches offer all history and 12/16/20/24/28/32-period windows. Forest and graph-risk
studies offer all history and 16/20/24-period windows. Finite windows limit training to recent history;
all-history candidates test whether older examples remain useful.

`RECENCY_MODE` controls individual search candidates and broad joint trials:

| Mode | Half-life |
|---|---|
| search | None, 4, 8, 12, 16, 24, 32 |
| recent | 4, 8, 12 |
| fixed | Positive `FIXED_HALF_LIFE` |
| uniform | None (equal weights) |

Weight = `2 ** ((period - newest_training_period) / half_life)`, anchored within each fit.
This emphasizes recent transaction patterns. Reference configurations, the window screen, graph teacher
and focused joint study use their own declared decay settings independently. Graph-risk search candidates
follow the selected recency mode.

Seed checks evaluate the top five unique individual models, all required references/windows, each family's
best standalone model, and each family's best complement to the fixed 16/20 reference. Complement
screening tests added weights **0.10, 0.25, 0.50**. This allows a lower-AUC model to qualify when its combined
predictions improve the score. Scores use averaged probabilities, not averages of component AUCs.

After seed checks, those models stay eligible outside the top-eight general blend cap. Explicit
complement proposals combine each with the reference and with the best checked joint ensemble. Base
recipes and standalone models remain eligible, so an unhelpful family can receive zero final weight.
Identical models and effective recipes are deduplicated; different windows are not merged merely
because their predictions happen to match on early folds.

## Validation, safeguards and limitations

Training cutoffs are 10, 14, 18, 21, 25, 28, with validation horizons 14, 14, 14, 14, 10, 7 periods.
The selection objective is weighted mean window ROC-AUC minus 0.05 times window-AUC standard deviation.
Fold weights are 0.2/0.2/0.2/0.2/0.1/0.1. Applicable early stopping uses only an inner training tail.

Blends must improve on the best screened single by 0.0001. Replacing the fixed 16/20 reference also
requires a selection-score gain of at least 0.0001. This is a tie policy, not a statistical guarantee.
Reference configurations are fitted from scratch: `reference_lgb` uses all training history with
half-life 16; `recent_reference_lgb` uses a 16-period window with half-life 32. The fixed 16/20 reference
averages that configuration with its 20-period counterpart. Kaggle scores never enter the selection
objective; the notebook does not have hidden test labels.

Development windows overlap, and repeated tuning can overfit these periods. Neither their scores nor
three-seed checks provide an independent test estimate. Heuristic labels, missing neighborhood attributes
and changing transaction patterns also limit generalization. More trials or a higher validation score
do not guarantee improved Kaggle performance. Batch graph features may include information unavailable
at the instant of a real-time decision.

## Reproducibility and output

Studies are sequential and seeded, with fixed counts; final seeds are 42/137/2026. Keep the same CSVs,
code, settings and pinned environment to repeat a run. Different libraries/hardware can change floating
point results and adaptive trial paths. All predictions are generated from the supplied data during the run.

Every run saves `outputs/fresh_search_<timestamp>/` plus a sibling ZIP. Existing output folders can stay.
The folder includes notebook source, configuration, package versions, input/source hashes, trials,
validation diagnostics, locked ensemble recipe, model artifacts and aligned submission CSVs.

Key files:

- `submission.csv`: test probabilities from the selected model or ensemble.
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

If the optional `tests/` directory is included, `python tests/test_fresh_run.py` runs reduced real fitting and Optuna searches twice in separate empty
folders using tiny synthetic CSVs. It checks all model families, prediction alignment, deterministic
results, window selection, canonical recipes, replacement margins, final refitting and artifact reload.
Additional checks cover directed risk aggregation, missing-neighbor behavior and invariance of earlier
fits when future labels change. These tests do not train on competition CSVs or measure full runtime.
Test forests use one thread for compatibility with restricted Windows environments;
normal notebook fitting uses the configured thread count. Forest prediction accumulation is serial
to avoid thread-order differences in probability averages.
