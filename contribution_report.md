# Contribution Report

Kept in the repo root; checked at the demo.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Hüseynova, Tahmina     | thuseynowa    | Data pipeline (`src/data_prep.py`): cleaning rules, leakage analysis, stratified split; orchestrator `src/run_all.py`; EDA and error-analysis figures; report |
| Samadzade, Elshan      | elshansamadzada7 | Decision tree (`src/decision_tree.py`): vectorised split search, criteria, stopping rules, importances; tree experiments (P2, P4a); forest bonus (`src/ensemble.py`) |
| Hajiyev, Alish         | Alishhajiyev  | Pegasos SVM (`src/svm.py`): solver, LR schedules, margin/support vectors, random Fourier features; SVM experiments (P3, P4b); SVR bonus (`src/baselines.py`) |
| Ismayilzade, Nihat     | NihatIsmayilzade | Metrics from scratch (`src/evaluate.py`); scikit-learn benchmark and 5-fold CV (P5); test suite (`tests/test_models.py`); Ridge bonus; k-means/PCA bonus; slides |

## Per-member summary
**Hüseynova, Tahmina —** Designed the cleaning pipeline (dedup by listing URL,
residential scope, Azerbaijani string→numeric parsing, fixed outlier
bounds, AZN currency check) and the leakage analysis (`unit_price`,
`total_price`, identifiers dropped); built the 55-feature design matrix
with the district vocabulary fitted on train only; implemented the
stratified 70/15/15 split on log-price deciles and the train-median tier
label; wrote `run_all.py` so one command reproduces every number and
figure; wrote the data, EDA and error-analysis sections of the report.

**Samadzade, Elshan —** Implemented the from-scratch decision tree: vectorised
split search over sorted thresholds with cumulative class counts /
Σy, Σy²; Gini/entropy/MSE criteria; sklearn-style weighted
impurity-decrease and all four stopping hyperparameters; feature
importances and `predict_proba`. Ran the depth/criterion/leaf sweeps and
the bias–variance analysis; built the bagged forest bonus on top of the
same tree via per-split `max_features`.

**Hajiyev, Alish —** Implemented the Pegasos solver (mini-batch sub-gradient,
three learning-rate schedules, epochs, unregularised bias), objective
tracking, support-vector and margin reporting, and the
random-Fourier-features extension; ran the λ / epochs / schedule / γ
sweeps; implemented the from-scratch SVR on the ε-insensitive loss as
the regression twin of Pegasos.

**Ismayilzade, Nihat —** Implemented all metrics from scratch (rank-based AUC with
midranks included) and the comparison tables; the matched scikit-learn
benchmark, timing and 5-fold CV with per-fold threshold/scaler refits;
the test suite that pins our models to scikit-learn on fully-determined
examples; the Ridge normal-equations bonus and the k-means/PCA
geospatial study; made the slides.

## Rough effort split
| Member | Approx. % |
|--------|-----------|
| Hajiyev, Alish      | 25% |
| Hüseynova, Tahmina  | 25% |
| Ismayilzade, Nihat  | 25% |
| Samadzade, Elshan   | 25% |

Equal split: modules were sized to be comparable, every member worked
through the full two weeks, and the report/slides were reviewed jointly.

## AI-assistance disclosure (course rule §8.5)
An AI assistant (Anthropic Claude, via Claude Code) was used for code
scaffolding, refactoring, drafting the report/slides text and reviewing
the project against the statement. All numbers come from our own
pipeline (`python -m src.run_all`); every line of the submitted
implementation was read, tested and is defended by the team.
