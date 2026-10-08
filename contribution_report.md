# Contribution Report

Kept in the repo root; checked at the demo.

> **TODO(team):** replace the `---` placeholders with real names/handles
> and adjust the role split to match what actually happened. Every
> member must be able to defend every listed item at the demo.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Hüseynova, T.          | ---           | Data pipeline & experiments (`src/data_prep.py`, `src/run_all.py`), EDA figures, report |
| ---                    | ---           | Decision tree (`src/decision_tree.py`), tree experiments (Problem 2, 4a), forest bonus (`src/ensemble.py`) |
| ---                    | ---           | Pegasos SVM + RFF (`src/svm.py`), SVM experiments (Problems 3, 4b), metrics (`src/evaluate.py`), tests, slides |

## Per-member summary
**Hüseynova, T. —** Designed the cleaning rules (dedup by listing URL,
residential scope, string→numeric parsing, fixed outlier bounds) and the
leakage analysis (`unit_price`/`total_price` dropped); built the 55-feature
design matrix, the seeded 70/15/15 split and the train-median tier label;
wrote `run_all.py` so one command reproduces every number/figure; wrote the
data/EDA/error-analysis sections of the report.

**---, --- —** Implemented the from-scratch decision tree: vectorised
split search over sorted thresholds with cumulative class counts /
Σy, Σy²; Gini/entropy/MSE criteria; sklearn-style weighted
impurity-decrease and all four stopping hyperparameters; feature
importances and `predict_proba`. Ran the depth/criterion sweeps and the
bias–variance analysis; built the bagged forest bonus on top of the same
tree via per-split `max_features`.

**---, --- —** Implemented the Pegasos solver (mini-batch sub-gradient,
η_t = 1/(λt), unregularised bias), objective tracking, support-vector and
margin reporting, and the random-Fourier-features extension; ran the
λ/γ sweeps; implemented all metrics from scratch (rank-based AUC included)
plus the test suite that pins our models to scikit-learn on
fully-determined examples; made the slides.

## Rough effort split
| Member | Approx. % |
|--------|-----------|
| Hüseynova, T. | ~34% |
| ---    | ~33%      |
| ---    | ~33%      |

No significant imbalance: modules were sized to be comparable, and the
report/slides were written jointly.

## AI-assistance disclosure (course rule §8.5)
An AI assistant (Anthropic Claude) was used for code scaffolding and
drafting text of the report/slides. All numbers come from our own pipeline
(`python -m src.run_all`), and the team reviewed, tested and can defend
every line. <!-- TODO(team): adjust to reflect actual usage before submitting. -->
