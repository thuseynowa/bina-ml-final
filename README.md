# MLE-AI-201 Machine Learning — Final Project (Starter Pack)

Build **two models from scratch** (a decision tree and an SVM) on the
**bina.az** real-estate dataset, then benchmark honestly against
scikit-learn. Read the full brief in the project statement PDF handed out
with this pack.

- **Released:** Sun 04 Oct 2026
- **Due:** **Sun 18 Oct 2026, 23:59 (Baku)** — two weeks, no late submissions.
- **Demo:** defended in the session after the deadline.

## What you submit (on Moodle)
Exactly **three** things; **one member submits** for the whole team:
1. **GitHub link** to your repo (tagged `v1.0-final`).
2. **Report PDF** (IEEE two-column — use `report/report.tex`).
3. **Slides PDF**.

Keep `contribution_report.md` in the repo root (checked at the demo).

## The two tasks
- **Task A — Regression:** predict `price` (consider `log(price)`).
- **Task B — Classification:** predict a derived **price tier**
  (premium vs standard, median split on the training set).

## Golden rule
Your **decision tree** and your **SVM** must be **your own NumPy code**.
scikit-learn is for the comparison baselines (and the optional bonuses)
only. For the SVM you do **not** need a QP/SMO solver — use the
**Pegasos** sub-gradient method (see the statement and `src/svm.py`).

## Repo layout
```
.
├── README.md               # this file
├── requirements.txt        # pin your versions
├── .gitignore
├── contribution_report.md  # who did what (keep in repo root)
├── data/                   # dataset lives here (not committed — see data/README.md)
├── src/                    # YOUR from-scratch code
│   ├── data_prep.py        #   loading, cleaning, stratified split, tier label
│   ├── decision_tree.py    #   DecisionTree (Gini/entropy/MSE)
│   ├── svm.py              #   PegasosSVM (hinge-loss sub-gradient) + RFF
│   ├── evaluate.py         #   metrics + comparison helpers
│   ├── baselines.py        #   bonus P8: Ridge (normal eq.), SVR (sub-gradient)
│   ├── ensemble.py         #   bonus P9: RandomForest over our trees
│   └── run_all.py          #   ONE command reproduces every number/figure
├── tests/test_models.py    # sanity tests vs scikit-learn (run first by run_all)
├── results/                # results.json + tables.md (generated)
├── report/                 # IEEE report (self-contained, no IEEEtran.cls)
│   ├── report.tex
│   └── figures/
└── presentation/           # your slides PDF
```

## Reproducibility (graded)
- Pin dependencies in `requirements.txt`; fix all random seeds.
- `python -m src.run_all` must reproduce **every** headline number and
  figure from a clean checkout.
- Do **not** commit the raw data archive (see `.gitignore`).

## Quick start
```bash
# Linux / macOS
python -m venv .venv && source .venv/bin/activate
# Windows (PowerShell):  python -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# put the dataset in data/ (see data/README.md), then ONE command:
python -m src.run_all
```

`python -m src.run_all` (≈10 min on a laptop) first runs the sanity tests
(`python -m tests.test_models` runs them alone, 10 checks), then reproduces
**every** headline number and figure: console tables + `results/results.json`
+ `results/tables.md`, figures in `report/figures/`.

## Headline results (test set, seed 42, stratified 70/15/15 split)
| Model | Task A RMSE(log) | Task B F1 |
|---|---|---|
| **Ours: DecisionTree** | **0.2289** | **0.9228** |
| sklearn DecisionTree (matched) | 0.2287 | 0.9233 |
| **Ours: PegasosSVM (linear)** | — | **0.8971** |
| sklearn SGDClassifier (hinge, matched λ) | — | 0.8989 |
| Ours: Pegasos + RFF | — | 0.8491 |
| Bonus: our RandomForest ×30 | 0.2211 | — |
| Bonus: our Ridge / our SVR | 0.3006 / 0.3044 | — |

5-fold CV: tree RMSE(log) 0.2298 ± 0.0022 (ours) vs 0.2295 ± 0.0021
(sklearn); Pegasos F1 0.8988 ± 0.0016 vs SGD 0.9000 ± 0.0012.

## Implementation notes
- `src/decision_tree.py` — vectorised split search (per-feature sort +
  cumulative class counts / Σy, Σy²); gini, entropy, mse; `max_depth`,
  `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`,
  `max_features` (per-split, used by the bonus forest); impurity-decrease
  feature importances; `predict_proba` from leaf frequencies.
- `src/svm.py` — Pegasos mini-batch sub-gradient solver with three
  learning-rate schedules (1/(λt), η₀/√t, constant) and `n_epochs`,
  objective tracking, support vectors & margin 2/‖w‖; non-linearity via
  **random Fourier features** (`RandomFourierFeatures`).
- `src/data_prep.py` — documented cleaning rules, leakage columns dropped,
  **stratified** split on log-price deciles, district vocabulary fitted on
  train only, tier label from the train median.
- `src/baselines.py` — bonus P8: Ridge via the normal equations and a
  linear SVR via sub-gradient descent on the ε-insensitive loss.
- `src/ensemble.py` — bonus P9: random forest over our own trees
  (bootstrap + per-split feature subsampling).
- `tests/test_models.py` — our tree matches sklearn's split on
  fully-determined examples; Pegasos ≈ LinearSVC; SVR ≈ LinearSVR;
  Ridge ≡ sklearn; metrics ≡ sklearn; stratified split balance.
