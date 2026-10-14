"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all

Order of operations:
  0. sanity tests (our models vs sklearn on toy data)
  1. load + clean the data, EDA figures
  2. stratified 70/15/15 split (log-price deciles), features with the
     district vocabulary fitted on train, tier label from the TRAIN median
  3. Task A: our DecisionTree regressor — depth & min_samples_leaf sweeps
  4. Task B: our DecisionTree classifier — gini vs entropy sweep
  5. Task B: our PegasosSVM — lambda / epochs / LR-schedule sweeps,
     RFF gamma sweep, convergence
  6. scikit-learn baselines with matched hyperparameters
  7. 5-fold CV uncertainty (mean ± std) for the headline models
  8. feature importances, SVM weights, error analysis
  9. bonuses: from-scratch Ridge + SVR (P8), from-scratch RandomForest (P9),
     k-means/PCA geospatial EDA (P10, sklearn allowed for bonuses)
 10. final results tables -> console + results/results.json + results/tables.md

Everything is seeded (SEED = 42); figures land in report/figures/.
Full run takes ~10-15 minutes on a laptop.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data_prep, evaluate as ev
from .baselines import LinearSVR, RidgeRegression
from .decision_tree import DecisionTree
from .ensemble import RandomForest
from .svm import PegasosSVM, RandomFourierFeatures

SEED = data_prep.SEED
FIGDIR = os.path.join("report", "figures")
RESDIR = "results"

# Okabe-Ito colourblind-safe palette (fixed assignment order)
C_BLUE, C_ORANGE, C_GREEN, C_VERM, C_PURPLE, C_SKY, C_GRAY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
    "#999999")

RESULTS: dict = {"seed": SEED}
TIMINGS: dict[str, dict] = {}


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def savefig(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  [fig] {path}")


def header(text):
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


# ---------------------------------------------------------------- step 0
def run_sanity_tests():
    header("0. SANITY TESTS (our implementations vs scikit-learn, toy data)")
    from tests.test_models import run_all_tests
    run_all_tests()


# ---------------------------------------------------------------- step 1-2
def load_and_prepare():
    header("1. DATA: load, clean, EDA")
    df = data_prep.clean(data_prep.load_raw())
    log = df.attrs["cleaning_log"]
    for step, n in log.items():
        print(f"  {step}: {n:,}")
    RESULTS["cleaning_log"] = log

    header("2. STRATIFIED SPLIT + PRICE-TIER LABEL")
    # Split FIRST (on price deciles), then build features, so the one
    # data-dependent encoding choice (district vocabulary) sees train only.
    y_all = df["price"].to_numpy(dtype=np.float64)
    idx_tr, idx_val, idx_te = data_prep.stratified_split_indices(y_all)
    X, y, names = data_prep.make_features(df, train_idx=idx_tr)
    print(f"  design matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    RESULTS["n_rows"], RESULTS["n_features"] = X.shape
    RESULTS["leakage_columns_dropped"] = data_prep.LEAKAGE_COLUMNS

    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_val, y_val = X[idx_val], y[idx_val]
    X_te, y_te = X[idx_te], y[idx_te]
    tier_tr, thr = data_prep.make_tier_label(y_tr)
    tier_val, _ = data_prep.make_tier_label(y_val, thr)
    tier_te, _ = data_prep.make_tier_label(y_te, thr)
    print(f"  split sizes: train={len(y_tr):,} val={len(y_val):,} "
          f"test={len(y_te):,}  (stratified on log-price deciles)")
    print(f"  tier threshold (TRAIN median) = {thr:,.0f} AZN")
    print(f"  premium share: train={tier_tr.mean():.3f} "
          f"val={tier_val.mean():.3f} test={tier_te.mean():.3f}")
    print(f"  median log-price: train={np.median(np.log(y_tr)):.4f} "
          f"val={np.median(np.log(y_val)):.4f} "
          f"test={np.median(np.log(y_te)):.4f}")
    RESULTS["split_sizes"] = [len(y_tr), len(y_val), len(y_te)]
    RESULTS["tier_threshold_azn"] = thr
    RESULTS["tier_balance"] = [float(tier_tr.mean()), float(tier_val.mean()),
                               float(tier_te.mean())]

    # ---- EDA figures (train split only — no test peeking in plots) ----
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    ax = axes[0, 0]
    ax.hist(y_tr / 1000, bins=80, color=C_BLUE)
    _style(ax, "(a) Price distribution (train)", "price, thousand AZN",
           "listings")
    ax = axes[0, 1]
    ax.hist(np.log(y_tr), bins=80, color=C_BLUE)
    ax.axvline(np.log(thr), color=C_VERM, lw=1.5, ls="--",
               label="tier threshold")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "(b) log(price) with tier threshold", "log(price)", "listings")
    ax = axes[1, 0]
    area = X_tr[:, 0]
    hb = ax.hexbin(area, np.log(y_tr), gridsize=45, cmap="viridis",
                   mincnt=1)
    fig.colorbar(hb, ax=ax, label="listings")
    _style(ax, "(c) Area vs log(price)", "area, m$^2$", "log(price)")
    ax = axes[1, 1]
    rooms = X_tr[:, 1].astype(int)
    rr = sorted(r for r in np.unique(rooms) if 1 <= r <= 8)
    med = [np.median(y_tr[rooms == r]) / 1000 for r in rr]
    ax.bar([str(r) for r in rr], med, color=C_BLUE)
    _style(ax, "(d) Median price by rooms", "rooms", "median price, kAZN")
    savefig(fig, "fig_eda.png")

    # geography
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sub = np.random.default_rng(SEED).choice(len(y_tr), 15000, replace=False)
    lat_i, lng_i = names.index("lat"), names.index("lng")
    sc = ax.scatter(X_tr[sub, lng_i], X_tr[sub, lat_i], c=np.log(y_tr[sub]),
                    s=2, cmap="viridis", alpha=0.6)
    fig.colorbar(sc, ax=ax, label="log(price)")
    _style(ax, "Listings around Baku, coloured by log(price)",
           "longitude", "latitude")
    savefig(fig, "fig_geo.png")

    return (df, X, y, names, X_tr, y_tr, X_val, y_val, X_te, y_te,
            idx_tr, idx_val, idx_te, tier_tr, tier_val, tier_te, thr)


# ---------------------------------------------------------------- step 3
def task_a_tree(X_tr, ylog_tr, X_val, ylog_val, X_te, ylog_te, y_te):
    header("3. TASK A — our DecisionTree regressor on log(price)")
    depths = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30]
    tr_curve, val_curve = [], []
    for d in depths:
        t = DecisionTree(task="regression", criterion="mse", max_depth=d,
                         min_samples_leaf=5).fit(X_tr, ylog_tr)
        tr_curve.append(ev.rmse(ylog_tr, t.predict(X_tr)))
        val_curve.append(ev.rmse(ylog_val, t.predict(X_val)))
        print(f"  depth={d:>2}  train RMSE(log)={tr_curve[-1]:.4f}  "
              f"val RMSE(log)={val_curve[-1]:.4f}")
    best_depth = depths[int(np.argmin(val_curve))]

    leafs = [1, 2, 5, 10, 20, 50]
    leaf_val = []
    for m in leafs:
        t = DecisionTree(task="regression", criterion="mse",
                         max_depth=best_depth, min_samples_leaf=m
                         ).fit(X_tr, ylog_tr)
        leaf_val.append(ev.rmse(ylog_val, t.predict(X_val)))
    best_leaf = leafs[int(np.argmin(leaf_val))]
    print(f"  chosen on validation: max_depth={best_depth}, "
          f"min_samples_leaf={best_leaf}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.plot(depths, tr_curve, "-o", ms=4, color=C_BLUE, label="train")
    ax.plot(depths, val_curve, "-o", ms=4, color=C_ORANGE, label="validation")
    ax.axvline(best_depth, color=C_GRAY, lw=1, ls="--")
    ax.legend(frameon=False)
    _style(ax, "(a) Bias-variance: RMSE(log) vs depth", "max_depth",
           "RMSE on log(price)")
    ax = axes[1]
    ax.plot(leafs, leaf_val, "-o", ms=4, color=C_ORANGE)
    ax.set_xscale("log")
    _style(ax, f"(b) min_samples_leaf at depth={best_depth}",
           "min_samples_leaf", "val RMSE(log)")
    savefig(fig, "fig_tree_depth.png")

    final = DecisionTree(task="regression", criterion="mse",
                         max_depth=best_depth, min_samples_leaf=best_leaf)
    _, fit_s = ev.timed(final.fit, X_tr, ylog_tr)
    pred_log, pred_s = ev.timed(final.predict, X_te)
    pred_azn = np.exp(pred_log)
    TIMINGS["Ours: DecisionTree (reg)"] = {"fit_s": fit_s, "pred_s": pred_s}
    m = {
        "RMSE_log": ev.rmse(ylog_te, pred_log),
        "MAE_log": ev.mae(ylog_te, pred_log),
        "R2_log": ev.r2(ylog_te, pred_log),
        "RMSE_AZN": ev.rmse(y_te, pred_azn),
        "MAE_AZN": ev.mae(y_te, pred_azn),
        "R2_AZN": ev.r2(y_te, pred_azn),
    }
    print("  TEST:", {k: round(v, 4) for k, v in m.items()})
    RESULTS["taskA_tree_config"] = {"max_depth": best_depth,
                                    "min_samples_leaf": best_leaf}
    RESULTS["taskA_tree_test"] = m
    return final, best_depth, best_leaf, m, pred_log


# ---------------------------------------------------------------- step 4
def task_b_tree(X_tr, tier_tr, X_val, tier_val, X_te, tier_te):
    header("4. TASK B — our DecisionTree classifier (gini vs entropy)")
    depths = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24]
    curves = {}
    for crit in ("gini", "entropy"):
        f1s = []
        for d in depths:
            t = DecisionTree(task="classification", criterion=crit,
                             max_depth=d, min_samples_leaf=5
                             ).fit(X_tr, tier_tr)
            f1s.append(ev.precision_recall_f1(tier_val, t.predict(X_val))[2])
        curves[crit] = f1s
        print(f"  {crit}: best val F1={max(f1s):.4f} "
              f"at depth={depths[int(np.argmax(f1s))]}")
    best_crit = max(curves, key=lambda c: max(curves[c]))
    best_depth = depths[int(np.argmax(curves[best_crit]))]
    print(f"  chosen on validation: criterion={best_crit}, "
          f"max_depth={best_depth}")

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(depths, curves["gini"], "-o", ms=4, color=C_BLUE, label="gini")
    ax.plot(depths, curves["entropy"], "-s", ms=4, color=C_GREEN,
            label="entropy")
    ax.axvline(best_depth, color=C_GRAY, lw=1, ls="--")
    ax.legend(frameon=False)
    _style(ax, "Validation F1 vs depth, gini vs entropy", "max_depth",
           "validation F1")
    savefig(fig, "fig_tree_cls.png")

    final = DecisionTree(task="classification", criterion=best_crit,
                         max_depth=best_depth, min_samples_leaf=5)
    _, fit_s = ev.timed(final.fit, X_tr, tier_tr)
    pred, pred_s = ev.timed(final.predict, X_te)
    proba = final.predict_proba(X_te)[:, 1]
    TIMINGS["Ours: DecisionTree (cls)"] = {"fit_s": fit_s, "pred_s": pred_s}
    p, r, f1 = ev.precision_recall_f1(tier_te, pred)
    m = {"accuracy": ev.accuracy(tier_te, pred), "precision": p, "recall": r,
         "F1": f1, "ROC_AUC": ev.roc_auc(tier_te, proba)}
    print("  TEST:", {k: round(v, 4) for k, v in m.items()})
    RESULTS["taskB_tree_config"] = {"criterion": best_crit,
                                    "max_depth": best_depth,
                                    "min_samples_leaf": 5}
    RESULTS["taskB_tree_test"] = m
    return final, best_crit, best_depth, m, pred, proba


# ---------------------------------------------------------------- step 5
def task_b_svm(Xs_tr, tier_tr, Xs_val, tier_val, Xs_te, tier_te):
    header("5. TASK B — our PegasosSVM (linear + random Fourier features)")
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    f1s, margins, hist_by_lambda = [], [], {}
    for lam in lambdas:
        svm = PegasosSVM(lambda_=lam, n_iters=30_000, batch_size=64,
                         random_state=SEED).fit(Xs_tr, tier_tr)
        f1 = ev.precision_recall_f1(tier_val, svm.predict(Xs_val))[2]
        f1s.append(f1)
        margins.append(svm.margin_width_)
        hist_by_lambda[lam] = svm.objective_history_
        print(f"  lambda={lam:.0e}  val F1={f1:.4f}  "
              f"margin 2/||w||={svm.margin_width_:.4f}  "
              f"support vectors={len(svm.support_):,}")
    best_lambda = lambdas[int(np.argmax(f1s))]

    # margin / metric trade-off (two separate axes — no dual-axis chart)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.plot(lambdas, f1s, "-o", ms=4, color=C_ORANGE)
    ax.set_xscale("log")
    _style(ax, "(a) Validation F1 vs lambda", "lambda", "validation F1")
    ax = axes[1]
    ax.plot(lambdas, margins, "-o", ms=4, color=C_BLUE)
    ax.set_xscale("log"); ax.set_yscale("log")
    _style(ax, "(b) Margin width 2/||w|| vs lambda", "lambda",
           "margin width")
    savefig(fig, "fig_svm_lambda.png")

    # objective convergence
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for lam, col in zip([1e-5, 1e-4, 1e-3], [C_SKY, C_ORANGE, C_PURPLE]):
        h = hist_by_lambda[lam]
        ax.plot([t for t, _ in h], [v for _, v in h], color=col,
                label=f"lambda={lam:.0e}")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    _style(ax, "Pegasos objective J(w,b) decreases", "iteration t",
           "J(w, b) (log scale)")
    savefig(fig, "fig_svm_objective.png")

    # --- P3b: the other two hyperparameters — epochs and the LR schedule ---
    steps_per_epoch = int(np.ceil(len(tier_tr) / 64))
    epochs = [1, 2, 5, 10, 20, 50]
    ep_f1, ep_margin = [], []
    for e in epochs:
        svm = PegasosSVM(lambda_=best_lambda, n_epochs=e, batch_size=64,
                         random_state=SEED, track_every=0).fit(Xs_tr, tier_tr)
        ep_f1.append(ev.precision_recall_f1(tier_val, svm.predict(Xs_val))[2])
        ep_margin.append(svm.margin_width_)
        print(f"  epochs={e:>2} ({svm.n_iters_:>6} steps)  val F1={ep_f1[-1]:.4f}"
              f"  margin={svm.margin_width_:.4f}")

    schedules = [("pegasos", "1/(λt)"), ("inv_sqrt", "η0/√t"),
                 ("constant", "η0")]
    sched_hist, sched_f1, sched_margin = {}, {}, {}
    for key, label in schedules:
        svm = PegasosSVM(lambda_=best_lambda, n_iters=30_000, batch_size=64,
                         random_state=SEED, lr_schedule=key, eta0=0.1
                         ).fit(Xs_tr, tier_tr)
        sched_hist[key] = svm.objective_history_
        sched_f1[key] = ev.precision_recall_f1(tier_val,
                                               svm.predict(Xs_val))[2]
        sched_margin[key] = svm.margin_width_
        print(f"  schedule={key:<9} ({label})  val F1={sched_f1[key]:.4f}"
              f"  margin={svm.margin_width_:.4f}"
              f"  final J={svm.objective_history_[-1][1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.plot(epochs, ep_f1, "-o", ms=4, color=C_ORANGE)
    ax.set_xscale("log")
    _style(ax, f"(a) Validation F1 vs epochs (λ={best_lambda:.0e})",
           f"epochs (1 epoch = {steps_per_epoch} mini-batch steps)",
           "validation F1")
    ax = axes[1]
    for (key, label), col in zip(schedules, [C_PURPLE, C_GREEN, C_VERM]):
        h = sched_hist[key]
        ax.plot([t for t, _ in h], [v for _, v in h], color=col,
                label=f"{label}: F1={sched_f1[key]:.3f}")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "(b) Objective J(w,b) by learning-rate schedule",
           "iteration t", "J(w, b) (log scale)")
    savefig(fig, "fig_svm_schedule.png")
    RESULTS["taskB_svm_epochs"] = {"epochs": epochs, "val_F1": ep_f1,
                                   "margin": ep_margin,
                                   "steps_per_epoch": steps_per_epoch}
    RESULTS["taskB_svm_schedules"] = {k: {"val_F1": sched_f1[k],
                                          "margin": sched_margin[k],
                                          "final_J": sched_hist[k][-1][1]}
                                      for k, _ in schedules}

    # final linear model
    lin = PegasosSVM(lambda_=best_lambda, n_iters=30_000, batch_size=64,
                     random_state=SEED)
    _, fit_s = ev.timed(lin.fit, Xs_tr, tier_tr)
    pred, pred_s = ev.timed(lin.predict, Xs_te)
    TIMINGS["Ours: PegasosSVM (linear)"] = {"fit_s": fit_s, "pred_s": pred_s}
    p, r, f1 = ev.precision_recall_f1(tier_te, pred)
    m_lin = {"accuracy": ev.accuracy(tier_te, pred), "precision": p,
             "recall": r, "F1": f1,
             "ROC_AUC": ev.roc_auc(tier_te, lin.decision_function(Xs_te))}
    print(f"  LINEAR final lambda={best_lambda:.0e}  ||w||="
          f"{np.linalg.norm(lin.w):.4f}  margin={lin.margin_width_:.4f}  "
          f"SVs={len(lin.support_):,}/{len(tier_tr):,}")
    print("  TEST:", {k: round(v, 4) for k, v in m_lin.items()})

    # non-linearity: random Fourier features + the SAME linear solver
    gammas = [0.005, 0.01, 0.02, 0.05, 0.1]
    rff_f1 = []
    for g in gammas:
        rff = RandomFourierFeatures(n_components=300, gamma=g,
                                    random_state=SEED).fit(Xs_tr)
        Ztr, Zval = rff.transform(Xs_tr), rff.transform(Xs_val)
        svm = PegasosSVM(lambda_=best_lambda, n_iters=30_000, batch_size=64,
                         random_state=SEED).fit(Ztr, tier_tr)
        rff_f1.append(ev.precision_recall_f1(tier_val,
                                             svm.predict(Zval))[2])
        print(f"  RFF gamma={g}  val F1={rff_f1[-1]:.4f}")
    best_gamma = gammas[int(np.argmax(rff_f1))]
    rff = RandomFourierFeatures(n_components=300, gamma=best_gamma,
                                random_state=SEED).fit(Xs_tr)
    Ztr, Zte = rff.transform(Xs_tr), rff.transform(Xs_te)
    rff_svm = PegasosSVM(lambda_=best_lambda, n_iters=30_000, batch_size=64,
                         random_state=SEED)
    _, fit_s = ev.timed(rff_svm.fit, Ztr, tier_tr)
    pred_rff, pred_s = ev.timed(rff_svm.predict, Zte)
    TIMINGS["Ours: Pegasos + RFF"] = {"fit_s": fit_s, "pred_s": pred_s}
    p, r, f1 = ev.precision_recall_f1(tier_te, pred_rff)
    m_rff = {"accuracy": ev.accuracy(tier_te, pred_rff), "precision": p,
             "recall": r, "F1": f1,
             "ROC_AUC": ev.roc_auc(tier_te, rff_svm.decision_function(Zte))}
    print(f"  RFF final gamma={best_gamma}:",
          {k: round(v, 4) for k, v in m_rff.items()})

    RESULTS["taskB_svm_config"] = {
        "lambda": best_lambda, "n_iters": 30_000, "batch_size": 64,
        "lr_schedule": "pegasos", "norm_w": float(np.linalg.norm(lin.w)),
        "margin_width": lin.margin_width_,
        "n_support_vectors": int(len(lin.support_)),
        "rff_gamma": best_gamma, "rff_components": 300,
    }
    RESULTS["taskB_svm_linear_test"] = m_lin
    RESULTS["taskB_svm_rff_test"] = m_rff
    return lin, m_lin, rff_svm, rff, m_rff, best_lambda, best_gamma, pred, pred_rff


# ---------------------------------------------------------------- step 6
def sklearn_baselines(X_tr, ylog_tr, X_te, ylog_te, y_te,
                      tier_tr, tier_te, X_val, Xs_tr, Xs_te,
                      cfgA, cfgB, best_lambda, best_gamma):
    header("6. SCIKIT-LEARN BASELINES (matched hyperparameters)")
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    out = {}
    # Task A tree
    sk_reg = DecisionTreeRegressor(max_depth=cfgA["max_depth"],
                                   min_samples_leaf=cfgA["min_samples_leaf"],
                                   random_state=SEED)
    _, fit_s = ev.timed(sk_reg.fit, X_tr, ylog_tr)
    pred_log, pred_s = ev.timed(sk_reg.predict, X_te)
    TIMINGS["sklearn: DecisionTreeRegressor"] = {"fit_s": fit_s,
                                                 "pred_s": pred_s}
    out["sk_tree_reg"] = {
        "RMSE_log": ev.rmse(ylog_te, pred_log),
        "MAE_log": ev.mae(ylog_te, pred_log),
        "R2_log": ev.r2(ylog_te, pred_log),
        "RMSE_AZN": ev.rmse(y_te, np.exp(pred_log)),
        "MAE_AZN": ev.mae(y_te, np.exp(pred_log)),
        "R2_AZN": ev.r2(y_te, np.exp(pred_log)),
    }
    print("  DecisionTreeRegressor:",
          {k: round(v, 4) for k, v in out["sk_tree_reg"].items()})

    # Task B tree
    sk_cls = DecisionTreeClassifier(criterion=cfgB["criterion"],
                                    max_depth=cfgB["max_depth"],
                                    min_samples_leaf=cfgB["min_samples_leaf"],
                                    random_state=SEED)
    _, fit_s = ev.timed(sk_cls.fit, X_tr, tier_tr)
    pred, pred_s = ev.timed(sk_cls.predict, X_te)
    TIMINGS["sklearn: DecisionTreeClassifier"] = {"fit_s": fit_s,
                                                  "pred_s": pred_s}
    p, r, f1 = ev.precision_recall_f1(tier_te, pred)
    out["sk_tree_cls"] = {"accuracy": ev.accuracy(tier_te, pred),
                          "precision": p, "recall": r, "F1": f1,
                          "ROC_AUC": ev.roc_auc(
                              tier_te, sk_cls.predict_proba(X_te)[:, 1])}
    print("  DecisionTreeClassifier:",
          {k: round(v, 4) for k, v in out["sk_tree_cls"].items()})

    # SGDClassifier(hinge) — the sklearn twin of Pegasos, scaler in Pipeline
    sgd = make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="hinge", alpha=best_lambda, max_iter=50,
                      tol=None, random_state=SEED))
    _, fit_s = ev.timed(sgd.fit, X_tr, tier_tr)
    pred, pred_s = ev.timed(sgd.predict, X_te)
    TIMINGS["sklearn: SGDClassifier (hinge)"] = {"fit_s": fit_s,
                                                 "pred_s": pred_s}
    p, r, f1 = ev.precision_recall_f1(tier_te, pred)
    out["sk_sgd"] = {"accuracy": ev.accuracy(tier_te, pred), "precision": p,
                     "recall": r, "F1": f1,
                     "ROC_AUC": ev.roc_auc(tier_te,
                                           sgd.decision_function(X_te))}
    print("  SGDClassifier(hinge):",
          {k: round(v, 4) for k, v in out["sk_sgd"].items()})

    # LinearSVC with C matched to lambda: C = 1/(n*lambda)
    C = 1.0 / (len(tier_tr) * best_lambda)
    lsvc = make_pipeline(StandardScaler(),
                         LinearSVC(C=C, dual=True, max_iter=5000,
                                   random_state=SEED))
    _, fit_s = ev.timed(lsvc.fit, X_tr, tier_tr)
    pred, pred_s = ev.timed(lsvc.predict, X_te)
    TIMINGS["sklearn: LinearSVC"] = {"fit_s": fit_s, "pred_s": pred_s}
    p, r, f1 = ev.precision_recall_f1(tier_te, pred)
    out["sk_linsvc"] = {"accuracy": ev.accuracy(tier_te, pred),
                        "precision": p, "recall": r, "F1": f1,
                        "ROC_AUC": ev.roc_auc(tier_te,
                                              lsvc.decision_function(X_te))}
    print(f"  LinearSVC(C={C:.3f}):",
          {k: round(v, 4) for k, v in out["sk_linsvc"].items()})

    # SVC-RBF: O(n^2) kernel solver -> documented 15k subsample
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(tier_tr), 15_000, replace=False)
    svc = make_pipeline(StandardScaler(),
                        SVC(C=C, gamma=best_gamma, random_state=SEED))
    _, fit_s = ev.timed(svc.fit, X_tr[sub], tier_tr[sub])
    pred, pred_s = ev.timed(svc.predict, X_te)
    TIMINGS["sklearn: SVC-RBF (15k sub)"] = {"fit_s": fit_s,
                                             "pred_s": pred_s}
    p, r, f1 = ev.precision_recall_f1(tier_te, pred)
    out["sk_svc"] = {"accuracy": ev.accuracy(tier_te, pred), "precision": p,
                     "recall": r, "F1": f1,
                     "ROC_AUC": ev.roc_auc(tier_te,
                                           svc.decision_function(X_te))}
    print("  SVC(rbf, 15k subsample):",
          {k: round(v, 4) for k, v in out["sk_svc"].items()})

    RESULTS["sklearn_baselines"] = out
    return out


# ---------------------------------------------------------------- step 7
def cross_validation(X, ylog, y, cfgA, cfgB, best_lambda):
    header("7. UNCERTAINTY — 5-fold CV (train+val pool) mean ± std")
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    cv = {}
    folds = ev.kfold_indices(len(ylog), k=5, seed=SEED)

    def run(name, fit_predict):
        scores = [fit_predict(tr, va) for tr, va in folds]
        cv[name] = {"mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "scores": [float(s) for s in scores]}
        print(f"  {name}: {ev.format_pm(scores)}")

    # Task A (score = RMSE on log price)
    run("Ours tree reg RMSE_log", lambda tr, va: ev.rmse(
        ylog[va], DecisionTree(task="regression", criterion="mse",
                               max_depth=cfgA["max_depth"],
                               min_samples_leaf=cfgA["min_samples_leaf"])
        .fit(X[tr], ylog[tr]).predict(X[va])))
    run("sklearn tree reg RMSE_log", lambda tr, va: ev.rmse(
        ylog[va], DecisionTreeRegressor(
            max_depth=cfgA["max_depth"],
            min_samples_leaf=cfgA["min_samples_leaf"], random_state=SEED)
        .fit(X[tr], ylog[tr]).predict(X[va])))

    # Task B (score = F1; tier threshold recomputed per fold from its train)
    def tree_cls_fold(tr, va):
        tier_tr_f, thr_f = data_prep.make_tier_label(y[tr])
        tier_va_f, _ = data_prep.make_tier_label(y[va], thr_f)
        pred = DecisionTree(task="classification",
                            criterion=cfgB["criterion"],
                            max_depth=cfgB["max_depth"],
                            min_samples_leaf=cfgB["min_samples_leaf"]
                            ).fit(X[tr], tier_tr_f).predict(X[va])
        return ev.precision_recall_f1(tier_va_f, pred)[2]
    run("Ours tree cls F1", tree_cls_fold)

    def sk_tree_cls_fold(tr, va):
        tier_tr_f, thr_f = data_prep.make_tier_label(y[tr])
        tier_va_f, _ = data_prep.make_tier_label(y[va], thr_f)
        pred = DecisionTreeClassifier(
            criterion=cfgB["criterion"], max_depth=cfgB["max_depth"],
            min_samples_leaf=cfgB["min_samples_leaf"], random_state=SEED
        ).fit(X[tr], tier_tr_f).predict(X[va])
        return ev.precision_recall_f1(tier_va_f, pred)[2]
    run("sklearn tree cls F1", sk_tree_cls_fold)

    def pegasos_fold(tr, va):
        tier_tr_f, thr_f = data_prep.make_tier_label(y[tr])
        tier_va_f, _ = data_prep.make_tier_label(y[va], thr_f)
        Xtr_s, Xva_s = _std2(X[tr], X[va])
        pred = PegasosSVM(lambda_=best_lambda, n_iters=30_000, batch_size=64,
                          random_state=SEED, track_every=0
                          ).fit(Xtr_s, tier_tr_f).predict(Xva_s)
        return ev.precision_recall_f1(tier_va_f, pred)[2]
    run("Ours PegasosSVM F1", pegasos_fold)

    def sgd_fold(tr, va):
        tier_tr_f, thr_f = data_prep.make_tier_label(y[tr])
        tier_va_f, _ = data_prep.make_tier_label(y[va], thr_f)
        pred = make_pipeline(
            StandardScaler(),
            SGDClassifier(loss="hinge", alpha=best_lambda, max_iter=50,
                          tol=None, random_state=SEED)
        ).fit(X[tr], tier_tr_f).predict(X[va])
        return ev.precision_recall_f1(tier_va_f, pred)[2]
    run("sklearn SGD(hinge) F1", sgd_fold)

    RESULTS["cv"] = cv
    return cv


def _std2(Xtr, Xva):
    Xtr_s, Xva_s, _, _ = data_prep.standardize(Xtr, Xva)
    return Xtr_s, Xva_s


# ---------------------------------------------------------------- step 8
def analysis(df, names, tree_reg, tree_cls, svm_lin, Xs_te,
             X_te, ylog_te, y_te, tier_te, idx_te,
             pred_log_tree, tree_pred_cls, tree_proba, svm_pred, thr):
    header("8. FEATURE IMPORTANCES, SVM WEIGHTS, ERROR ANALYSIS")

    # importances + weights
    imp = tree_reg.feature_importances_
    top = np.argsort(imp)[::-1][:15]
    w = svm_lin.w
    topw = np.argsort(np.abs(w))[::-1][:15]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.barh([names[i] for i in top][::-1], imp[top][::-1], color=C_BLUE)
    _style(ax, "(a) Tree impurity-decrease importances (Task A)",
           "importance share", None)
    ax = axes[1]
    cols = [C_GREEN if w[i] > 0 else C_VERM for i in topw][::-1]
    ax.barh([names[i] for i in topw][::-1], w[topw][::-1], color=cols)
    ax.axvline(0, color=C_GRAY, lw=0.8)
    _style(ax, "(b) Pegasos SVM weights (Task B)",
           "weight (+ = premium, standardized)", None)
    savefig(fig, "fig_importance.png")
    RESULTS["top_features_tree"] = [(names[i], float(imp[i])) for i in top]
    RESULTS["top_weights_svm"] = [(names[i], float(w[i])) for i in topw]
    print("  top tree features:", [names[i] for i in top[:5]])
    print("  top |svm| weights:", [names[i] for i in topw[:5]])

    # confusion matrices + ROC
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (pred, title) in zip(
            axes[:2], [(tree_pred_cls, "Tree (ours)"),
                       (svm_pred, "Pegasos SVM (ours)")]):
        cm = ev.confusion_matrix(tier_te, pred)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=10)
        ax.set_xticks([0, 1], ["standard", "premium"])
        ax.set_yticks([0, 1], ["standard", "premium"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
        ax.set_title(f"{title} — confusion", fontsize=10)
    ax = axes[2]
    fpr, tpr = ev.roc_curve(tier_te, tree_proba)
    ax.plot(fpr, tpr, color=C_BLUE,
            label=f"tree AUC={ev.roc_auc(tier_te, tree_proba):.3f}")
    sc = svm_lin.decision_function(Xs_te)
    fpr, tpr = ev.roc_curve(tier_te, sc)
    ax.plot(fpr, tpr, color=C_ORANGE,
            label=f"SVM AUC={ev.roc_auc(tier_te, sc):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color=C_GRAY, lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "ROC (test)", "false positive rate", "true positive rate")
    savefig(fig, "fig_confusion_roc.png")

    # error analysis: relative error by price decile + by district
    pred_azn = np.exp(pred_log_tree)
    rel_err = np.abs(pred_azn - y_te) / y_te
    deciles = np.quantile(y_te, np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(y_te, deciles[1:-1]), 0, 9)
    med_err = [np.median(rel_err[bins == b]) * 100 for b in range(10)]

    df_te = df.iloc[idx_te]
    locs = df_te["location"].values
    top_locs = df_te["location"].value_counts().head(12).index
    loc_err = [(loc, float(np.median(rel_err[locs == loc]) * 100),
                int((locs == loc).sum())) for loc in top_locs]
    loc_err.sort(key=lambda t: -t[1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.bar(range(10), med_err, color=C_BLUE)
    ax.set_xticks(range(10), [f"{deciles[i]/1000:.0f}k" for i in range(10)],
                  rotation=45, fontsize=7)
    _style(ax, "(a) Tree relative error by price decile (test)",
           "price decile (lower edge)", "median |error| %")
    ax = axes[1]
    ax.barh([l for l, _, _ in loc_err][::-1],
            [e for _, e, _ in loc_err][::-1], color=C_ORANGE)
    _style(ax, "(b) Median |error| % in the 12 largest districts",
           "median |error| %", None)
    savefig(fig, "fig_error_analysis.png")
    RESULTS["error_by_decile_pct"] = med_err
    RESULTS["error_by_district_pct"] = loc_err

    # how many Task B mistakes sit near the threshold?
    for nm, pred in (("tree", tree_pred_cls), ("svm", svm_pred)):
        wrong = pred != tier_te
        near = np.abs(y_te - thr) / thr < 0.15
        share = float((wrong & near).sum() / max(wrong.sum(), 1))
        RESULTS[f"taskB_{nm}_errors_within_15pct_of_threshold"] = share
        print(f"  {nm}: {share:.1%} of misclassifications lie within "
              f"±15% of the {thr:,.0f} AZN threshold")


# ---------------------------------------------------------------- step 9
def bonuses(X_tr, ylog_tr, X_te, ylog_te, y_te, Xs_tr, Xs_te,
            df, idx_te, pred_log_tree, Xs_val, ylog_val):
    header("9. BONUSES — P8 Ridge + SVR (scratch), P9 RandomForest (scratch), "
           "P10 k-means/PCA")
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    out = {}
    # ---- P8: ridge via normal equations on standardized features ----
    ridge = RidgeRegression(lambda_=1.0)
    _, fit_s = ev.timed(ridge.fit, Xs_tr, ylog_tr)
    pred, pred_s = ev.timed(ridge.predict, Xs_te)
    TIMINGS["Ours: Ridge (normal eq.)"] = {"fit_s": fit_s, "pred_s": pred_s}
    out["ridge_ours"] = {"RMSE_log": ev.rmse(ylog_te, pred),
                         "MAE_log": ev.mae(ylog_te, pred),
                         "R2_log": ev.r2(ylog_te, pred),
                         "RMSE_AZN": ev.rmse(y_te, np.exp(pred)),
                         "MAE_AZN": ev.mae(y_te, np.exp(pred)),
                         "R2_AZN": ev.r2(y_te, np.exp(pred))}
    sk_r = Ridge(alpha=1.0).fit(Xs_tr, ylog_tr)
    skp = sk_r.predict(Xs_te)
    out["ridge_sklearn"] = {"RMSE_log": ev.rmse(ylog_te, skp),
                            "R2_log": ev.r2(ylog_te, skp)}
    print("  Ridge ours:", {k: round(v, 4) for k, v in out['ridge_ours'].items()})
    print("  Ridge sklearn RMSE_log:",
          round(out["ridge_sklearn"]["RMSE_log"], 6),
          "(diff vs ours:",
          f"{abs(out['ridge_ours']['RMSE_log']-out['ridge_sklearn']['RMSE_log']):.2e})")

    # ---- P8b: SVR via sub-gradient descent on the epsilon-insensitive loss ----
    from sklearn.svm import LinearSVR as SkLinearSVR
    eps_grid = [0.02, 0.05, 0.1, 0.2]
    eps_val = []
    for eps in eps_grid:
        m_ = LinearSVR(lambda_=1e-4, epsilon=eps, n_epochs=1_500,
                       eta0=0.5).fit(Xs_tr, ylog_tr)
        eps_val.append(ev.rmse(ylog_val, m_.predict(Xs_val)))
        print(f"  SVR epsilon={eps}  val RMSE_log={eps_val[-1]:.4f}")
    best_eps = eps_grid[int(np.argmin(eps_val))]
    svr = LinearSVR(lambda_=1e-4, epsilon=best_eps, n_epochs=1_500, eta0=0.5)
    _, fit_s = ev.timed(svr.fit, Xs_tr, ylog_tr)
    pred, pred_s = ev.timed(svr.predict, Xs_te)
    TIMINGS["Ours: LinearSVR (sub-grad)"] = {"fit_s": fit_s, "pred_s": pred_s}
    out["svr_ours"] = {"RMSE_log": ev.rmse(ylog_te, pred),
                       "MAE_log": ev.mae(ylog_te, pred),
                       "R2_log": ev.r2(ylog_te, pred),
                       "RMSE_AZN": ev.rmse(y_te, np.exp(pred)),
                       "MAE_AZN": ev.mae(y_te, np.exp(pred)),
                       "R2_AZN": ev.r2(y_te, np.exp(pred))}
    sk_svr = SkLinearSVR(C=1.0 / (1e-4 * len(ylog_tr)), epsilon=best_eps,
                         loss="epsilon_insensitive", max_iter=20_000,
                         random_state=SEED).fit(Xs_tr, ylog_tr)
    skp = sk_svr.predict(Xs_te)
    out["svr_sklearn"] = {"RMSE_log": ev.rmse(ylog_te, skp),
                          "R2_log": ev.r2(ylog_te, skp)}
    out["svr_epsilon"] = best_eps
    print(f"  SVR ours (eps={best_eps}):",
          {k: round(v, 4) for k, v in out["svr_ours"].items()})
    print("  SVR sklearn LinearSVR RMSE_log:",
          round(out["svr_sklearn"]["RMSE_log"], 4),
          f"| objective {svr.objective_history_[0][1]:.4f} -> "
          f"{svr.objective_history_[-1][1]:.4f}")

    # ---- P9: random forest from OUR trees ----
    rf = RandomForest(n_estimators=30, task="regression", max_depth=None,
                      min_samples_leaf=5, max_features="sqrt",
                      random_state=SEED)
    _, fit_s = ev.timed(rf.fit, X_tr, ylog_tr)
    scores = rf._staged_scores(X_te)
    curve = [ev.rmse(ylog_te, scores[:k].mean(axis=0))
             for k in range(1, rf.n_estimators + 1)]
    pred_rf, pred_s = ev.timed(rf.predict, X_te)
    TIMINGS["Ours: RandomForest (30 trees)"] = {"fit_s": fit_s,
                                                "pred_s": pred_s}
    out["rf_ours"] = {"RMSE_log": ev.rmse(ylog_te, pred_rf),
                      "MAE_log": ev.mae(ylog_te, pred_rf),
                      "R2_log": ev.r2(ylog_te, pred_rf),
                      "RMSE_AZN": ev.rmse(y_te, np.exp(pred_rf)),
                      "MAE_AZN": ev.mae(y_te, np.exp(pred_rf)),
                      "R2_AZN": ev.r2(y_te, np.exp(pred_rf))}
    sk_rf = RandomForestRegressor(n_estimators=30, max_depth=None,
                                  min_samples_leaf=5, max_features="sqrt",
                                  random_state=SEED, n_jobs=-1)
    _, fit_s = ev.timed(sk_rf.fit, X_tr, ylog_tr)
    skp, pred_s = ev.timed(sk_rf.predict, X_te)
    TIMINGS["sklearn: RandomForestRegressor"] = {"fit_s": fit_s,
                                                 "pred_s": pred_s}
    out["rf_sklearn"] = {"RMSE_log": ev.rmse(ylog_te, skp),
                         "R2_log": ev.r2(ylog_te, skp)}
    print("  RF ours RMSE_log:", round(out["rf_ours"]["RMSE_log"], 4),
          "| sklearn:", round(out["rf_sklearn"]["RMSE_log"], 4))

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(range(1, 31), curve, "-o", ms=3, color=C_GREEN,
            label="our forest")
    ax.axhline(out["rf_sklearn"]["RMSE_log"], color=C_GRAY, ls="--", lw=1,
               label="sklearn RF (30 trees)")
    ax.axhline(RESULTS["taskA_tree_test"]["RMSE_log"], color=C_ORANGE,
               ls=":", lw=1, label="our single tree")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Test RMSE(log) drops as trees are added", "number of trees",
           "test RMSE(log)")
    savefig(fig, "fig_rf_curve.png")

    # ---- P10: k-means on lat/lng (sklearn allowed for bonuses) ----
    km = KMeans(n_clusters=8, n_init=10, random_state=SEED)
    km.fit(df.iloc[np.setdiff1d(np.arange(len(df)), idx_te)][
        ["lat", "lng"]].values)
    cl_te = km.predict(df.iloc[idx_te][["lat", "lng"]].values)
    rel_err = np.abs(np.exp(pred_log_tree) - y_te) / y_te
    clus_err = [(int(c), float(np.median(rel_err[cl_te == c]) * 100),
                 int((cl_te == c).sum()))
                for c in range(8) if (cl_te == c).sum() > 50]
    clus_err.sort(key=lambda t: -t[1])
    out["kmeans_cluster_err_pct"] = clus_err
    print("  k-means clusters, median |error|% (test):",
          [(c, round(e, 1)) for c, e, _ in clus_err])

    # PCA of the standardized feature space, coloured by k-means segment
    pca = PCA(n_components=2, random_state=SEED)
    Z_te = pca.fit_transform(Xs_te)
    out["pca_explained_var"] = [float(v)
                                for v in pca.explained_variance_ratio_]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    sub = np.random.default_rng(SEED).choice(len(idx_te),
                                             min(8000, len(idx_te)),
                                             replace=False)
    ax.scatter(df.iloc[idx_te]["lng"].values[sub],
               df.iloc[idx_te]["lat"].values[sub],
               c=cl_te[sub], cmap="tab10", s=3, alpha=0.6)
    ax.set_xlim(49.3, 50.45)   # zoom to the Absheron peninsula,
    ax.set_ylim(40.15, 40.75)  # where >99% of listings sit
    _style(ax, "(a) k-means market segments (test, Baku area)", "longitude",
           "latitude")
    ax = axes[1]
    ax.bar([str(c) for c, _, _ in clus_err], [e for _, e, _ in clus_err],
           color=C_BLUE)
    _style(ax, "(b) Median |error| % by segment", "cluster",
           "median |error| %")
    ax = axes[2]
    ax.scatter(Z_te[sub, 0], Z_te[sub, 1], c=cl_te[sub], cmap="tab10",
               s=3, alpha=0.6)
    _style(ax, "(c) PCA of features, coloured by segment",
           f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)",
           f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    savefig(fig, "fig_kmeans.png")

    RESULTS["bonuses"] = out
    return out


# ---------------------------------------------------------------- step 10
def final_tables(mA_tree, skA, mB_tree, m_svm_lin, m_svm_rff, skB, bon):
    header("10. FINAL RESULTS TABLES")
    lines = []
    tA = ev.compare({
        "Ours: DecisionTree (mse)": mA_tree,
        "sklearn: DecisionTreeRegressor": skA["sk_tree_reg"],
        "Ours: Ridge (normal eq.) [B8]": bon["ridge_ours"],
        "Ours: LinearSVR (sub-grad) [B8]": bon["svr_ours"],
        "Ours: RandomForest x30 [B9]": bon["rf_ours"],
    }, title="TASK A — price regression (test set, log target)")
    lines.append(tA)
    print()
    tB = ev.compare({
        "Ours: DecisionTree": mB_tree,
        "Ours: PegasosSVM (linear)": m_svm_lin,
        "Ours: Pegasos + RFF": m_svm_rff,
        "sklearn: DecisionTreeClassifier": skB["sk_tree_cls"],
        "sklearn: SGDClassifier (hinge)": skB["sk_sgd"],
        "sklearn: LinearSVC": skB["sk_linsvc"],
        "sklearn: SVC-RBF (15k sub)": skB["sk_svc"],
    }, title="TASK B — price-tier classification (test set)")
    lines.append(tB)
    print()
    tT = ev.compare(
        {k: {"fit_s": round(v["fit_s"], 3), "pred_s": round(v["pred_s"], 3)}
         for k, v in TIMINGS.items()},
        title="FIT / PREDICT WALL-CLOCK TIME (seconds)")
    lines.append(tT)
    RESULTS["timings"] = TIMINGS

    os.makedirs(RESDIR, exist_ok=True)
    with open(os.path.join(RESDIR, "tables.md"), "w", encoding="utf-8") as f:
        f.write("```\n" + "\n\n".join(lines) + "\n```\n")
    with open(os.path.join(RESDIR, "results.json"), "w",
              encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  saved {RESDIR}/results.json and {RESDIR}/tables.md")


# ---------------------------------------------------------------------------
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    np.random.seed(SEED)
    os.makedirs(FIGDIR, exist_ok=True)
    t_start = time.perf_counter()

    run_sanity_tests()
    (df, X, y, names, X_tr, y_tr, X_val, y_val, X_te, y_te,
     idx_tr, idx_val, idx_te, tier_tr, tier_val, tier_te,
     thr) = load_and_prepare()

    ylog_tr, ylog_val, ylog_te = np.log(y_tr), np.log(y_val), np.log(y_te)
    Xs_tr, Xs_val, Xs_te, _, _ = data_prep.standardize(X_tr, X_val, X_te)

    tree_reg, dA, lA, mA_tree, pred_log_tree = task_a_tree(
        X_tr, ylog_tr, X_val, ylog_val, X_te, ylog_te, y_te)
    tree_cls, critB, dB, mB_tree, tree_pred_cls, tree_proba = task_b_tree(
        X_tr, tier_tr, X_val, tier_val, X_te, tier_te)
    (svm_lin, m_svm_lin, svm_rff, rff, m_svm_rff, best_lambda, best_gamma,
     svm_pred, svm_pred_rff) = task_b_svm(
        Xs_tr, tier_tr, Xs_val, tier_val, Xs_te, tier_te)

    skl = sklearn_baselines(
        X_tr, ylog_tr, X_te, ylog_te, y_te, tier_tr, tier_te, X_val,
        Xs_tr, Xs_te, RESULTS["taskA_tree_config"],
        RESULTS["taskB_tree_config"], best_lambda, best_gamma)

    X_pool = np.vstack([X_tr, X_val])
    y_pool = np.concatenate([y_tr, y_val])
    cross_validation(X_pool, np.log(y_pool), y_pool,
                     RESULTS["taskA_tree_config"],
                     RESULTS["taskB_tree_config"], best_lambda)

    analysis(df, names, tree_reg, tree_cls, svm_lin, Xs_te, X_te, ylog_te,
             y_te, tier_te, idx_te, pred_log_tree, tree_pred_cls,
             tree_proba, svm_pred, thr)

    bon = bonuses(X_tr, ylog_tr, X_te, ylog_te, y_te, Xs_tr, Xs_te,
                  df, idx_te, pred_log_tree, Xs_val, ylog_val)

    final_tables(mA_tree, skl, mB_tree, m_svm_lin, m_svm_rff, skl, bon)

    print(f"\nDONE in {time.perf_counter() - t_start:.0f} s. "
          f"Figures: {FIGDIR}/  Numbers: {RESDIR}/results.json")


if __name__ == "__main__":
    main()
