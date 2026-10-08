"""
Sanity tests: our from-scratch models vs scikit-learn on small,
fully-determined examples. Run from the repo root:

    python -m tests.test_models

Every assert must pass; the script prints PASS lines and exits 0.
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.baselines import RidgeRegression
from src.decision_tree import DecisionTree
from src.evaluate import (confusion_matrix, mae, precision_recall_f1, r2,
                          rmse, roc_auc)
from src.svm import PegasosSVM

RNG = np.random.default_rng(0)


def test_tree_matches_sklearn_on_tiny_example():
    """Required check: identical splits on a tiny, fully-determined set."""
    # 1-D data with one unambiguous best threshold at 3.5
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    ours = DecisionTree(task="classification", criterion="gini").fit(X, y)
    sk = DecisionTreeClassifier(criterion="gini", random_state=0).fit(X, y)
    assert ours.root.feature == sk.tree_.feature[0] == 0
    assert abs(ours.root.threshold - sk.tree_.threshold[0]) < 1e-9, (
        ours.root.threshold, sk.tree_.threshold[0])
    assert np.array_equal(ours.predict(X), sk.predict(X))

    # 2-D deterministic example — split feature AND threshold must agree
    X2 = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [3.0, 40.0],
                   [4.0, 50.0], [5.0, 5.0], [6.0, 15.0], [7.0, 25.0]])
    y2 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    ours2 = DecisionTree(task="classification", criterion="entropy",
                         max_depth=2).fit(X2, y2)
    sk2 = DecisionTreeClassifier(criterion="entropy", max_depth=2,
                                 random_state=0).fit(X2, y2)
    assert ours2.root.feature == sk2.tree_.feature[0]
    assert abs(ours2.root.threshold - sk2.tree_.threshold[0]) < 1e-9
    print("PASS tree matches sklearn splits on tiny deterministic examples")


def test_tree_classification_accuracy_close_to_sklearn():
    X, y = make_classification(n_samples=600, n_features=10, n_informative=5,
                               random_state=1)
    Xtr, Xte, ytr, yte = X[:450], X[450:], y[:450], y[450:]
    for crit in ("gini", "entropy"):
        ours = DecisionTree(task="classification", criterion=crit,
                            max_depth=6).fit(Xtr, ytr)
        sk = DecisionTreeClassifier(criterion=crit, max_depth=6,
                                    random_state=0).fit(Xtr, ytr)
        acc_ours = np.mean(ours.predict(Xte) == yte)
        acc_sk = np.mean(sk.predict(Xte) == yte)
        assert abs(acc_ours - acc_sk) < 0.05, (crit, acc_ours, acc_sk)
    print("PASS tree test accuracy within 5pp of sklearn (gini & entropy)")


def test_tree_regression_close_to_sklearn():
    X, y = make_regression(n_samples=600, n_features=8, noise=10.0,
                           random_state=2)
    Xtr, Xte, ytr, yte = X[:450], X[450:], y[:450], y[450:]
    ours = DecisionTree(task="regression", criterion="mse",
                        max_depth=6).fit(Xtr, ytr)
    sk = DecisionTreeRegressor(max_depth=6, random_state=0).fit(Xtr, ytr)
    assert abs(r2(yte, ours.predict(Xte)) - r2(yte, sk.predict(Xte))) < 0.05
    print("PASS regression tree R2 within 0.05 of sklearn")


def test_tree_hyperparameters_change_the_tree():
    X, y = make_classification(n_samples=400, n_features=6, random_state=3)
    full = DecisionTree(task="classification").fit(X, y)
    shallow = DecisionTree(task="classification", max_depth=2).fit(X, y)
    assert shallow.depth_ <= 2 < full.depth_
    big_leaf = DecisionTree(task="classification",
                            min_samples_leaf=50).fit(X, y)
    assert big_leaf.n_leaves_ < full.n_leaves_
    coarse = DecisionTree(task="classification",
                          min_samples_split=200).fit(X, y)
    assert coarse.n_leaves_ < full.n_leaves_
    pruned = DecisionTree(task="classification",
                          min_impurity_decrease=0.05).fit(X, y)
    assert pruned.n_leaves_ < full.n_leaves_
    print("PASS every tree hyperparameter actually changes the tree")


def test_pegasos_objective_decreases_and_matches_linear_svm():
    X, y = make_classification(n_samples=800, n_features=12, n_informative=8,
                               class_sep=1.5, random_state=4)
    X = (X - X.mean(0)) / X.std(0)
    Xtr, Xte, ytr, yte = X[:600], X[600:], y[:600], y[600:]
    svm = PegasosSVM(lambda_=1e-3, n_iters=20_000, random_state=0).fit(Xtr, ytr)
    hist = [v for _, v in svm.objective_history_]
    assert hist[-1] < hist[0], "objective did not decrease"
    acc_ours = np.mean(svm.predict(Xte) == yte)
    sk = LinearSVC(C=1.0 / (1e-3 * len(ytr)), dual=True, max_iter=20_000,
                   random_state=0).fit(Xtr, ytr)
    acc_sk = np.mean(sk.predict(Xte) == yte)
    assert abs(acc_ours - acc_sk) < 0.05, (acc_ours, acc_sk)
    assert len(svm.support_) > 0 and svm.margin_width_ > 0
    print("PASS Pegasos converges and matches LinearSVC within 5pp")


def test_ridge_matches_sklearn():
    X, y = make_regression(n_samples=300, n_features=10, noise=5.0,
                           random_state=5)
    ours = RidgeRegression(lambda_=2.0).fit(X, y)
    sk = Ridge(alpha=2.0).fit(X, y)
    assert np.allclose(ours.w, sk.coef_, atol=1e-6)
    assert abs(ours.b - sk.intercept_) < 1e-6
    print("PASS from-scratch Ridge equals sklearn Ridge to 1e-6")


def test_metrics_match_sklearn():
    y_true = RNG.integers(0, 2, size=500)
    scores = RNG.normal(size=500) + y_true * 1.2
    y_pred = (scores > 0.5).astype(int)
    y_reg_true = RNG.normal(size=500) * 10
    y_reg_pred = y_reg_true + RNG.normal(size=500) * 3

    assert abs(rmse(y_reg_true, y_reg_pred)
               - np.sqrt(skm.mean_squared_error(y_reg_true, y_reg_pred))) < 1e-9
    assert abs(mae(y_reg_true, y_reg_pred)
               - skm.mean_absolute_error(y_reg_true, y_reg_pred)) < 1e-9
    assert abs(r2(y_reg_true, y_reg_pred)
               - skm.r2_score(y_reg_true, y_reg_pred)) < 1e-9
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    assert abs(p - skm.precision_score(y_true, y_pred)) < 1e-9
    assert abs(r - skm.recall_score(y_true, y_pred)) < 1e-9
    assert abs(f1 - skm.f1_score(y_true, y_pred)) < 1e-9
    assert np.array_equal(confusion_matrix(y_true, y_pred),
                          skm.confusion_matrix(y_true, y_pred))
    assert abs(roc_auc(y_true, scores)
               - skm.roc_auc_score(y_true, scores)) < 1e-9
    print("PASS all from-scratch metrics equal sklearn to 1e-9")


def run_all_tests():
    test_tree_matches_sklearn_on_tiny_example()
    test_tree_classification_accuracy_close_to_sklearn()
    test_tree_regression_close_to_sklearn()
    test_tree_hyperparameters_change_the_tree()
    test_pegasos_objective_decreases_and_matches_linear_svm()
    test_ridge_matches_sklearn()
    test_metrics_match_sklearn()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_all_tests()
