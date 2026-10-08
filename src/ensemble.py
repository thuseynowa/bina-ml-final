"""
ensemble.py — BONUS (Problem 9): a random forest built from OUR OWN trees.

Bagging (bootstrap resampling) + PER-SPLIT feature subsampling (the
tree's own `max_features`, same scheme as sklearn's forests), with
src.decision_tree.DecisionTree as the weak learner. Works for both
tasks; predictions are averaged (regression) or majority-voted via
averaged leaf probabilities (classification).
"""

from __future__ import annotations

import numpy as np

from .decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 30,
        task: str = "regression",
        criterion: str | None = None,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: str | int = "sqrt",   # per-tree feature subsample
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.task = task
        self.criterion = criterion or ("mse" if task == "regression" else "gini")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_: list[DecisionTree] = []
        self.classes_: np.ndarray | None = None

    def _n_features_per_tree(self, d: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        if isinstance(self.max_features, int):
            return min(d, self.max_features)
        return d

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n, d = X.shape
        if self.task == "classification":
            self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        k = self._n_features_per_tree(d)
        self.trees_ = []
        for i in range(self.n_estimators):
            rows = rng.integers(0, n, size=n)                 # bootstrap
            tree = DecisionTree(
                task=self.task, criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=k,                # per-split feature subsample
                random_state=self.random_state + i,
            ).fit(X[rows], y[rows])
            self.trees_.append(tree)
        return self

    def _staged_scores(self, X):
        """Per-tree raw scores: predictions (reg) or P(class=1) (cls)."""
        X = np.asarray(X, dtype=np.float64)
        out = []
        for tree in self.trees_:
            if self.task == "regression":
                out.append(tree.predict(X))
            else:
                out.append(tree.predict_proba(X)[:, 1])
        return np.vstack(out)

    def predict(self, X, n_trees: int | None = None):
        scores = self._staged_scores(X)[: (n_trees or self.n_estimators)]
        avg = scores.mean(axis=0)
        if self.task == "regression":
            return avg
        return self.classes_[(avg >= 0.5).astype(int)]

    def predict_proba(self, X):
        p1 = self._staged_scores(X).mean(axis=0)
        return np.column_stack([1 - p1, p1])
