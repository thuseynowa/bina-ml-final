"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Supports both tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

No scikit-learn here. The split search is vectorised: for every feature
the samples are sorted once per node and all candidate thresholds are
scored in one pass using cumulative class counts (classification) or
cumulative sums of y and y² (regression). This keeps the tree fast
enough to run full hyperparameter sweeps on ~40k training rows.

Split objective (same weighting scheme as scikit-learn, so
`min_impurity_decrease` is directly comparable):

    decrease = N_node/N_total * ( I(node)
               - N_left/N_node * I(left) - N_right/N_node * I(right) )

with I = Gini, entropy (bits) or variance (MSE) of the node.
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree."""

    __slots__ = ("feature", "threshold", "left", "right", "value",
                 "n_samples", "impurity")

    def __init__(self):
        self.feature = None      # int: feature index (internal nodes)
        self.threshold = None    # float: go left if x[feature] <= threshold
        self.left = None         # Node
        self.right = None        # Node
        self.value = None        # leaf: mean (regression) or class counts
        self.n_samples = 0
        self.impurity = 0.0

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: int | None = None,   # features tried per split (None = all)
        random_state: int = 42,
    ):
        if task == "regression" and criterion != "mse":
            criterion = "mse"
        if task == "classification" and criterion == "mse":
            raise ValueError("mse criterion is for regression")
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        self.root: Node | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_: int = 0
        self.feature_importances_: np.ndarray | None = None
        self.n_leaves_: int = 0
        self.depth_: int = 0

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        """Gini / entropy of encoded labels, or variance for regression."""
        y = np.asarray(y)
        if self.task == "regression":
            return float(np.var(y))
        counts = np.bincount(y, minlength=len(self.classes_))
        p = counts / counts.sum()
        if self.criterion == "gini":
            return float(1.0 - np.sum(p ** 2))
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    # --- vectorised split search ---------------------------------------------
    def _node_impurity_arrays_cls(self, counts_left, counts_right, nl, nr):
        """Impurity of left/right children for every candidate position."""
        with np.errstate(divide="ignore", invalid="ignore"):
            pl = counts_left / nl[:, None]
            pr = counts_right / nr[:, None]
            if self.criterion == "gini":
                il = 1.0 - np.sum(pl ** 2, axis=1)
                ir = 1.0 - np.sum(pr ** 2, axis=1)
            else:  # entropy (bits)
                il = -np.sum(np.where(pl > 0, pl * np.log2(pl), 0.0), axis=1)
                ir = -np.sum(np.where(pr > 0, pr * np.log2(pr), 0.0), axis=1)
        return il, ir

    def _best_split(self, X, y, parent_impurity):
        """
        Return (feature, threshold, decrease) of the split with the largest
        weighted impurity decrease, or None if no valid split exists.
        """
        n, n_feat = X.shape
        msl = self.min_samples_leaf
        best_gain = 0.0
        best = None

        if self.task == "classification":
            onehot = np.eye(len(self.classes_), dtype=np.float64)[y]

        if self.max_features is not None and self.max_features < n_feat:
            features = self._rng.choice(n_feat, size=self.max_features,
                                        replace=False)
        else:
            features = range(n_feat)

        for f in features:
            order = np.argsort(X[:, f], kind="stable")
            xs = X[order, f]
            # candidate split after position i (0-based): left = [0..i]
            diff = xs[1:] != xs[:-1]
            pos = np.nonzero(diff)[0]
            pos = pos[(pos + 1 >= msl) & (n - pos - 1 >= msl)]
            if pos.size == 0:
                continue
            nl = (pos + 1).astype(np.float64)
            nr = n - nl

            if self.task == "classification":
                cum = np.cumsum(onehot[order], axis=0)
                counts_left = cum[pos]
                counts_right = cum[-1] - counts_left
                il, ir = self._node_impurity_arrays_cls(
                    counts_left, counts_right, nl, nr)
            else:
                ys = y[order].astype(np.float64)
                cy = np.cumsum(ys)
                cy2 = np.cumsum(ys * ys)
                sl, sl2 = cy[pos], cy2[pos]
                sr, sr2 = cy[-1] - sl, cy2[-1] - sl2
                il = sl2 / nl - (sl / nl) ** 2
                ir = sr2 / nr - (sr / nr) ** 2

            weighted = (nl * il + nr * ir) / n
            gains = parent_impurity - weighted
            j = int(np.argmax(gains))
            if gains[j] > best_gain + 1e-12:
                best_gain = float(gains[j])
                thr = 0.5 * (xs[pos[j]] + xs[pos[j] + 1])
                best = (f, float(thr), best_gain)
        return best

    # --- grow / predict -------------------------------------------------------
    def _leaf(self, y) -> Node:
        node = Node()
        node.n_samples = len(y)
        node.impurity = self._impurity(y)
        if self.task == "regression":
            node.value = float(np.mean(y))
        else:
            node.value = np.bincount(y, minlength=len(self.classes_)).astype(float)
        self.n_leaves_ += 1
        return node

    def _grow(self, X, y, depth: int) -> Node:
        self.depth_ = max(self.depth_, depth)
        n = len(y)
        impurity = self._impurity(y)

        stop = (
            n < self.min_samples_split
            or n < 2 * self.min_samples_leaf
            or impurity <= 1e-12
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        if not stop:
            split = self._best_split(X, y, impurity)
            # sklearn-equivalent pruning rule
            if split is not None:
                f, thr, gain = split
                if (n / self._n_total) * gain < self.min_impurity_decrease:
                    split = None
            if split is not None:
                f, thr, gain = split
                mask = X[:, f] <= thr
                node = Node()
                node.feature, node.threshold = f, thr
                node.n_samples, node.impurity = n, impurity
                self._importance_acc[f] += (n / self._n_total) * gain
                node.left = self._grow(X[mask], y[mask], depth + 1)
                node.right = self._grow(X[~mask], y[~mask], depth + 1)
                return node
        return self._leaf(y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self.n_leaves_ = 0
        self.depth_ = 0
        self._n_total = len(y)
        self._rng = np.random.default_rng(self.random_state)
        self._importance_acc = np.zeros(self.n_features_)
        if self.task == "classification":
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            y = y.astype(np.float64)
        self.root = self._grow(X, y, depth=0)
        total = self._importance_acc.sum()
        self.feature_importances_ = (
            self._importance_acc / total if total > 0 else self._importance_acc
        )
        return self

    def _predict_one(self, x, node: Node):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def _leaf_values(self, X) -> list:
        X = np.asarray(X, dtype=np.float64)
        return [self._predict_one(x, self.root) for x in X]

    def predict(self, X):
        vals = self._leaf_values(X)
        if self.task == "regression":
            return np.asarray(vals, dtype=float)
        counts = np.vstack(vals)
        return self.classes_[np.argmax(counts, axis=1)]

    def predict_proba(self, X):
        """Classification only: class probabilities from leaf frequencies."""
        if self.task != "classification":
            raise ValueError("predict_proba is for classification")
        counts = np.vstack(self._leaf_values(X))
        return counts / counts.sum(axis=1, keepdims=True)
