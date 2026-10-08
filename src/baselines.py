"""
baselines.py — BONUS (Problem 8): from-scratch linear baselines.

RidgeRegression solves the normal equations directly:

    w_hat = (X^T X + lambda * I)^(-1) X^T y

(the intercept column is excluded from the penalty, as usual).
"""

from __future__ import annotations

import numpy as np


class RidgeRegression:
    """Linear / Ridge regression via the normal equations (lambda=0 → OLS)."""

    def __init__(self, lambda_: float = 1.0):
        self.lambda_ = lambda_
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])        # fold in the intercept
        reg = self.lambda_ * np.eye(d + 1)
        reg[-1, -1] = 0.0                            # don't penalise the bias
        theta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
        self.w, self.b = theta[:-1], float(theta[-1])
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.w + self.b
