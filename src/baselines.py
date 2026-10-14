"""
baselines.py — BONUS (Problem 8): from-scratch linear baselines.

RidgeRegression solves the normal equations directly:

    w_hat = (X^T X + lambda * I)^(-1) X^T y

(the intercept column is excluded from the penalty, as usual).

LinearSVR minimises the regularised epsilon-insensitive loss

    J(w, b) = (lambda/2) ||w||^2
              + (1/n) sum_i max(0, |y_i - (w·x_i + b)| - epsilon)

by full-batch sub-gradient descent (the regression twin of Pegasos):
residuals inside the epsilon-tube contribute zero gradient, the rest
contribute -sign(r_i) x_i. The same trick as the SVM — no QP solver.
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


class LinearSVR:
    """
    Support-vector regression via full-batch sub-gradient descent on the
    epsilon-insensitive loss (see module docstring). Features should be
    standardized; y is centred internally so b starts at mean(y).

      lambda_   L2 regularisation (C-equivalent: C = 1/(n*lambda))
      epsilon   half-width of the no-penalty tube around the prediction
      n_epochs  full-batch gradient steps
      eta0      step size, decayed as eta0 / sqrt(t)
    """

    def __init__(self, lambda_: float = 1e-4, epsilon: float = 0.05,
                 n_epochs: int = 2_000, eta0: float = 0.5):
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.n_epochs = n_epochs
        self.eta0 = eta0
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.objective_history_: list[tuple[int, float]] = []

    def objective(self, X, y) -> float:
        r = y - (X @ self.w + self.b)
        tube = np.maximum(0.0, np.abs(r) - self.epsilon).mean()
        return float(0.5 * self.lambda_ * self.w @ self.w + tube)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = float(y.mean())
        self.objective_history_ = []
        for t in range(1, self.n_epochs + 1):
            eta = self.eta0 / np.sqrt(t)
            r = y - (X @ self.w + self.b)
            # sub-gradient of the tube loss wrt the prediction
            g = -np.sign(r) * (np.abs(r) > self.epsilon)
            self.w -= eta * (self.lambda_ * self.w + X.T @ g / n)
            self.b -= eta * g.mean()
            if t % 100 == 0 or t == 1:
                self.objective_history_.append((t, self.objective(X, y)))
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.w + self.b
