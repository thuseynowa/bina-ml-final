"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

Minimises the regularised hinge-loss primal

    J(w, b) = (lambda/2) ||w||^2 + (1/n) sum_i max(0, 1 - y_i (w·x_i + b))

with labels y in {-1, +1}, by (mini-batch) stochastic sub-gradient
descent with the Pegasos schedule eta_t = 1 / (lambda * t):

    hinge active (y_i (w·x_i + b) < 1):
        w <- (1 - eta_t*lambda) w + eta_t * mean(y_i x_i)
        b <- b + eta_t * mean(y_i)              (bias not regularised)
    otherwise:
        w <- (1 - eta_t*lambda) w

Non-linear extension: RANDOM FOURIER FEATURES (Rahimi & Recht, 2007) —
`RandomFourierFeatures` maps X to z(x) = sqrt(2/D) cos(Wx + c) whose
inner products approximate the RBF kernel exp(-gamma ||x - x'||^2); the
same LINEAR Pegasos solver then runs unchanged on z(X). We chose RFF
over kernelised Pegasos because training stays O(n·D) instead of O(n²)
on ~40k samples, with one knob (D) trading accuracy for compute.

scikit-learn's SVC/SGDClassifier are BASELINES only — nothing from
sklearn is used inside these classes. Standardize features first
(data_prep.standardize); the hinge loss is scale-sensitive.
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (~ 1/(n*C))
        n_iters: int = 30_000,     # sub-gradient steps
        batch_size: int = 64,      # mini-batch size per step
        random_state: int = 42,
        track_every: int = 500,    # record J(w,b) every k steps (0 = off)
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.random_state = random_state
        self.track_every = track_every
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.classes_: np.ndarray | None = None
        self.objective_history_: list[tuple[int, float]] = []
        self.support_: np.ndarray | None = None   # training margin <= 1

    def _to_pm1(self, y):
        """Map any 2-class labels to {-1, +1}; store the class order."""
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("PegasosSVM is a binary classifier")
        return np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0)

    def objective(self, X, y_pm1) -> float:
        """J(w, b) on the given (already ±1-encoded) data."""
        margins = y_pm1 * (X @ self.w + self.b)
        hinge = np.maximum(0.0, 1.0 - margins).mean()
        return float(0.5 * self.lambda_ * self.w @ self.w + hinge)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y_pm1 = self._to_pm1(y)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        self.w = np.zeros(d)
        self.b = 0.0
        self.objective_history_ = []

        for t in range(1, self.n_iters + 1):
            eta = 1.0 / (self.lambda_ * t)
            idx = rng.integers(0, n, size=min(self.batch_size, n))
            Xb, yb = X[idx], y_pm1[idx]
            active = yb * (Xb @ self.w + self.b) < 1.0

            self.w *= (1.0 - eta * self.lambda_)
            if active.any():
                self.w += (eta / len(idx)) * (yb[active] @ Xb[active])
                self.b += (eta / len(idx)) * yb[active].sum()

            if self.track_every and (t % self.track_every == 0 or t == 1):
                self.objective_history_.append((t, self.objective(X, y_pm1)))

        # support vectors: training points on or inside the margin
        margins = y_pm1 * (X @ self.w + self.b)
        self.support_ = np.nonzero(margins <= 1.0 + 1e-9)[0]
        return self

    @property
    def margin_width_(self) -> float:
        """Geometric margin width 2 / ||w||."""
        norm = float(np.linalg.norm(self.w))
        return float("inf") if norm == 0 else 2.0 / norm

    def decision_function(self, X):
        """Signed margin w·x + b for each row."""
        return np.asarray(X, dtype=np.float64) @ self.w + self.b

    def predict(self, X):
        """Class labels (sign of the decision function → original labels)."""
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]


class RandomFourierFeatures:
    """
    Approximate the RBF kernel exp(-gamma ||x - x'||^2) with the explicit
    map z(x) = sqrt(2/D) * cos(W x + c), W ~ N(0, 2*gamma I), c ~ U[0, 2pi].
    fit() draws (W, c); transform() applies the map.
    """

    def __init__(self, n_components: int = 300, gamma: float = 0.1,
                 random_state: int = 42):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.W: np.ndarray | None = None
        self.c: np.ndarray | None = None

    def fit(self, X):
        d = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.W = rng.normal(0.0, np.sqrt(2.0 * self.gamma),
                            size=(d, self.n_components))
        self.c = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        proj = np.asarray(X, dtype=np.float64) @ self.W + self.c
        return np.sqrt(2.0 / self.n_components) * np.cos(proj)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
