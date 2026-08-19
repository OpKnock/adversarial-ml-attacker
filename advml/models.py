"""Small interpretable classifiers implemented in pure NumPy.

Both models expose ``predict``, ``predict_proba`` and ``accuracy`` so that
attacks, detectors and reports can use them interchangeably.
"""

from __future__ import annotations

import numpy as np

from .data import accuracy


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


class LogisticRegression:
    """Binary logistic regression trained by gradient descent."""

    def __init__(
        self, learning_rate: float = 0.5, epochs: int = 600, seed: int = 0
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.weights = rng.normal(0.0, 0.05, size=d)
        self.bias = 0.0
        for _ in range(self.epochs):
            p = self.predict_proba(X)
            error = p - y
            gradient_w = X.T @ error / n
            gradient_b = float(np.mean(error))
            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b
        return self

    def logit(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("model is not fitted")
        return X @ self.weights + self.bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logit(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def gradient_wrt_input(self, x: np.ndarray, y_true: int) -> np.ndarray:
        """Analytic gradient of cross-entropy loss w.r.t. the input."""
        if self.weights is None:
            raise RuntimeError("model is not fitted")
        p = float(_sigmoid(float(x @ self.weights) + self.bias))
        return (p - float(y_true)) * self.weights

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy(self, X, y)


class DecisionTree:
    """A tiny binary decision tree trained with Gini impurity."""

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_split: int = 5,
        seed: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.root = self._build(X, y, depth=0)
        return self

    def _build(
        self, X: np.ndarray, y: np.ndarray, depth: int
    ) -> tuple:
        rng = np.random.default_rng(self.seed + depth)
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return ("leaf", float(np.mean(y)), len(y))
        best = None
        best_gain = -1.0
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            if len(thresholds) < 2:
                continue
            candidates = (thresholds[:-1] + thresholds[1:]) / 2
            for threshold in candidates:
                left = X[:, feature] <= threshold
                if np.all(left) or not np.any(left):
                    continue
                gain = self._gini_gain(y, left)
                if gain > best_gain:
                    best_gain = gain
                    best = (feature, threshold)
        if best is None:
            return ("leaf", float(np.mean(y)), len(y))
        feature, threshold = best
        left_mask = X[:, feature] <= threshold
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return ("split", feature, threshold, left, right)

    @staticmethod
    def _gini_gain(y: np.ndarray, left_mask: np.ndarray) -> float:
        def gini(labels: np.ndarray) -> float:
            if len(labels) == 0:
                return 0.0
            counts = np.bincount(labels)
            proportions = counts / counts.sum()
            return 1.0 - float(np.sum(proportions ** 2))

        n = len(y)
        left_weight = float(np.mean(left_mask))
        return gini(y) - (
            left_weight * gini(y[left_mask])
            + (1.0 - left_weight) * gini(y[~left_mask])
        )

    def _predict_proba_row(self, x: np.ndarray, node) -> float:
        if node[0] == "leaf":
            return node[1]
        _, feature, threshold, left, right = node
        if x[feature] <= threshold:
            return self._predict_proba_row(x, left)
        return self._predict_proba_row(x, right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("model is not fitted")
        return np.array([self._predict_proba_row(x, self.root) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy(self, X, y)
