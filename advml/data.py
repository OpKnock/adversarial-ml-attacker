"""Synthetic in-memory dataset generation."""

from __future__ import annotations

import numpy as np


def make_synthetic_dataset(
    n_samples: int = 600,
    n_features: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two overlapping Gaussian clusters normalized to [0, 1]."""
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0
    center0 = rng.uniform(0.30, 0.40, size=n_features)
    center1 = rng.uniform(0.60, 0.70, size=n_features)
    cov = np.eye(n_features) * 0.05
    X0 = rng.multivariate_normal(center0, cov, size=n0)
    X1 = rng.multivariate_normal(center1, cov, size=n1)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]
    return np.clip(X, 0.0, 1.0), y


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically split data into train and test sets."""
    rng = np.random.default_rng(seed)
    n_train = int(round(len(X) * train_frac))
    perm = rng.permutation(len(X))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of correctly predicted samples."""
    if len(X) == 0:
        return 0.0
    return float(np.mean(model.predict(X) == y))
