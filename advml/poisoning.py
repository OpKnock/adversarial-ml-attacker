"""Label-poisoning simulation for training data."""

from __future__ import annotations

import numpy as np

from .data import accuracy
from .models import LogisticRegression


def craft_label_poison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_poison: int,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Craft n label-flipped samples.

    Picks class-1 samples closest to the class-0 centroid (the ones most
    likely to still be "natural" after a label flip) and nudges them toward
    the class-0 centroid, imitating a poisoned training set.
    """
    rng = np.random.default_rng(seed)
    class1 = X_train[y_train == 1]
    if len(class1) == 0:
        return X_train.copy(), y_train.copy()
    centroid0 = np.mean(X_train[y_train == 0], axis=0) if np.any(y_train == 0) else 0.5
    distances = np.linalg.norm(class1 - centroid0, axis=1)
    order = np.argsort(distances)
    picked = class1[order[: min(n_poison, len(class1))]].copy()
    picked = picked + 0.15 * (centroid0 - picked)
    picked = np.clip(picked, 0.0, 1.0)
    X_poisoned = np.vstack([X_train, picked])
    y_poisoned = np.concatenate(
        [y_train, np.zeros(len(picked), dtype=int)]
    )
    perm = rng.permutation(len(X_poisoned))
    return X_poisoned[perm], y_poisoned[perm]


def measure_poison_effect(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_poison: int = 20,
    seed: int = 7,
    model_cls=LogisticRegression,
) -> dict:
    """Train on clean vs. poisoned data and report the accuracy drop."""
    clean_model = model_cls(seed=1).fit(X_train, y_train)
    X_poisoned, y_poisoned = craft_label_poison(
        X_train, y_train, n_poison, seed=seed
    )
    poisoned_model = model_cls(seed=1).fit(X_poisoned, y_poisoned)
    baseline = accuracy(clean_model, X_test, y_test)
    poisoned = accuracy(poisoned_model, X_test, y_test)
    return {
        "baseline_accuracy": baseline,
        "poisoned_accuracy": poisoned,
        "accuracy_drop": baseline - poisoned,
        "poisoned_samples": int(len(X_poisoned) - len(X_train)),
        "test_samples": int(len(X_test)),
    }
