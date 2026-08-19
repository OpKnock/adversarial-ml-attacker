"""Adversarial training defense.

Retrains a model on a mixture of clean and adversarial examples, then
measures how much the retrained model resists the same attack.
"""

from __future__ import annotations

import numpy as np

from .attacks import attack_success_rate, pgd
from .models import LogisticRegression


def craft_adversarial_batch(
    model,
    X: np.ndarray,
    y: np.ndarray,
    attack=pgd,
    frac: float = 0.3,
    seed: int = 3,
    **attack_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Adversarially perturb a random subset of the training set."""
    rng = np.random.default_rng(seed)
    n = int(round(len(X) * frac))
    indices = rng.choice(len(X), size=min(n, len(X)), replace=False)
    adv = np.array(
        [attack(model, X[i], int(y[i]), **attack_kwargs) for i in indices]
    )
    return adv, y[indices]


def adversarial_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_cls=LogisticRegression,
    frac: float = 0.3,
    seed: int = 3,
) -> dict:
    """Train a baseline and an adversarially-trained model, then compare."""
    baseline = model_cls(seed=1).fit(X_train, y_train)
    adv_X, adv_y = craft_adversarial_batch(
        baseline, X_train, y_train, frac=frac, seed=seed, eps=0.15
    )
    X_aug = np.vstack([X_train, adv_X])
    y_aug = np.concatenate([y_train, adv_y])
    robust = model_cls(seed=1).fit(X_aug, y_aug)

    baseline_success, _ = attack_success_rate(
        baseline, X_test, y_test, pgd, eps=0.15
    )
    robust_success, _ = attack_success_rate(
        robust, X_test, y_test, pgd, eps=0.15
    )
    return {
        "baseline_accuracy": baseline.accuracy(X_test, y_test),
        "robust_accuracy": robust.accuracy(X_test, y_test),
        "baseline_pgd_success": baseline_success,
        "robust_pgd_success": robust_success,
        "augmented_samples": int(len(X_aug)),
    }
