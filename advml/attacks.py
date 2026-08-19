"""Adversarial example attacks.

FGSM and PGD are gradient-based (usable with ``LogisticRegression`` which
exposes an analytic gradient).  ``iterative_feature_perturbation`` is
gradient-free and works with any model exposing ``predict`` and
``predict_proba`` (including decision trees), which mirrors a black-box
scenario where only model outputs are observable.
"""

from __future__ import annotations

import numpy as np


def _flipped(model, x: np.ndarray, y_true: int) -> bool:
    return int(model.predict(x.reshape(1, -1))[0]) != int(y_true)


def fgsm(
    model,
    x: np.ndarray,
    y_true: int,
    eps: float,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Fast Gradient Sign Method: one-step perturbation along the gradient."""
    gradient = model.gradient_wrt_input(np.asarray(x, dtype=float), int(y_true))
    perturbation = eps * np.sign(gradient)
    lo, hi = bounds
    return np.clip(np.asarray(x, dtype=float) + perturbation, lo, hi)


def pgd(
    model,
    x: np.ndarray,
    y_true: int,
    eps: float,
    alpha: float = 0.02,
    iters: int = 50,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Projected Gradient Descent within an L-infinity budget."""
    x = np.asarray(x, dtype=float)
    adv = x.copy()
    lo, hi = bounds
    for _ in range(iters):
        gradient = model.gradient_wrt_input(adv, int(y_true))
        adv = adv + alpha * np.sign(gradient)
        adv = np.clip(adv, lo, hi)
        adv = np.clip(adv, x - eps, x + eps)
    return adv


def iterative_feature_perturbation(
    model,
    x: np.ndarray,
    y_true: int,
    step: float = 0.02,
    max_iters: int = 300,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Gradient-free coordinate search.

    At each iteration every feature is nudged by +/- ``step`` and the move
    that most increases the model's confidence in the target (wrong) class
    is applied.  Stops early once the prediction flips.  Only model outputs
    are consulted, so it works for non-differentiable models like trees.
    """
    x = np.asarray(x, dtype=float)
    current = x.copy()
    lo, hi = bounds
    prev_best = -1.0
    for _ in range(max_iters):
        if _flipped(model, current, y_true):
            break
        best: np.ndarray | None = None
        best_score = -1.0
        for scale in (1.0, 2.0, 4.0):
            trial_step = step * scale
            for feature in range(len(x)):
                for sign in (+1.0, -1.0):
                    candidate = current.copy()
                    candidate[feature] = np.clip(
                        candidate[feature] + sign * trial_step, lo, hi
                    )
                    if _flipped(model, candidate, y_true):
                        return candidate
                    p = float(model.predict_proba(candidate.reshape(1, -1))[0])
                    score = p if int(y_true) == 0 else 1.0 - p
                    if score > best_score:
                        best_score = score
                        best = candidate
        if best is None or best_score <= prev_best + 1e-9:
            break
        prev_best = best_score
        current = best
    return current


def random_perturbation(
    model,
    x: np.ndarray,
    y_true: int,
    eps: float,
    trials: int = 400,
    seed: int = 0,
) -> np.ndarray:
    """Black-box baseline: random jitter inside the L-infinity ball."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    for _ in range(trials):
        noise = rng.uniform(-eps, eps, size=x.shape)
        candidate = np.clip(x + noise, 0.0, 1.0)
        if _flipped(model, candidate, y_true):
            return candidate
    return np.clip(x + rng.uniform(-eps, eps, size=x.shape), 0.0, 1.0)


def linf_distance(x0: np.ndarray, x1: np.ndarray) -> float:
    """L-infinity distance between original and perturbed inputs."""
    return float(np.max(np.abs(np.asarray(x0, dtype=float) - np.asarray(x1, dtype=float))))


def attack_success_rate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    attack,
    budget: float | None = None,
    **kwargs,
) -> tuple[float, float]:
    """Fraction of correctly-classified samples that the attack flips.

    Returns ``(success_rate, eligible_fraction)`` where the eligible
    fraction is the share of samples the model classified correctly in the
    first place (adversarial examples only make sense on those).
    """
    eligible = model.predict(X) == y
    if eligible.sum() == 0:
        return 0.0, 0.0
    flipped = 0
    for i, is_eligible in enumerate(eligible):
        if not is_eligible:
            continue
        adv = attack(model, X[i], int(y[i]), **kwargs)
        if budget is not None and linf_distance(X[i], adv) > budget + 1e-9:
            continue
        if _flipped(model, adv, int(y[i])):
            flipped += 1
    return flipped / eligible.sum(), eligible.sum() / len(y)
