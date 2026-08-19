"""Adversarial sample detectors.

A confidence-based detector flags inputs the model is unsure about; an
outlier detector flags inputs that sit far from the training distribution
in feature space (a hallmark of crafted perturbations).
"""

from __future__ import annotations

import numpy as np


def _confidence(model, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    return np.maximum(p, 1.0 - p)


class ConfidenceDetector:
    """Flags samples whose prediction confidence falls below a threshold."""

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def flag(self, model, X: np.ndarray) -> np.ndarray:
        return _confidence(model, X) < self.threshold

    def score(self, model, X: np.ndarray) -> np.ndarray:
        return _confidence(model, X)


class OutlierDetector:
    """Flags samples far from the centroid of their predicted class.

    The reference class is the model's own prediction, so a sample is only
    suspicious when it sits far outside the distribution of the class the
    model believes it belongs to.
    """

    def __init__(self, k: float = 3.0) -> None:
        self.k = k
        self.means: dict[int, np.ndarray] = {}
        self.stds: dict[int, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OutlierDetector":
        for label in np.unique(y):
            cluster = X[y == label]
            self.means[int(label)] = np.mean(cluster, axis=0)
            self.stds[int(label)] = np.std(cluster, axis=0) + 1e-6
        return self

    def score(self, model, X: np.ndarray) -> np.ndarray:
        predictions = model.predict(X).astype(int)
        scores = []
        for row, label in zip(X, predictions):
            z = np.max(np.abs(row - self.means[int(label)]) / self.stds[int(label)])
            scores.append(float(z))
        return np.array(scores)

    def flag(self, model, X: np.ndarray) -> np.ndarray:
        return self.score(model, X) > self.k


class AdversarialDetector:
    """Combines confidence and outlier detection."""

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        outlier_k: float = 3.0,
    ) -> None:
        self.confidence = ConfidenceDetector(confidence_threshold)
        self.outlier = OutlierDetector(outlier_k)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdversarialDetector":
        self.outlier.fit(X, y)
        return self

    def flag(self, model, X: np.ndarray) -> np.ndarray:
        return self.confidence.flag(model, X) | self.outlier.flag(model, X)

    def report_stats(
        self, model, benign: np.ndarray, adversarial: np.ndarray
    ) -> dict[str, float]:
        """False-positive and true-positive detection rates."""
        benign_flags = self.flag(model, benign)
        adv_flags = self.flag(model, adversarial)
        return {
            "false_positive_rate": float(np.mean(benign_flags))
            if len(benign) > 0
            else 0.0,
            "true_positive_rate": float(np.mean(adv_flags))
            if len(adversarial) > 0
            else 0.0,
            "adversarial_samples": int(len(adversarial)),
            "benign_samples": int(len(benign)),
        }
