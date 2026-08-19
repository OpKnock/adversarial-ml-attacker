import json

import numpy as np
import pytest

from advml.attacks import iterative_feature_perturbation
from advml.detector import (
    AdversarialDetector,
    ConfidenceDetector,
    OutlierDetector,
)

from tests.test_models import fit_fixture


def make_crafted_set(model, X_test, y_test, n=40):
    crafted = []
    for i in range(n):
        sample = iterative_feature_perturbation(
            model, X_test[i], int(y_test[i]), step=0.05, max_iters=400
        )
        crafted.append(sample)
    return np.array(crafted)


def test_confidence_detector_flags_low_confidence():
    model, X_test, y_test, X_train, y_train = fit_fixture()
    crafted = make_crafted_set(model, X_test, y_test)
    detector = ConfidenceDetector(threshold=0.6)
    flags = detector.flag(model, crafted)
    assert flags.sum() > 0


def test_outlier_detector_fit_and_score():
    model, X_test, y_test, X_train, y_train = fit_fixture()
    benign, _ = make_benign_and_adversarial(model, X_test, y_test)
    outlier = OutlierDetector(k=3.0).fit(X_train, y_train)
    scores = outlier.score(model, benign)
    assert len(scores) == len(benign)
    assert scores.min() >= 0.0


def make_benign_and_adversarial(model, X_test, y_test, n=60):
    benign = X_test[:n]
    adversarial = make_crafted_set(model, X_test, y_test, n=n)
    return benign, adversarial


def test_combined_detector_flags_crafted_samples_more_than_benign():
    model, X_test, y_test, X_train, y_train = fit_fixture()
    benign, adversarial = make_benign_and_adversarial(model, X_test, y_test)
    detector = AdversarialDetector(confidence_threshold=0.7, outlier_k=3.0)
    detector.fit(X_train, y_train)
    stats = detector.report_stats(model, benign, adversarial)
    assert stats["true_positive_rate"] > stats["false_positive_rate"]


def test_detector_flagged_crafted_share_is_substantial():
    model, X_test, y_test, X_train, y_train = fit_fixture()
    benign, adversarial = make_benign_and_adversarial(model, X_test, y_test, n=40)
    detector = AdversarialDetector(confidence_threshold=0.7, outlier_k=3.0)
    detector.fit(X_train, y_train)
    stats = detector.report_stats(model, benign, adversarial)
    assert stats["true_positive_rate"] > 0.4


def test_report_stats_json_roundtrip():
    model, X_test, y_test, X_train, y_train = fit_fixture()
    benign, adversarial = make_benign_and_adversarial(model, X_test, y_test)
    detector = AdversarialDetector().fit(X_train, y_train)
    stats = detector.report_stats(model, benign, adversarial)
    payload = json.loads(json.dumps(stats))
    assert payload["adversarial_samples"] == len(adversarial)
    assert payload["benign_samples"] == len(benign)
