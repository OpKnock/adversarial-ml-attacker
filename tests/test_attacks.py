import numpy as np

from advml.attacks import (
    attack_success_rate,
    fgsm,
    iterative_feature_perturbation,
    linf_distance,
    pgd,
    random_perturbation,
)
from advml.data import make_synthetic_dataset, train_test_split
from advml.defenses import adversarial_training
from advml.models import DecisionTree, LogisticRegression

from tests.test_models import fit_fixture


def pick_correctly_classified(model, X, y):
    for i in range(len(X)):
        if int(model.predict(X[i : i + 1])[0]) == int(y[i]):
            return i
    raise AssertionError("no correctly classified sample found")


def pick_lowest_confidence_correct(model, X, y):
    best_index = None
    best_confidence = float("inf")
    for i in range(len(X)):
        if int(model.predict(X[i : i + 1])[0]) != int(y[i]):
            continue
        p = float(model.predict_proba(X[i : i + 1])[0])
        confidence = max(p, 1.0 - p)
        if confidence < best_confidence:
            best_confidence = confidence
            best_index = i
    assert best_index is not None, "no correctly classified sample found"
    return best_index


def test_fgsm_flips_prediction_within_budget():
    model, X_test, y_test, _, _ = fit_fixture()
    i = pick_lowest_confidence_correct(model, X_test, y_test)
    adv = fgsm(model, X_test[i], int(y_test[i]), eps=0.2)
    assert model.predict(adv.reshape(1, -1))[0] != y_test[i]
    assert linf_distance(X_test[i], adv) <= 0.2 + 1e-9


def test_pgd_flips_prediction_within_budget():
    model, X_test, y_test, _, _ = fit_fixture()
    i = pick_lowest_confidence_correct(model, X_test, y_test)
    adv = pgd(model, X_test[i], int(y_test[i]), eps=0.15, iters=80)
    assert model.predict(adv.reshape(1, -1))[0] != y_test[i]
    assert linf_distance(X_test[i], adv) <= 0.15 + 1e-9


def test_iterative_feature_perturbation_flips_prediction():
    model, X_test, y_test, _, _ = fit_fixture()
    i = pick_lowest_confidence_correct(model, X_test, y_test)
    adv = iterative_feature_perturbation(
        model, X_test[i], int(y_test[i]), step=0.05, max_iters=400
    )
    assert model.predict(adv.reshape(1, -1))[0] != y_test[i]


def test_iterative_attack_works_on_decision_tree():
    X, y = make_synthetic_dataset(n_samples=400, seed=11)
    X_train, y_train, X_test, y_test = train_test_split(X, y, seed=0)
    tree = DecisionTree(seed=3).fit(X_train, y_train)
    i = pick_lowest_confidence_correct(tree, X_test, y_test)
    adv = iterative_feature_perturbation(
        tree, X_test[i], int(y_test[i]), step=0.05, max_iters=600
    )
    assert tree.predict(adv.reshape(1, -1))[0] != y_test[i]


def test_random_perturbation_respects_budget():
    model, X_test, y_test, _, _ = fit_fixture()
    i = 0
    adv = random_perturbation(model, X_test[i], int(y_test[i]), eps=0.1, seed=5)
    assert linf_distance(X_test[i], adv) <= 0.1 + 1e-9


def test_attack_success_rate_high_for_fgsm():
    model, X_test, y_test, _, _ = fit_fixture()
    rate, eligible = attack_success_rate(
        model, X_test, y_test, fgsm, budget=0.3, eps=0.3
    )
    assert rate > 0.6
    assert 0 < eligible <= 1.0


def test_attack_success_rate_zero_for_zero_budget():
    model, X_test, y_test, _, _ = fit_fixture()
    rate, _ = attack_success_rate(model, X_test, y_test, fgsm, budget=0.0, eps=0.0)
    assert rate == 0.0


def test_linf_distance_roundtrip_symmetry():
    a = np.array([0.1, 0.5, 0.9])
    b = np.array([0.2, 0.5, 0.85])
    assert abs(linf_distance(a, b) - linf_distance(b, a)) < 1e-12
    assert linf_distance(a, a) == 0.0


def test_adversarial_training_reduces_attack_success():
    X, y = make_synthetic_dataset(n_samples=400, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y, seed=0)
    results = adversarial_training(
        X_train, y_train, X_test, y_test, frac=0.3, seed=3
    )
    assert results["baseline_pgd_success"] > 0.5
    assert results["robust_pgd_success"] < results["baseline_pgd_success"]
