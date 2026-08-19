import numpy as np

from advml.attacks import fgsm, iterative_feature_perturbation, linf_distance, pgd
from advml.data import accuracy, make_synthetic_dataset, train_test_split
from advml.models import DecisionTree, LogisticRegression


def fit_fixture(seed=42, n_samples=400, n_features=6):
    X, y = make_synthetic_dataset(n_samples=n_samples, n_features=n_features, seed=seed)
    X_train, y_train, X_test, y_test = train_test_split(X, y, seed=0)
    model = LogisticRegression(seed=1).fit(X_train, y_train)
    return model, X_test, y_test, X_train, y_train


def test_synthetic_data_shape_and_range():
    X, y = make_synthetic_dataset(seed=1)
    assert X.shape == (600, 6)
    assert set(np.unique(y)) == {0, 1}
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_synthetic_data_deterministic():
    X1, y1 = make_synthetic_dataset(seed=7)
    X2, y2 = make_synthetic_dataset(seed=7)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_train_test_split_is_deterministic_and_disjoint():
    X, y = make_synthetic_dataset(seed=5)
    a = train_test_split(X, y, seed=0)
    b = train_test_split(X, y, seed=0)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    X_train, _, X_test, _ = a
    assert len(set(map(tuple, X_train)) & set(map(tuple, X_test))) == 0


def test_logistic_regression_achieves_high_accuracy():
    model, X_test, y_test, _, _ = fit_fixture()
    assert model.accuracy(X_test, y_test) > 0.85


def test_decision_tree_achieves_high_accuracy():
    X, y = make_synthetic_dataset(n_samples=400, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y, seed=0)
    tree = DecisionTree(seed=3).fit(X_train, y_train)
    assert tree.accuracy(X_test, y_test) > 0.8


def test_predict_proba_in_unit_interval():
    model, X_test, _, _, _ = fit_fixture()
    probs = model.predict_proba(X_test)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_models_agree_on_binary_outputs():
    model, X_test, _, _, _ = fit_fixture()
    assert set(np.unique(model.predict(X_test))) <= {0, 1}
