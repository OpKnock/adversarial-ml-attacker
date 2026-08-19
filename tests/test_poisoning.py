from advml.data import make_synthetic_dataset, train_test_split
from advml.poisoning import craft_label_poison, measure_poison_effect


def fixture():
    X, y = make_synthetic_dataset(n_samples=400, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y, seed=0)
    return X_train, y_train, X_test, y_test


def test_poison_samples_added_and_label_flipped():
    X_train, y_train, X_test, y_test = fixture()
    X_poisoned, y_poisoned = craft_label_poison(
        X_train, y_train, n_poison=20, seed=7
    )
    assert len(X_poisoned) == len(X_train) + 20
    assert len(y_poisoned) == len(y_train) + 20
    ones_before = sum(int(v) for v in y_train)
    ones_after = sum(int(v) for v in y_poisoned)
    zeros_before = len(y_train) - ones_before
    zeros_after = len(y_poisoned) - ones_after
    assert zeros_after == zeros_before + 20
    assert ones_after == ones_before
    assert X_poisoned.max() <= 1.0 and X_poisoned.min() >= 0.0


def test_poisoning_drop_is_measurable():
    X_train, y_train, X_test, y_test = fixture()
    result = measure_poison_effect(
        X_train, y_train, X_test, y_test, n_poison=60, seed=7
    )
    assert result["baseline_accuracy"] > 0.85
    assert result["accuracy_drop"] > 0.03
    assert result["poisoned_accuracy"] < result["baseline_accuracy"]


def test_poison_effect_is_deterministic():
    X_train, y_train, X_test, y_test = fixture()
    first = measure_poison_effect(X_train, y_train, X_test, y_test, n_poison=20)
    second = measure_poison_effect(X_train, y_train, X_test, y_test, n_poison=20)
    assert first == second


def test_poison_with_zero_samples_is_noop():
    X_train, y_train, X_test, y_test = fixture()
    result = measure_poison_effect(
        X_train, y_train, X_test, y_test, n_poison=0
    )
    assert result["poisoned_samples"] == 0
    assert result["accuracy_drop"] == 0.0
