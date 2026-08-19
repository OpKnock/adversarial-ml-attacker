# Adversarial ML Attacker

An **educational** toolkit for testing the robustness of ML-based security
systems. It generates adversarial examples against small pure-NumPy models,
simulates training-data poisoning, detects crafted samples, and evaluates an
adversarial-training defense — all on synthetic in-memory data.

The purpose of this project is defensive: understanding how attackers fool
ML models is the first step to securing ML deployments (malware classifiers,
intrusion detectors, spam filters).

## Features

- **Gradient-based attacks** — FGSM and PGD (projected gradient descent in an
  L-infinity budget) using the analytic gradient of logistic regression.
- **Gradient-free attack** — iterative feature perturbation that only queries
  model outputs, so it also works against decision trees (black-box style).
- **Black-box baseline** — random perturbation inside the perturbation budget.
- **Label poisoning** — crafts label-flipped, centroid-nudged training
  samples and measures the accuracy drop after retraining.
- **Adversarial sample detection** — confidence-threshold detector plus
  per-class z-score outlier detector, usable separately or combined.
- **Adversarial training defense** — retrains on clean + adversarial data and
  quantifies robustness improvement.
- **Reports** — Markdown and JSON robustness reports (attack success rates by
  epsilon, detector rates, poisoning effect, defense delta).
- **Deterministic** — fixed seeds everywhere; offline and fully reproducible.

## Install

```bash
python -m pip install -e .[dev]
```

Requires NumPy (pure Python lists are supported by no other dependency).

## Usage

```bash
# Run the full evaluation campaign and write reports
python -m advml campaign --seed 42 --out reports
```

Programmatic use:

```python
from advml.data import make_synthetic_dataset, train_test_split
from advml.models import LogisticRegression, DecisionTree
from advml.attacks import fgsm, pgd, iterative_feature_perturbation

X, y = make_synthetic_dataset(seed=42)
X_train, y_train, X_test, y_test = train_test_split(X, y)
model = LogisticRegression(seed=1).fit(X_train, y_train)

adv = pgd(model, X_test[0], int(y_test[0]), eps=0.1)
print("original:", model.predict(X_test[0:1]), "adversarial:", model.predict(adv.reshape(1, -1)))
```

## Testing

```bash
python -m pytest
```

or `just test`.

## Legal and ethical notice

This project is for **security education and defense only**. Use it solely to
evaluate and harden models you own or are authorized to test. It operates on
synthetic data and performs no network or production activity. Do not use
these techniques against live ML services without explicit written
authorization.
