# Adversarial ML Attacker

An educational toolkit for testing the robustness of ML-based security systems. It generates adversarial examples against small pure-NumPy models, simulates training-data poisoning, detects crafted samples, and evaluates an adversarial-training defense — all on synthetic in-memory data.

The purpose of this project is defensive: understanding how attackers fool ML models is the first step to securing ML deployments (malware classifiers, intrusion detectors, spam filters).

## Overview

The Adversarial ML Attacker is an educational security tool designed to demonstrate adversarial machine learning techniques and defenses. This toolkit helps security researchers and ML practitioners understand how adversarial examples are generated, how to detect them, and how to build defenses — all in a controlled, educational environment using synthetic data.

**Important:** This tool is intended solely for educational and authorized ML security testing purposes. Only test ML models on datasets and models you own or have explicit written permission to test. Unauthorized testing of machine learning systems may violate applicable terms of service and academic integrity policies.

## Features

### Adversarial Example Generation

- **FGSM (Fast Gradient Sign Method)**: Fast gradient-based attack using the sign of the loss gradient
- **PGD (Projected Gradient Descent)**: Iterative gradient-based attack within L-infinity budget
- **Black-box baseline**: Random perturbation within the perturbation budget
- All attacks operate on small pure-NumPy models without requiring deep learning frameworks

### Training Data Poisoning

- **Label poisoning**: Craft label-flipped training samples
- **Centroid nudging**: Nudge training sample centroids to impact model accuracy
- **Accuracy drop measurement**: Quantify the impact of poisoning on model performance

### Adversarial Sample Detection

- **Confidence-threshold detector**: Identify adversarial samples based on prediction confidence
- **Per-class z-score outlier detector**: Class-specific outlier detector
- **Combined usage**: Use detectors separately or combined for enhanced detection

### Adversarial Training Defense

- **Retraining on clean + adversarial data**: Improve model robustness
- **Robustness improvement quantification**: Measure the defense's effectiveness
- **Defense delta reporting**: Report the improvement achieved

### Report Generation

- **Markdown reports**: Comprehensive robustness reports with attack success rates
- **JSON reports**: Structured data for further analysis and integration
- **Report categories**: Attack success rates by epsilon, detector rates, poisoning effect, defense delta

### Deterministic & Reproducible

- **Fixed seeds everywhere**: Ensures reproducible results
- **Offline operation**: No network dependency
- **Synthetic in-memory data**: All testing uses generated synthetic data
- **Fully offline**: No production or live service interaction

## Installation

### Requirements

- **Python 3.9+**
- **NumPy**: Supported via pure Python lists if NumPy not available
- **Optional**: `just` command runner (for command execution)

### Install with pip

```bash
python -m pip install -e .[dev]
```

### Just Command Runner

```bash
# Using curl (recommended)
curl -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin

# Or via package manager
# Debian/Ubuntu: apt install just
# Fedora: dnf install just
# macOS: brew install just
```

### Verify Installation

```bash
python -m pytest --version
advml --help
```

## Usage

### Run Full Evaluation Campaign

```bash
python -m advml campaign --seed 42 --out reports
```

### Programmatic Use

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

### Test Suite

```bash
python -m pytest
```

or `just test`.

## Legal and Ethical Notes

### Authorized Testing Only

This tool is designed for authorized ML security education and testing. Key principles:

- **Only test ML models on datasets and models you own or have explicit permission to evaluate**
- **Do not use these techniques against live ML services** without explicit written authorization
- **Report any discovered vulnerabilities** to the appropriate model owners
- **Never test models** you do not have explicit authorization to evaluate

### Legal Compliance

- Unauthorized ML security testing may violate computer fraud and abuse laws
- Academic integrity policies may apply in educational contexts
- Always obtain explicit written permission before testing any ML system

### Educational Value

Understanding adversarial ML helps security teams:

- Defend ML models against attack techniques
- Design robust model training procedures
- Evaluate model reliability in security-critical applications
- Build awareness of ML-related threat vectors

### Responsible Use

- This project operates on synthetic data and performs no network or production activity
- Do not use these techniques against live ML services without explicit written authorization
- Report any discovered vulnerabilities following responsible disclosure practices

## License

MIT - This project is free to use, modify, and distribute for educational purposes. See the LICENSE file for full terms and conditions.