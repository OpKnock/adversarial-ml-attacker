"""Robustness evaluation campaign and report generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .attacks import (
    attack_success_rate,
    fgsm,
    iterative_feature_perturbation,
    pgd,
    random_perturbation,
)
from .data import make_synthetic_dataset, train_test_split
from .defenses import adversarial_training
from .detector import AdversarialDetector
from .models import LogisticRegression
from .poisoning import measure_poison_effect


def run_campaign(seed: int = 42, sample_limit: int = 120) -> dict:
    """Run the full evaluation: attacks, detector, poisoning, defense."""
    X, y = make_synthetic_dataset(seed=seed)
    X_train, y_train, X_test, y_test = train_test_split(X, y, seed=seed)
    model = LogisticRegression(seed=1).fit(X_train, y_train)

    epsilons = [0.05, 0.1, 0.2]
    attack_results = {}
    for eps in epsilons:
        attack_results[str(eps)] = {
            "fgsm": attack_success_rate(
                model, X_test, y_test, fgsm, budget=eps, eps=eps
            )[0],
            "pgd": attack_success_rate(
                model, X_test, y_test, pgd, budget=eps, eps=eps
            )[0],
            "iterative_feature_perturbation": attack_success_rate(
                model, X_test, y_test, iterative_feature_perturbation,
                budget=eps, step=min(0.05, eps / 2),
            )[0],
            "random_perturbation": attack_success_rate(
                model, X_test, y_test, random_perturbation,
                budget=eps, eps=eps,
            )[0],
        }

    adv_batch = np.array(
        [
            pgd(model, X_test[i], int(y_test[i]), eps=0.1)
            for i in range(min(sample_limit, len(X_test)))
        ]
    )
    benign_batch = X_test[: len(adv_batch)]
    detector = AdversarialDetector(confidence_threshold=0.7, outlier_k=3.0)
    detector.fit(X_train, y_train)
    detector_stats = detector.report_stats(model, benign_batch, adv_batch)

    poisoning = measure_poison_effect(
        X_train, y_train, X_test, y_test, n_poison=20, seed=7
    )
    defense = adversarial_training(
        X_train, y_train, X_test, y_test, frac=0.3, seed=3
    )

    return {
        "seed": seed,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "baseline_accuracy": model.accuracy(X_test, y_test),
        "attacks": attack_results,
        "detector": detector_stats,
        "poisoning": poisoning,
        "adversarial_training": defense,
    }


def render_json(results: dict) -> str:
    return json.dumps(results, indent=2)


def render_markdown(results: dict) -> str:
    lines = [
        "# Adversarial ML Robustness Report",
        "",
        "## Executive Summary",
        "",
        f"- Baseline test accuracy: {results['baseline_accuracy']:.3f}",
        f"- Training samples: {results['train_samples']}",
        f"- Test samples: {results['test_samples']}",
        "",
        "## Attack Success Rates (fraction of correctly-classified samples flipped)",
        "",
        "| Epsilon | FGSM | PGD | Iterative (gradient-free) | Random |",
        "|---------|------|-----|---------------------------|--------|",
    ]
    for eps, rates in results["attacks"].items():
        lines.append(
            f"| {eps} | {rates['fgsm']:.3f} | {rates['pgd']:.3f} | "
            f"{rates['iterative_feature_perturbation']:.3f} | "
            f"{rates['random_perturbation']:.3f} |"
        )
    lines.append("")
    lines.append("## Adversarial Sample Detector")
    lines.append("")
    detector = results["detector"]
    lines.append(f"- True positive rate: {detector['true_positive_rate']:.3f}")
    lines.append(f"- False positive rate: {detector['false_positive_rate']:.3f}")
    lines.append("")
    lines.append("## Label Poisoning")
    lines.append("")
    poisoning = results["poisoning"]
    lines.append(
        f"- Baseline accuracy: {poisoning['baseline_accuracy']:.3f} -> "
        f"poisoned accuracy: {poisoning['poisoned_accuracy']:.3f} "
        f"(drop {poisoning['accuracy_drop']:.3f})"
    )
    lines.append("")
    lines.append("## Adversarial Training Defense")
    lines.append("")
    defense = results["adversarial_training"]
    lines.append(
        f"- PGD success before defense: {defense['baseline_pgd_success']:.3f} -> "
        f"after: {defense['robust_pgd_success']:.3f}"
    )
    lines.append("")
    lines.append("## Methodology and Limitations")
    lines.append("")
    lines.append(
        "All models are trained on synthetic in-memory data. FGSM and PGD use "
        "the analytic gradient of logistic regression; the iterative "
        "perturbation attack is gradient-free and also works on decision "
        "trees. Results are deterministic for a fixed seed. This toolkit is "
        "educational: it exists to help secure ML-based security systems, "
        "not to attack deployed services."
    )
    return "\n".join(lines)


def write_report(results: dict, out_dir: str | Path) -> list[Path]:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    md_path = directory / "robustness_report.md"
    json_path = directory / "robustness_report.json"
    md_path.write_text(render_markdown(results), encoding="utf-8")
    json_path.write_text(render_json(results), encoding="utf-8")
    return [md_path, json_path]
