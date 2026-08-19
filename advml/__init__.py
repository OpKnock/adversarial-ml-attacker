"""An educational adversarial machine learning attack and detection toolkit."""

from .data import make_synthetic_dataset, train_test_split
from .models import DecisionTree, LogisticRegression
from .attacks import (
    attack_success_rate,
    fgsm,
    iterative_feature_perturbation,
    linf_distance,
    pgd,
    random_perturbation,
)
from .detector import AdversarialDetector, ConfidenceDetector, OutlierDetector
from .poisoning import craft_label_poison, measure_poison_effect
from .report import render_json, render_markdown, run_campaign, write_report

__all__ = [
    "make_synthetic_dataset",
    "train_test_split",
    "DecisionTree",
    "LogisticRegression",
    "attack_success_rate",
    "fgsm",
    "iterative_feature_perturbation",
    "linf_distance",
    "pgd",
    "random_perturbation",
    "AdversarialDetector",
    "ConfidenceDetector",
    "OutlierDetector",
    "craft_label_poison",
    "measure_poison_effect",
    "render_json",
    "render_markdown",
    "run_campaign",
    "write_report",
]

__version__ = "0.1.0"
