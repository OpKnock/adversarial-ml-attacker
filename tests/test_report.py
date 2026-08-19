import json

import pytest

from advml.report import render_json, render_markdown, run_campaign, write_report


@pytest.fixture(scope="module")
def results():
    return run_campaign(seed=42, sample_limit=40)


def test_campaign_reports_baseline_accuracy(results):
    assert results["baseline_accuracy"] > 0.85


def test_campaign_has_all_attack_epsilons(results):
    assert list(results["attacks"].keys()) == ["0.05", "0.1", "0.2"]
    for rates in results["attacks"].values():
        assert set(rates) == {
            "fgsm",
            "pgd",
            "iterative_feature_perturbation",
            "random_perturbation",
        }


def test_campaign_success_rates_in_unit_interval(results):
    for rates in results["attacks"].values():
        for value in rates.values():
            assert 0.0 <= value <= 1.0


def test_markdown_report_has_core_sections(results):
    md = render_markdown(results)
    for section in [
        "Adversarial ML Robustness Report",
        "Executive Summary",
        "Attack Success Rates",
        "Adversarial Sample Detector",
        "Label Poisoning",
        "Adversarial Training Defense",
        "Methodology and Limitations",
    ]:
        assert section in md


def test_json_report_roundtrip(results):
    payload = json.loads(render_json(results))
    assert payload["seed"] == 42
    assert payload["attacks"]["0.1"]["pgd"] == results["attacks"]["0.1"]["pgd"]
    assert payload["detector"]["true_positive_rate"] >= 0.0
    assert payload["poisoning"]["accuracy_drop"] >= 0.0


def test_write_report_creates_files(results, tmp_path):
    written = write_report(results, tmp_path)
    assert len(written) == 2
    assert all(path.exists() for path in written)
    md = (tmp_path / "robustness_report.md").read_text(encoding="utf-8")
    data = json.loads(
        (tmp_path / "robustness_report.json").read_text(encoding="utf-8")
    )
    assert "Attack Success Rates" in md
    assert data["baseline_accuracy"] > 0.0


def test_campaign_is_deterministic():
    first = run_campaign(seed=7, sample_limit=30)
    second = run_campaign(seed=7, sample_limit=30)
    assert first == second
