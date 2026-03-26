"""Tests for numeric triage grading."""

from rl_secdef.benchmark_numeric import compute_metrics
from rl_secdef.runner.numeric_triage import grade_numeric_triage_task


def test_grade_numeric_triage_positive_exact():
    task = {"vulnerable": True, "subtype": "CWE-190", "metadata": {"gold_vulnerable": True, "gold_subtype": "CWE-190"}}
    response = '{"vulnerable": true, "subtype": "CWE-190", "location": "len * count", "reason": "overflow"}'
    grade = grade_numeric_triage_task(task, response)
    assert grade.unit_pass_rate == 1.0
    assert grade.process_score == 1.0
    assert grade.reward == 1.0


def test_grade_numeric_triage_positive_partial():
    task = {"vulnerable": True, "subtype": "CWE-190", "metadata": {"gold_vulnerable": True, "gold_subtype": "CWE-190"}}
    response = '{"vulnerable": true, "subtype": "CWE-191", "location": "n--", "reason": "underflow"}'
    grade = grade_numeric_triage_task(task, response)
    assert grade.unit_pass_rate == 0.5
    assert grade.process_score == 1.0


def test_grade_numeric_triage_negative_exact():
    task = {"vulnerable": False, "subtype": "NONE", "metadata": {"gold_vulnerable": False, "gold_subtype": "NONE"}}
    response = '{"vulnerable": false, "subtype": "NONE", "location": "NONE", "reason": "not target"}'
    grade = grade_numeric_triage_task(task, response)
    assert grade.unit_pass_rate == 1.0
    assert grade.process_score == 1.0


def test_invalid_json_details_are_metric_safe():
    task = {"vulnerable": False, "subtype": "NONE", "metadata": {"gold_vulnerable": False, "gold_subtype": "NONE"}}
    grade = grade_numeric_triage_task(task, "not json")

    assert grade.details["binary_correct"] is False
    metrics = compute_metrics(
        [
            {
                "reward": grade.reward,
                "unit_pass_rate": grade.unit_pass_rate,
                "process_score": grade.process_score,
                "grading_details": grade.details,
            }
        ]
    )
    assert metrics["overall"]["binary_accuracy"] == 0.0
