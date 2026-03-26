import json
import tempfile
from pathlib import Path

from scripts.rebalance_numeric_triage import rebalance_train_rows


def make_row(task_id: str, split: str, vulnerable: bool, subtype: str, category: str):
    return {
        "task_id": task_id,
        "split": split,
        "vulnerable": vulnerable,
        "subtype": subtype,
        "metadata": {"source_category": category},
        "prompt": "p",
        "response": "r",
    }


def test_rebalance_drops_distractors_and_balances_positive_subtypes():
    rows = [
        make_row("p190_a", "train", True, "CWE-190", "positive"),
        make_row("p190_b", "train", True, "CWE-190", "positive"),
        make_row("p191_a", "train", True, "CWE-191", "positive"),
        make_row("hn_a", "train", False, "NONE", "hard_negative"),
        make_row("hn_b", "train", False, "NONE", "hard_negative"),
        make_row("hn_c", "train", False, "NONE", "hard_negative"),
        make_row("dist_a", "train", False, "NONE", "distractor_negative"),
        make_row("valid_a", "valid", True, "CWE-190", "positive"),
    ]

    out = rebalance_train_rows(
        rows,
        positive_to_hard_ratio=1.0,
        distractor_fraction=0.0,
        balance_positive_subtypes=True,
        seed=1,
    )

    train_rows = [row for row in out if row["split"] == "train"]
    valid_rows = [row for row in out if row["split"] == "valid"]

    assert len(valid_rows) == 1
    assert not any(row["metadata"]["source_category"] == "distractor_negative" for row in train_rows)

    positives = [row for row in train_rows if row["vulnerable"]]
    hard_negatives = [row for row in train_rows if row["metadata"]["source_category"] == "hard_negative"]
    subtype_counts = {}
    for row in positives:
        subtype_counts[row["subtype"]] = subtype_counts.get(row["subtype"], 0) + 1

    assert len(positives) >= len(hard_negatives)
    assert subtype_counts["CWE-190"] == subtype_counts["CWE-191"] == 2
    assert any("::" in row["task_id"] for row in positives)
