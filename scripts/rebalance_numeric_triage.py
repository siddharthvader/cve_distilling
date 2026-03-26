#!/usr/bin/env python3
"""Rebalance numeric triage training rows without touching validation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def clone_with_suffix(row: dict[str, Any], suffix: str) -> dict[str, Any]:
    cloned = copy.deepcopy(row)
    cloned["task_id"] = f"{row['task_id']}::{suffix}"
    metadata = cloned.setdefault("metadata", {})
    metadata["rebalanced_from_task_id"] = row["task_id"]
    return cloned


def oversample_to_count(
    rows: list[dict[str, Any]],
    target_count: int,
    *,
    seed: int,
    suffix_prefix: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    output: list[dict[str, Any]] = []
    full_repeats, remainder = divmod(target_count, len(rows))

    for repeat_idx in range(full_repeats):
        for row_idx, row in enumerate(rows):
            if repeat_idx == 0:
                output.append(copy.deepcopy(row))
            else:
                output.append(clone_with_suffix(row, f"{suffix_prefix}rep{repeat_idx}_{row_idx}"))

    if remainder:
        rng = random.Random(seed)
        sampled_indices = rng.sample(range(len(rows)), remainder)
        for sample_idx, row_idx in enumerate(sampled_indices):
            row = rows[row_idx]
            output.append(clone_with_suffix(row, f"{suffix_prefix}extra{sample_idx}_{row_idx}"))

    return output


def rebalance_train_rows(
    rows: list[dict[str, Any]],
    *,
    positive_to_hard_ratio: float,
    distractor_fraction: float,
    balance_positive_subtypes: bool,
    seed: int,
) -> list[dict[str, Any]]:
    train_rows = [row for row in rows if row.get("split") == "train"]
    valid_rows = [row for row in rows if row.get("split") != "train"]

    positives = [row for row in train_rows if row.get("vulnerable") is True]
    hard_negatives = [
        row
        for row in train_rows
        if row.get("vulnerable") is False
        and row.get("metadata", {}).get("source_category") == "hard_negative"
    ]
    distractor_negatives = [
        row
        for row in train_rows
        if row.get("vulnerable") is False
        and row.get("metadata", {}).get("source_category") == "distractor_negative"
    ]

    if not positives or not hard_negatives:
        raise ValueError("Need both positive and hard-negative training rows to rebalance.")

    balanced_positives = positives
    if balance_positive_subtypes:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in positives:
            grouped[row.get("subtype", "UNKNOWN")].append(row)
        max_count = max(len(group_rows) for group_rows in grouped.values())
        balanced_positives = []
        for subtype, group_rows in sorted(grouped.items()):
            balanced_positives.extend(
                oversample_to_count(
                    group_rows,
                    max_count,
                    seed=seed + sum(ord(ch) for ch in subtype),
                    suffix_prefix=f"{subtype.lower()}_",
                )
            )

    target_positive_count = max(
        len(balanced_positives),
        int(math.ceil(len(hard_negatives) * positive_to_hard_ratio)),
    )
    positive_rows = oversample_to_count(
        balanced_positives,
        target_positive_count,
        seed=seed + 17,
        suffix_prefix="pos_",
    )

    distractor_target_count = int(math.ceil(len(hard_negatives) * distractor_fraction))
    distractor_rows = oversample_to_count(
        distractor_negatives,
        distractor_target_count,
        seed=seed + 29,
        suffix_prefix="dist_",
    )

    rebalanced_train = positive_rows + [copy.deepcopy(row) for row in hard_negatives] + distractor_rows
    random.Random(seed).shuffle(rebalanced_train)

    return rebalanced_train + [copy.deepcopy(row) for row in valid_rows]


def build_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row.get("split", "unknown") for row in rows)
    train_rows = [row for row in rows if row.get("split") == "train"]
    train_categories = Counter(
        row.get("metadata", {}).get("source_category", "unknown") for row in train_rows
    )
    train_subtypes = Counter(row.get("subtype", "UNKNOWN") for row in train_rows if row.get("vulnerable"))
    return {
        "rows": len(rows),
        "split_counts": dict(sorted(counts.items())),
        "train_category_counts": dict(sorted(train_categories.items())),
        "train_positive_subtypes": dict(sorted(train_subtypes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--positive-to-hard-ratio", type=float, default=1.0)
    parser.add_argument("--distractor-fraction", type=float, default=0.0)
    parser.add_argument("--no-balance-positive-subtypes", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.input)
    rebalanced_rows = rebalance_train_rows(
        rows,
        positive_to_hard_ratio=args.positive_to_hard_ratio,
        distractor_fraction=args.distractor_fraction,
        balance_positive_subtypes=not args.no_balance_positive_subtypes,
        seed=args.seed,
    )
    write_rows(args.output, rebalanced_rows)

    manifest = build_manifest(rebalanced_rows)
    manifest["source_input"] = str(args.input)
    manifest["positive_to_hard_ratio"] = args.positive_to_hard_ratio
    manifest["distractor_fraction"] = args.distractor_fraction
    manifest["balance_positive_subtypes"] = not args.no_balance_positive_subtypes
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"output": str(args.output), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
