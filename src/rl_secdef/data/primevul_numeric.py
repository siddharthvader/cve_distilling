"""Build a narrow PrimeVul-based benchmark for integer overflow/underflow triage."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from rich.console import Console

from .eval_builder import clean_code_artifacts, extract_code_block
from .juliet_clean import family_key_from_filename, load_jsonl, save_jsonl, strip_juliet_metadata

console = Console()

TARGET_CWES = ("CWE-190", "CWE-191")
NEAR_MISS_CWES = ("CWE-195", "CWE-197", "CWE-131", "CWE-680", "CWE-789")
DISTRACTOR_CWES = (
    "CWE-119",
    "CWE-125",
    "CWE-787",
    "CWE-416",
    "CWE-415",
    "CWE-476",
    "CWE-20",
    "CWE-134",
    "CWE-362",
    "CWE-89",
    "CWE-78",
    "CWE-264",
    "CWE-399",
)
CWE_NAMES = {
    "CWE-190": "Integer Overflow or Wraparound",
    "CWE-191": "Integer Underflow",
    "CWE-195": "Signed to Unsigned Conversion Error",
    "CWE-197": "Numeric Truncation Error",
    "CWE-131": "Incorrect Calculation of Buffer Size",
    "CWE-680": "Integer Overflow to Buffer Overflow",
    "CWE-789": "Memory Allocation with Excessive Size Value",
}
JSON_KEYS = ("vulnerable", "subtype", "location", "reason")


def create_numeric_triage_prompt(language: str, code: str) -> str:
    lang = language or "c"
    return f"""You are reviewing a C/C++ function for one specific vulnerability family.

Target vulnerabilities:
- CWE-190 Integer Overflow or Wraparound
- CWE-191 Integer Underflow

Task:
1. Decide whether the function contains a target vulnerability.
2. Ignore unrelated vulnerability types.
3. Return strict JSON only with exactly these keys:
   vulnerable, subtype, location, reason

CODE:
```{lang}
{code}
```

Return:
{{"vulnerable": true|false, "subtype": "CWE-190"|"CWE-191"|"NONE", "location": "<short span or NONE>", "reason": "<one sentence>"}}
"""


def _positive_reason(cwe_id: str) -> str:
    if cwe_id == "CWE-191":
        return "The function performs arithmetic that can underflow before the value is used in a size, bound, or index context."
    return "The function performs arithmetic that can overflow before the value is used in a size, bound, or index context."


def template_numeric_response(
    vulnerable: bool,
    subtype: str = "NONE",
    *,
    reason: Optional[str] = None,
    location: str = "NONE",
) -> str:
    if vulnerable:
        payload = {
            "vulnerable": True,
            "subtype": subtype,
            "location": location,
            "reason": reason or _positive_reason(subtype),
        }
    else:
        payload = {
            "vulnerable": False,
            "subtype": "NONE",
            "location": location,
            "reason": reason or "The function does not show a CWE-190 or CWE-191 pattern in the provided code.",
        }
    return json.dumps(payload, sort_keys=True)


def _sample_round_robin(rows: list[dict[str, Any]], max_samples: int, seed: int, group_key: str) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return list(rows)

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["metadata"].get(group_key, "unknown"))].append(row)

    rng = random.Random(seed)
    groups = sorted(grouped)
    rng.shuffle(groups)
    for key in groups:
        rng.shuffle(grouped[key])

    selected: list[dict[str, Any]] = []
    while len(selected) < max_samples and groups:
        next_groups: list[str] = []
        for key in groups:
            bucket = grouped[key]
            if bucket and len(selected) < max_samples:
                selected.append(bucket.pop())
            if bucket:
                next_groups.append(key)
        groups = next_groups
    return selected


def _choose_target_subtype(cwes: Iterable[str], allowed: tuple[str, ...]) -> Optional[str]:
    hits = [cwe for cwe in cwes if cwe in allowed]
    hits = sorted(set(hits))
    if len(hits) != 1:
        return None
    return hits[0]


def _base_record(
    *,
    source: str,
    source_split: str,
    source_category: str,
    task_id: str,
    prompt: str,
    split: str,
    vulnerable: bool,
    subtype: str,
    response: Optional[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "task_id": task_id,
        "split": split,
        "prompt": prompt,
        "vulnerable": vulnerable,
        "subtype": subtype,
        "metadata": {
            "source": source,
            "source_split": source_split,
            "source_category": source_category,
            "gold_vulnerable": vulnerable,
            "gold_subtype": subtype,
            **metadata,
        },
    }
    if response is not None:
        row["response"] = response
    return row


def build_primevul_numeric_rows(
    *,
    split: str,
    target_cwes: tuple[str, ...] = TARGET_CWES,
    distractor_cwes: tuple[str, ...] = DISTRACTOR_CWES,
    hard_negative_ratio: int = 2,
    distractor_ratio: int = 1,
    seed: int = 42,
    include_response: bool = True,
    source_rows: Optional[Iterable[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    if source_rows is None:
        from datasets import load_dataset

        source_rows = load_dataset("colin/PrimeVul", split=split, streaming=True)

    positives: list[dict[str, Any]] = []
    hard_negatives: list[dict[str, Any]] = []
    distractor_negatives: list[dict[str, Any]] = []

    for row in source_rows:
        code = clean_code_artifacts(row.get("func", ""))
        if len(code) < 40:
            continue

        cwes = tuple(row.get("cwe") or [])
        target_subtype = _choose_target_subtype(cwes, target_cwes)
        distractor_subtype = _choose_target_subtype(cwes, distractor_cwes)
        task_base = f"primevul_{split}_{row.get('idx')}"
        prompt = create_numeric_triage_prompt("c", code)
        base_metadata = {
            "project": row.get("project"),
            "commit_id": row.get("commit_id"),
            "cve": row.get("cve"),
            "cwe": list(cwes),
            "idx": row.get("idx"),
            "target_label": row.get("target"),
        }

        if row.get("target") == 1 and target_subtype:
            positives.append(
                _base_record(
                    source="primevul",
                    source_split=split,
                    source_category="positive",
                    task_id=f"{task_base}_positive",
                    prompt=prompt,
                    split="train" if split == "train" else ("valid" if split == "validation" else "test"),
                    vulnerable=True,
                    subtype=target_subtype,
                    response=template_numeric_response(True, target_subtype) if include_response else None,
                    metadata=base_metadata,
                )
            )
        elif row.get("target") == 0 and target_subtype:
            hard_negatives.append(
                _base_record(
                    source="primevul",
                    source_split=split,
                    source_category="hard_negative",
                    task_id=f"{task_base}_hard_negative",
                    prompt=prompt,
                    split="train" if split == "train" else ("valid" if split == "validation" else "test"),
                    vulnerable=False,
                    subtype="NONE",
                    response=template_numeric_response(False) if include_response else None,
                    metadata=base_metadata,
                )
            )
        elif row.get("target") == 1 and distractor_subtype:
            distractor_negatives.append(
                _base_record(
                    source="primevul",
                    source_split=split,
                    source_category="distractor_negative",
                    task_id=f"{task_base}_distractor_negative",
                    prompt=prompt,
                    split="train" if split == "train" else ("valid" if split == "validation" else "test"),
                    vulnerable=False,
                    subtype="NONE",
                    response=template_numeric_response(
                        False,
                        reason=f"The function may contain {distractor_subtype}, but not a target CWE-190 or CWE-191 issue.",
                    )
                    if include_response
                    else None,
                    metadata=base_metadata,
                )
            )

    rng = random.Random(seed)
    pos_count = len(positives)
    hard_negatives = _sample_round_robin(hard_negatives, pos_count * hard_negative_ratio, seed, "project")
    distractor_negatives = _sample_round_robin(distractor_negatives, pos_count * distractor_ratio, seed + 1, "project")

    rows = positives + hard_negatives + distractor_negatives
    rng.shuffle(rows)
    return rows


def build_juliet_numeric_rows(
    tasks_file: Path,
    *,
    target_cwes: tuple[str, ...] = TARGET_CWES,
    near_miss_cwes: tuple[str, ...] = NEAR_MISS_CWES,
    valid_fraction: float = 0.15,
    seed: int = 42,
) -> list[dict[str, Any]]:
    tasks = load_jsonl(tasks_file)
    family_to_tasks: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    relevant: list[dict[str, Any]] = []
    for task in tasks:
        metadata = task.get("metadata", {})
        if metadata.get("source") != "juliet" or task.get("type") != "detect":
            continue
        cwe_id = task.get("grading", {}).get("expected_cwe")
        if cwe_id not in target_cwes and cwe_id not in near_miss_cwes:
            continue
        relevant.append(task)
        family_to_tasks[family_key_from_filename(metadata.get("filename", task["task_id"]))].append(task)

    families = sorted(family_to_tasks)
    rng = random.Random(seed)
    rng.shuffle(families)
    valid_count = int(round(len(families) * valid_fraction))
    if len(families) > 1 and valid_count == 0:
        valid_count = 1
    if len(families) > 1:
        valid_count = min(valid_count, len(families) - 1)
    valid_families = set(families[:valid_count])

    rows: list[dict[str, Any]] = []
    for task in relevant:
        metadata = task.get("metadata", {})
        cwe_id = task.get("grading", {}).get("expected_cwe")
        family = family_key_from_filename(metadata.get("filename", task["task_id"]))
        split = "valid" if family in valid_families else "train"
        bad_code = strip_juliet_metadata(metadata.get("bad_code") or extract_code_block(task.get("prompt", "")))
        good_code = strip_juliet_metadata(metadata.get("good_code") or "")
        if len(bad_code) < 40 or len(good_code) < 40:
            continue

        source_base = task["task_id"]
        base_metadata = {
            "filename": metadata.get("filename"),
            "family_id": family,
            "source_task_id": source_base,
            "source_cwe": cwe_id,
        }

        if cwe_id in target_cwes:
            rows.append(
                _base_record(
                    source="juliet",
                    source_split="synthetic",
                    source_category="positive",
                    task_id=f"{source_base}_positive",
                    prompt=create_numeric_triage_prompt(task.get("language", "c"), bad_code),
                    split=split,
                    vulnerable=True,
                    subtype=cwe_id,
                    response=template_numeric_response(True, cwe_id),
                    metadata=base_metadata,
                )
            )
        else:
            rows.append(
                _base_record(
                    source="juliet",
                    source_split="synthetic",
                    source_category="near_miss_negative",
                    task_id=f"{source_base}_near_miss",
                    prompt=create_numeric_triage_prompt(task.get("language", "c"), bad_code),
                    split=split,
                    vulnerable=False,
                    subtype="NONE",
                    response=template_numeric_response(
                        False,
                        reason=f"The function may contain {cwe_id}, but not a target CWE-190 or CWE-191 issue.",
                    ),
                    metadata=base_metadata,
                )
            )

        rows.append(
            _base_record(
                source="juliet",
                source_split="synthetic",
                source_category="clean_negative",
                task_id=f"{source_base}_clean",
                prompt=create_numeric_triage_prompt(task.get("language", "c"), good_code),
                split=split,
                vulnerable=False,
                subtype="NONE",
                response=template_numeric_response(False),
                metadata=base_metadata,
            )
        )

    rng.shuffle(rows)
    return rows


def write_primevul_numeric_files(
    *,
    train_output: Path,
    eval_output: Path,
    seed: int = 42,
    target_cwes: tuple[str, ...] = TARGET_CWES,
    distractor_cwes: tuple[str, ...] = DISTRACTOR_CWES,
    hard_negative_ratio_train: int = 2,
    distractor_ratio_train: int = 1,
    hard_negative_ratio_eval: int = 3,
    distractor_ratio_eval: int = 3,
) -> dict[str, Any]:
    train_rows = build_primevul_numeric_rows(
        split="train",
        target_cwes=target_cwes,
        distractor_cwes=distractor_cwes,
        hard_negative_ratio=hard_negative_ratio_train,
        distractor_ratio=distractor_ratio_train,
        seed=seed,
        include_response=True,
    )
    valid_rows = build_primevul_numeric_rows(
        split="validation",
        target_cwes=target_cwes,
        distractor_cwes=distractor_cwes,
        hard_negative_ratio=hard_negative_ratio_train,
        distractor_ratio=distractor_ratio_train,
        seed=seed + 1,
        include_response=True,
    )
    for row in valid_rows:
        row["split"] = "valid"

    eval_rows = build_primevul_numeric_rows(
        split="test",
        target_cwes=target_cwes,
        distractor_cwes=distractor_cwes,
        hard_negative_ratio=hard_negative_ratio_eval,
        distractor_ratio=distractor_ratio_eval,
        seed=seed + 2,
        include_response=False,
    )
    for row in eval_rows:
        row["split"] = "test"

    all_train_rows = train_rows + valid_rows
    save_jsonl(train_output, all_train_rows)
    save_jsonl(eval_output, eval_rows)

    stats = {
        "train_rows": len(all_train_rows),
        "eval_rows": len(eval_rows),
        "train_split_counts": dict(Counter(row["split"] for row in all_train_rows)),
        "train_category_counts": dict(Counter(row["metadata"]["source_category"] for row in all_train_rows)),
        "eval_category_counts": dict(Counter(row["metadata"]["source_category"] for row in eval_rows)),
        "train_positive_subtypes": dict(Counter(row["subtype"] for row in all_train_rows if row["vulnerable"])),
        "eval_positive_subtypes": dict(Counter(row["subtype"] for row in eval_rows if row["vulnerable"])),
    }
    return stats


def write_juliet_numeric_file(
    *,
    tasks_file: Path,
    output_path: Path,
    seed: int = 42,
) -> dict[str, Any]:
    rows = build_juliet_numeric_rows(tasks_file, seed=seed)
    save_jsonl(output_path, rows)
    return {
        "rows": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "category_counts": dict(Counter(row["metadata"]["source_category"] for row in rows)),
        "positive_subtypes": dict(Counter(row["subtype"] for row in rows if row["vulnerable"])),
    }
