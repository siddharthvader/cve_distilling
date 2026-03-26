"""Build a cleaned BigVul detect corpus for real-world calibration."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from rich.console import Console

from .bigvul import is_suitable_sample, normalize_cwe_id, normalize_language
from .eval_builder import clean_code_artifacts, create_blind_detect_prompt, extract_code_block
from .juliet_clean import format_detect_response, save_jsonl

console = Console()


def bigvul_identity_key(project: str, commit_id: str, cve_id: str, code: str) -> str:
    cleaned_code = clean_code_artifacts(code)
    digest = hashlib.sha1(cleaned_code.encode("utf-8")).hexdigest()[:16]
    return f"{project or '-'}::{commit_id or '-'}::{cve_id or '-'}::{digest}"


def load_eval_identity_keys(eval_file: Path) -> set[str]:
    keys: set[str] = set()
    with open(eval_file) as f:
        for line in f:
            if not line.strip():
                continue
            task = json.loads(line)
            metadata = task.get("metadata", {})
            code = metadata.get("bad_code") or extract_code_block(task.get("prompt", ""))
            keys.add(
                bigvul_identity_key(
                    metadata.get("project", ""),
                    metadata.get("commit_id", ""),
                    metadata.get("cve_id", ""),
                    code,
                )
            )
    return keys


def _select_diverse_candidates(
    candidates: list[dict[str, Any]],
    max_samples: int,
    max_per_cwe: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    by_cwe: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_cwe[row["cwe_id"]].append(row)

    rng = random.Random(seed)
    cwes = sorted(by_cwe)
    rng.shuffle(cwes)
    for cwe in cwes:
        rng.shuffle(by_cwe[cwe])
        if max_per_cwe > 0:
            by_cwe[cwe] = by_cwe[cwe][:max_per_cwe]

    selected: list[dict[str, Any]] = []
    limit = max_samples if max_samples > 0 else sum(len(rows) for rows in by_cwe.values())
    while len(selected) < limit and cwes:
        next_cwes: list[str] = []
        for cwe in cwes:
            bucket = by_cwe[cwe]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
            if bucket:
                next_cwes.append(cwe)
        cwes = next_cwes

    rng.shuffle(selected)
    return selected


def _split_projects(
    projects: list[str],
    valid_fraction: float,
    seed: int,
) -> set[str]:
    if not projects:
        return set()

    rng = random.Random(seed)
    shuffled = list(projects)
    rng.shuffle(shuffled)
    valid_count = int(round(len(shuffled) * valid_fraction))
    if len(shuffled) > 1 and valid_count == 0:
        valid_count = 1
    if len(shuffled) > 1:
        valid_count = min(valid_count, len(shuffled) - 1)
    return set(shuffled[:valid_count])


def build_bigvul_detect_rows_from_records(
    records: Iterable[dict[str, Any]],
    eval_identity_keys: set[str],
    valid_fraction: float = 0.15,
    seed: int = 42,
    max_samples: int = 0,
    max_per_cwe: int = 12,
    include_negative: bool = False,
    response_overrides: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not is_suitable_sample(record):
            continue

        cwe_id, cwe_name = normalize_cwe_id(record.get("CWE ID", ""))
        if not cwe_id.startswith("CWE-") or cwe_id == "CWE-Unknown":
            continue

        project = str(record.get("project", "") or "")
        commit_id = str(record.get("commit_id", "") or "")
        cve_id = str(record.get("CVE ID", "") or "")
        bad_code = clean_code_artifacts(record.get("func_before", ""))
        good_code = clean_code_artifacts(record.get("func_after", ""))
        if len(bad_code) < 80 or len(good_code) < 80:
            continue

        identity = bigvul_identity_key(project, commit_id, cve_id, bad_code)
        if identity in eval_identity_keys:
            continue

        language = normalize_language(record.get("lang", "c"))
        code_suffix = identity.rsplit("::", 1)[-1][:8]
        source_task_id = f"bigvul::{project or 'unknown'}::{commit_id or cve_id or index}::{code_suffix}"
        candidates.append(
            {
                "source_task_id": source_task_id,
                "identity": identity,
                "project": project or "unknown",
                "commit_id": commit_id,
                "cve_id": cve_id,
                "language": language,
                "cwe_id": cwe_id,
                "cwe_name": cwe_name,
                "bad_code": bad_code,
                "good_code": good_code,
            }
        )

    selected = _select_diverse_candidates(candidates, max_samples=max_samples, max_per_cwe=max_per_cwe, seed=seed)
    valid_projects = _split_projects(sorted({row["project"] for row in selected}), valid_fraction=valid_fraction, seed=seed)

    rows: list[dict[str, Any]] = []
    for row in selected:
        split = "valid" if row["project"] in valid_projects else "train"
        positive_response = response_overrides.get(row["source_task_id"]) if response_overrides else None
        if not positive_response:
            positive_response = format_detect_response(row["cwe_id"], row["cwe_name"], vulnerable=True)

        base_task_id = f"{row['source_task_id'].replace('::', '_')}"
        rows.append(
            {
                "task_id": f"{base_task_id}_positive",
                "source_task_id": row["source_task_id"],
                "type": "detect",
                "split": split,
                "prompt": create_blind_detect_prompt(row["language"], row["bad_code"]),
                "response": positive_response,
                "is_vulnerable": True,
                "cwe_id": row["cwe_id"],
                "cwe_name": row["cwe_name"],
                "template_family": row["project"],
                "family_id": row["project"],
                "metadata": {
                    "source": "bigvul",
                    "source_kind": "bad",
                    "source_task_id": row["source_task_id"],
                    "identity_key": row["identity"],
                    "project": row["project"],
                    "commit_id": row["commit_id"],
                    "cve_id": row["cve_id"],
                    "cwe_id": row["cwe_id"],
                    "cwe_name": row["cwe_name"],
                    "is_vulnerable": True,
                    "artifact_cleaned": True,
                    "prompt_variant": "blind_detect_v2",
                },
            }
        )
        if include_negative:
            rows.append(
                {
                    "task_id": f"{base_task_id}_negative",
                    "source_task_id": row["source_task_id"],
                    "type": "detect",
                    "split": split,
                    "prompt": create_blind_detect_prompt(row["language"], row["good_code"]),
                    "response": format_detect_response(row["cwe_id"], row["cwe_name"], vulnerable=False),
                    "is_vulnerable": False,
                    "cwe_id": None,
                    "cwe_name": None,
                    "template_family": row["project"],
                    "family_id": row["project"],
                    "metadata": {
                        "source": "bigvul",
                        "source_kind": "good",
                        "source_task_id": row["source_task_id"],
                        "identity_key": row["identity"],
                        "project": row["project"],
                        "commit_id": row["commit_id"],
                        "cve_id": row["cve_id"],
                        "cwe_id": row["cwe_id"],
                        "cwe_name": row["cwe_name"],
                        "is_vulnerable": False,
                        "artifact_cleaned": True,
                        "prompt_variant": "blind_detect_v2",
                    },
                }
            )

    random.Random(seed).shuffle(rows)
    return rows


def build_bigvul_detect_rows(
    eval_file: Path,
    split: str = "train",
    valid_fraction: float = 0.15,
    seed: int = 42,
    max_samples: int = 120,
    max_per_cwe: int = 12,
    include_negative: bool = False,
    response_overrides: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    eval_identity_keys = load_eval_identity_keys(eval_file)
    stream = load_dataset("bstee615/bigvul", split=split, streaming=True)
    return build_bigvul_detect_rows_from_records(
        stream,
        eval_identity_keys=eval_identity_keys,
        valid_fraction=valid_fraction,
        seed=seed,
        max_samples=max_samples,
        max_per_cwe=max_per_cwe,
        include_negative=include_negative,
        response_overrides=response_overrides,
    )


def audit_bigvul_rows(rows: Iterable[dict[str, Any]], eval_identity_keys: set[str]) -> dict[str, int]:
    prompt_leaks = 0
    overlap_leaks = 0
    split_leaks = 0
    project_splits: defaultdict[str, set[str]] = defaultdict(set)

    for row in rows:
        prompt = row.get("prompt", "")
        if "CVE-" in prompt or "Source project:" in prompt:
            prompt_leaks += 1
        metadata = row.get("metadata", {})
        if metadata.get("identity_key") in eval_identity_keys:
            overlap_leaks += 1
        project_splits[metadata.get("project", "unknown")].add(row.get("split", "train"))

    split_leaks = sum(1 for splits in project_splits.values() if len(splits) > 1)
    return {
        "prompt_leaks": prompt_leaks,
        "overlap_leaks": overlap_leaks,
        "split_leaks": split_leaks,
    }


def build_bigvul_detect_jsonl(
    output_path: Path,
    eval_file: Path,
    split: str = "train",
    valid_fraction: float = 0.15,
    seed: int = 42,
    max_samples: int = 120,
    max_per_cwe: int = 12,
    include_negative: bool = False,
    response_overrides: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    eval_identity_keys = load_eval_identity_keys(eval_file)
    rows = build_bigvul_detect_rows(
        eval_file=eval_file,
        split=split,
        valid_fraction=valid_fraction,
        seed=seed,
        max_samples=max_samples,
        max_per_cwe=max_per_cwe,
        include_negative=include_negative,
        response_overrides=response_overrides,
    )
    save_jsonl(output_path, rows)

    split_counts = Counter(row["split"] for row in rows)
    label_counts = Counter("vulnerable" if row["is_vulnerable"] else "clean" for row in rows)
    cwe_counts = Counter(row.get("metadata", {}).get("cwe_id", "CWE-Unknown") for row in rows if row.get("is_vulnerable"))
    project_counts = Counter(row.get("metadata", {}).get("project", "unknown") for row in rows if row.get("is_vulnerable"))
    audit = audit_bigvul_rows(rows, eval_identity_keys=eval_identity_keys)
    stats = {
        "output_path": str(output_path),
        "num_rows": len(rows),
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "num_cwes": len(cwe_counts),
        "num_projects": len(project_counts),
        "audit": audit,
    }
    console.print(
        f"[green]Wrote {len(rows)} cleaned BigVul detect rows to {output_path}[/]\n"
        f"[dim]Splits: {dict(split_counts)} | Labels: {dict(label_counts)} | Projects: {len(project_counts)} | CWEs: {len(cwe_counts)}[/]"
    )
    return stats
