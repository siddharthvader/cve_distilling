"""Build cleaner, blinded evaluation task files from existing datasets."""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

from rich.console import Console

console = Console()

CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", re.DOTALL)
S2SV_MARKER_RE = re.compile(r"<S2SV_[^>]+>")
CWE_PREFIX_RE = re.compile(r"^\s*CWE-[A-Za-z0-9_-]+\s+")
SUPPORTED_CWE_RE = re.compile(r"^CWE-\d+$")


def extract_code_block(text: str) -> str:
    """Extract the first fenced code block from a prompt."""
    match = CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else ""


def clean_code_artifacts(code: str) -> str:
    """Remove dataset-specific markers and leaked labels from code snippets."""
    if not code:
        return ""

    cleaned = S2SV_MARKER_RE.sub(" ", code)
    cleaned = CWE_PREFIX_RE.sub("", cleaned)
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def create_blind_detect_prompt(language: str, code: str) -> str:
    """Create a detection prompt without leaking hidden labels."""
    lang = language if language else "c"
    return f"""Analyze the following code for security vulnerabilities.

CODE:
```{lang}
{code}
```

INSTRUCTIONS:
1. Identify the most likely vulnerability type
2. Include a CWE ID if you know it
3. Identify the affected lines or function
4. Explain the root cause
5. Describe the likely security impact
6. Suggest a specific fix

Format your response as:
VULNERABILITY: [type and optional CWE]
LOCATION: [affected lines/functions]
ROOT_CAUSE: [why this is vulnerable]
IMPACT: [potential consequences]
FIX: [specific remediation steps]
"""


def create_blind_patch_prompt(language: str, code: str) -> str:
    """Create a patch prompt without leaking hidden labels."""
    lang = language if language else "c"
    return f"""Fix the security vulnerability in the following code.

VULNERABLE CODE:
```{lang}
{code}
```

INSTRUCTIONS:
1. Identify the vulnerability
2. Provide a corrected version that fixes the security issue
3. Preserve the original intent of the code as much as possible
4. Briefly explain your fix

Provide the fixed code in a code block, followed by your explanation.
"""


def _is_supported_task(task: dict, sources: set[str], task_types: set[str]) -> bool:
    metadata = task.get("metadata", {})
    source = metadata.get("source", "unknown")
    expected_cwe = task.get("grading", {}).get("expected_cwe", "")
    return (
        source in sources
        and task.get("type") in task_types
        and SUPPORTED_CWE_RE.match(expected_cwe) is not None
    )


def _sanitize_task(
    task: dict,
    min_code_length: int,
    min_patch_target_length: int,
) -> Optional[dict]:
    metadata = dict(task.get("metadata", {}))
    task_type = task["type"]

    raw_bad_code = metadata.get("bad_code") or extract_code_block(task["prompt"])
    bad_code = clean_code_artifacts(raw_bad_code)
    if len(bad_code) < min_code_length:
        return None

    prompt_builder = create_blind_detect_prompt
    good_code = metadata.get("good_code", "")

    if task_type == "patch":
        prompt_builder = create_blind_patch_prompt
        good_code = clean_code_artifacts(good_code)
        if len(good_code) < min_patch_target_length:
            return None

    cleaned = {
        "task_id": task["task_id"],
        "type": task["type"],
        "language": task.get("language", "c"),
        "prompt": prompt_builder(task.get("language", "c"), bad_code),
        "starter_files": task.get("starter_files", []),
        "tests": task.get("tests", []),
        "grading": dict(task.get("grading", {})),
        "rubric": dict(task.get("rubric", {})),
        "safety": dict(task.get("safety", {})),
        "metadata": metadata,
    }

    cleaned["metadata"]["bad_code"] = bad_code
    cleaned["metadata"]["good_code"] = good_code
    cleaned["metadata"]["prompt_variant"] = "blind_v1"
    cleaned["metadata"]["artifact_cleaned"] = True
    return cleaned


def _sample_diverse(tasks: list[dict], max_tasks: int, seed: int) -> list[dict]:
    """Sample tasks round-robin across CWE labels to keep a diverse eval set."""
    if max_tasks <= 0 or len(tasks) <= max_tasks:
        return tasks

    rng = random.Random(seed)
    by_cwe: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        expected_cwe = task.get("grading", {}).get("expected_cwe", "CWE-Unknown")
        by_cwe[expected_cwe].append(task)

    cwes = list(by_cwe.keys())
    rng.shuffle(cwes)
    for cwe in cwes:
        rng.shuffle(by_cwe[cwe])

    selected: list[dict] = []
    while len(selected) < max_tasks and cwes:
        next_cwes = []
        for cwe in cwes:
            bucket = by_cwe[cwe]
            if bucket and len(selected) < max_tasks:
                selected.append(bucket.pop())
            if bucket:
                next_cwes.append(cwe)
        cwes = next_cwes

    return selected


def _load_tasks(paths: Iterable[Path]) -> list[dict]:
    tasks: list[dict] = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
    return tasks


def build_clean_eval_set(
    output_path: Path,
    input_paths: Optional[Sequence[Path]] = None,
    sources: Sequence[str] = ("bigvul",),
    task_types: Sequence[str] = ("detect",),
    max_tasks: int = 0,
    seed: int = 42,
    min_code_length: int = 80,
    min_patch_target_length: int = 300,
) -> dict:
    """Create a cleaned and blinded evaluation file from existing JSONL tasks."""
    if input_paths is None:
        input_paths = (
            Path("data/bigvul_tasks.jsonl"),
            Path("data/cvefixes_tasks.jsonl"),
        )

    requested_sources = set(sources)
    requested_types = set(task_types)
    raw_tasks = _load_tasks(input_paths)

    cleaned_tasks: list[dict] = []
    for task in raw_tasks:
        if not _is_supported_task(task, requested_sources, requested_types):
            continue
        cleaned = _sanitize_task(
            task,
            min_code_length=min_code_length,
            min_patch_target_length=min_patch_target_length,
        )
        if cleaned is not None:
            cleaned_tasks.append(cleaned)

    cleaned_tasks = _sample_diverse(cleaned_tasks, max_tasks, seed)
    rng = random.Random(seed)
    rng.shuffle(cleaned_tasks)

    if not cleaned_tasks:
        raise ValueError("No clean evaluation tasks matched the requested filters")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for task in cleaned_tasks:
            f.write(json.dumps(task) + "\n")

    source_counts = Counter(t.get("metadata", {}).get("source", "unknown") for t in cleaned_tasks)
    type_counts = Counter(t["type"] for t in cleaned_tasks)
    cwe_counts = Counter(t.get("grading", {}).get("expected_cwe", "CWE-Unknown") for t in cleaned_tasks)

    stats = {
        "output_path": str(output_path),
        "num_tasks": len(cleaned_tasks),
        "sources": dict(source_counts),
        "types": dict(type_counts),
        "unique_cwes": len(cwe_counts),
        "task_ids": [task["task_id"] for task in cleaned_tasks],
    }

    console.print(f"[green]Wrote {len(cleaned_tasks)} cleaned eval tasks to {output_path}[/]")
    console.print(f"[dim]Sources: {dict(source_counts)} | Types: {dict(type_counts)} | Unique CWEs: {len(cwe_counts)}[/]")
    return stats
