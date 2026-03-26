"""Build a cleaned, blinded Juliet detect-only training corpus."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from rich.console import Console

from .eval_builder import create_blind_detect_prompt, extract_code_block

console = Console()

JULIET_TASK_SOURCE = "juliet"
FAMILY_SUFFIX_RE = re.compile(r"_(\d+[a-zA-Z]?)$")
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)
S2SV_RE = re.compile(r"<S2SV_[^>]+>")
CWE_ID_RE = re.compile(r"\bCWE-\d+\b", re.IGNORECASE)
CWE_IDENT_RE = re.compile(r"\bCWE\d+_[A-Za-z0-9_]+\b")
LEAK_MARKERS = (
    "TEMPLATE GENERATED TESTCASE FILE",
    "Filename:",
    "Label Definition File:",
    "Template File:",
    "@description",
    "BadSource:",
    "GoodSource:",
    "BadSink:",
    "GoodSink:",
    "Sink:",
)
LEAK_IDENTIFIER_PATTERNS = (
    (re.compile(r"\bgoodG2B\d*\b", re.IGNORECASE), "safe_variant"),
    (re.compile(r"\bgoodB2G\d*\b", re.IGNORECASE), "safe_variant"),
    (re.compile(r"\bgoodSink\d*\b", re.IGNORECASE), "safe_sink"),
    (re.compile(r"\bbadSink\d*\b", re.IGNORECASE), "unsafe_sink"),
    (re.compile(r"\bgoodSource\d*\b", re.IGNORECASE), "safe_source"),
    (re.compile(r"\bbadSource\d*\b", re.IGNORECASE), "unsafe_source"),
    (re.compile(r"\bgood\d+\b", re.IGNORECASE), "safe_variant"),
    (re.compile(r"\bgood\b", re.IGNORECASE), "safe_variant"),
    (re.compile(r"\bbad\b", re.IGNORECASE), "unsafe_variant"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def family_key_from_filename(filename: str) -> str:
    path = Path(filename)
    stem = FAMILY_SUFFIX_RE.sub("", path.stem)
    family_dir = next((part for part in path.parts if part.startswith("CWE")), path.parent.name or "unknown")
    return f"{family_dir}/{stem}"


def _clean_comment_block(block: re.Match[str]) -> str:
    text = block.group(0)
    if any(marker in text for marker in LEAK_MARKERS) or CWE_IDENT_RE.search(text) or CWE_ID_RE.search(text):
        return ""
    return text


def strip_juliet_metadata(code: str) -> str:
    """Remove Juliet template markers and obvious label leakage from code."""
    if not code:
        return ""

    cleaned = code.replace("\r\n", "\n")
    cleaned = S2SV_RE.sub(" ", cleaned)
    cleaned = BLOCK_COMMENT_RE.sub(_clean_comment_block, cleaned)
    cleaned = LINE_COMMENT_RE.sub("", cleaned)
    cleaned = CWE_IDENT_RE.sub("juliet_case", cleaned)
    cleaned = CWE_ID_RE.sub("", cleaned)

    for pattern, repl in LEAK_IDENTIFIER_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)

    lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if any(marker in stripped for marker in LEAK_MARKERS):
            continue
        line = re.sub(r"[ \t]+", " ", line).rstrip()
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def format_detect_response(cwe_id: str, cwe_name: str, vulnerable: bool) -> str:
    if vulnerable:
        label = cwe_id if not cwe_name else f"{cwe_id} - {cwe_name}"
        return (
            f"VULNERABILITY: {label}\n"
            "LOCATION: The unsafe operation in the shown function.\n"
            f"ROOT_CAUSE: The code follows a vulnerable {cwe_name.lower()} pattern without enough validation or safety checks.\n"
            f"IMPACT: {_impact_hint(cwe_id)}\n"
            f"FIX: {_fix_hint(cwe_id)}"
        )

    return (
        "VULNERABILITY: NONE\n"
        "LOCATION: NONE\n"
        "ROOT_CAUSE: The cleaned variant does not show the vulnerable pattern from this Juliet family.\n"
        "IMPACT: NONE\n"
        "FIX: NONE"
    )


def _impact_hint(cwe_id: str) -> str:
    hints = {
        "CWE-20": "Unexpected input can drive the program into unsafe behavior.",
        "CWE-22": "An attacker may access or overwrite files outside the intended path.",
        "CWE-78": "An attacker may execute unintended shell commands.",
        "CWE-79": "An attacker may inject script content into a rendered page or output.",
        "CWE-89": "An attacker may manipulate database queries or data.",
        "CWE-119": "The program may read or write outside valid memory.",
        "CWE-120": "An overflow may corrupt adjacent memory or crash the process.",
        "CWE-121": "A stack overwrite may crash the process or redirect control flow.",
        "CWE-122": "A heap overwrite may corrupt adjacent objects or allocator state.",
        "CWE-125": "An out-of-bounds read may leak memory or crash the process.",
        "CWE-126": "A buffer over-read may disclose adjacent memory or crash the program.",
        "CWE-134": "An attacker may leak memory or write through format specifiers.",
        "CWE-190": "Overflowed arithmetic may produce unsafe sizes or indices.",
        "CWE-191": "Underflow may produce invalid sizes, indices, or pointer math.",
        "CWE-200": "Sensitive information may be exposed to an unauthorized caller.",
        "CWE-284": "Unauthorized users may perform a restricted action.",
        "CWE-362": "Concurrent execution may corrupt shared state or object lifetime.",
        "CWE-400": "An attacker may exhaust resources and deny service.",
        "CWE-401": "Resources may be leaked until the program slows down or fails.",
        "CWE-415": "Double free may corrupt allocator metadata or crash the process.",
        "CWE-416": "Use-after-free may corrupt memory or execute attacker-controlled data.",
        "CWE-457": "Uninitialized state may cause undefined behavior or leak stale data.",
        "CWE-476": "A null dereference may crash the process or service.",
        "CWE-772": "Missing resource release may leak handles or memory over time.",
        "CWE-787": "Out-of-bounds writes may corrupt memory and enable code execution.",
        "CWE-835": "A non-terminating loop may cause denial of service.",
    }
    return hints.get(cwe_id, "The vulnerable pattern may let an attacker crash the process or corrupt program state.")


def _fix_hint(cwe_id: str) -> str:
    hints = {
        "CWE-20": "Validate and normalize input before using it.",
        "CWE-22": "Canonicalize the path and block traversal outside the intended directory.",
        "CWE-78": "Avoid shelling out with untrusted input or strictly sanitize arguments.",
        "CWE-79": "Encode or sanitize untrusted data before rendering it.",
        "CWE-89": "Use parameterized queries and strict input validation.",
        "CWE-119": "Check every read and write against the valid buffer size.",
        "CWE-120": "Use bounded copy APIs and preserve null termination.",
        "CWE-121": "Keep writes within the fixed stack buffer size.",
        "CWE-122": "Keep writes within the allocated heap buffer bounds.",
        "CWE-125": "Check read bounds before accessing the buffer.",
        "CWE-126": "Verify the read length stays inside the source buffer.",
        "CWE-134": "Pass a fixed format string and treat input as data.",
        "CWE-190": "Check arithmetic for overflow before using the result.",
        "CWE-191": "Check arithmetic for underflow before using the result.",
        "CWE-200": "Do not expose sensitive values in externally visible output.",
        "CWE-284": "Enforce authorization before the sensitive action.",
        "CWE-362": "Protect shared state with the correct synchronization primitive.",
        "CWE-400": "Cap resource growth and release resources promptly.",
        "CWE-401": "Release the resource on every exit path.",
        "CWE-415": "Track ownership and free each allocation exactly once.",
        "CWE-416": "Stop using freed objects and invalidate stale references.",
        "CWE-457": "Initialize state before first use.",
        "CWE-476": "Check for null before dereferencing the pointer.",
        "CWE-772": "Release the resource when it is no longer needed.",
        "CWE-787": "Keep writes inside the destination buffer bounds.",
        "CWE-835": "Ensure the loop has a reachable exit condition.",
    }
    return hints.get(cwe_id, "Apply the safe implementation pattern from the corresponding clean variant.")


def build_juliet_detect_rows(
    tasks_file: Path,
    valid_fraction: float = 0.1,
    seed: int = 42,
    max_families: int = 0,
    response_overrides: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Create cleaned prompt/response training rows from normalized Juliet tasks."""
    tasks = load_jsonl(tasks_file)
    juliet_detect_tasks = []
    families: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        metadata = task.get("metadata", {})
        if metadata.get("source") != JULIET_TASK_SOURCE or task.get("type") != "detect":
            continue
        filename = metadata.get("filename", task["task_id"])
        family = family_key_from_filename(filename)
        families[family].append(task)
        juliet_detect_tasks.append(task)

    family_ids = sorted(families)
    if not family_ids:
        raise ValueError(f"No Juliet detect tasks found in {tasks_file}")

    rng = random.Random(seed)
    rng.shuffle(family_ids)
    if max_families > 0:
        chosen = set(family_ids[:max_families])
        juliet_detect_tasks = [
            task for task in juliet_detect_tasks if family_key_from_filename(task.get("metadata", {}).get("filename", task["task_id"])) in chosen
        ]
        family_ids = [family for family in family_ids if family in chosen]

    valid_family_count = int(round(len(family_ids) * valid_fraction))
    if len(family_ids) > 1 and valid_family_count == 0:
        valid_family_count = 1
    if len(family_ids) > 1:
        valid_family_count = min(valid_family_count, len(family_ids) - 1)
    valid_families = set(family_ids[:valid_family_count])

    rows: list[dict[str, Any]] = []
    for task in juliet_detect_tasks:
        metadata = task.get("metadata", {})
        filename = metadata.get("filename", task["task_id"])
        family = family_key_from_filename(filename)
        split = "valid" if family in valid_families else "train"
        source_task_id = task["task_id"]
        cwe_id = task.get("grading", {}).get("expected_cwe", metadata.get("cwe_id", "CWE-Unknown"))
        cwe_name = metadata.get("cwe_name", "")
        bad_code = strip_juliet_metadata(metadata.get("bad_code") or extract_code_block(task.get("prompt", "")))
        good_code = strip_juliet_metadata(metadata.get("good_code", ""))
        if len(bad_code) < 40 or len(good_code) < 40:
            continue

        positive_response = response_overrides.get(source_task_id) if response_overrides else None
        if not positive_response:
            positive_response = format_detect_response(cwe_id, cwe_name, vulnerable=True)

        rows.append(
            {
                "task_id": f"{source_task_id}_positive",
                "source_task_id": source_task_id,
                "type": "detect",
                "split": split,
                "prompt": create_blind_detect_prompt(task.get("language", "c"), bad_code),
                "response": positive_response,
                "is_vulnerable": True,
                "cwe_id": cwe_id,
                "cwe_name": cwe_name,
                "template_family": family,
                "family_id": family,
                "metadata": {
                    "source": JULIET_TASK_SOURCE,
                    "source_kind": "bad",
                    "source_task_id": source_task_id,
                    "filename": filename,
                    "family_id": family,
                    "cwe_id": cwe_id,
                    "cwe_name": cwe_name,
                    "is_vulnerable": True,
                    "artifact_cleaned": True,
                    "prompt_variant": "blind_detect_v2",
                },
            }
        )
        rows.append(
            {
                "task_id": f"{source_task_id}_negative",
                "source_task_id": source_task_id,
                "type": "detect",
                "split": split,
                "prompt": create_blind_detect_prompt(task.get("language", "c"), good_code),
                "response": format_detect_response(cwe_id, cwe_name, vulnerable=False),
                "is_vulnerable": False,
                "cwe_id": None,
                "cwe_name": None,
                "template_family": family,
                "family_id": family,
                "metadata": {
                    "source": JULIET_TASK_SOURCE,
                    "source_kind": "good",
                    "source_task_id": source_task_id,
                    "filename": filename,
                    "family_id": family,
                    "cwe_id": cwe_id,
                    "cwe_name": cwe_name,
                    "is_vulnerable": False,
                    "artifact_cleaned": True,
                    "prompt_variant": "blind_detect_v2",
                },
            }
        )

    rng.shuffle(rows)
    return rows


def build_juliet_detect_jsonl(
    tasks_file: Path,
    output_path: Path,
    valid_fraction: float = 0.1,
    seed: int = 42,
    max_families: int = 0,
    response_overrides: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    rows = build_juliet_detect_rows(
        tasks_file=tasks_file,
        valid_fraction=valid_fraction,
        seed=seed,
        max_families=max_families,
        response_overrides=response_overrides,
    )
    save_jsonl(output_path, rows)

    split_counts = Counter(row["split"] for row in rows)
    family_counts = Counter(row["family_id"] for row in rows)
    cwe_counts = Counter(row.get("metadata", {}).get("cwe_id", "CWE-Unknown") for row in rows)
    label_counts = Counter("vulnerable" if row["is_vulnerable"] else "clean" for row in rows)
    audit = audit_juliet_rows(rows)
    stats = {
        "output_path": str(output_path),
        "num_rows": len(rows),
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "num_families": len(family_counts),
        "num_cwes": len(cwe_counts),
        "audit": audit,
    }
    console.print(
        f"[green]Wrote {len(rows)} Juliet detect rows to {output_path}[/]\n"
        f"[dim]Splits: {dict(split_counts)} | Labels: {dict(label_counts)} | Families: {len(family_counts)} | CWEs: {len(cwe_counts)}[/]"
    )
    return stats


def load_response_overrides(path: Optional[Path]) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    overrides: dict[str, str] = {}
    for row in load_jsonl(path):
        key = row.get("source_task_id") or row.get("task_id")
        response = row.get("response")
        if key and response:
            overrides[str(key)] = response
    return overrides


def normalize_training_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized.setdefault("split", "train")
    normalized.setdefault("metadata", {})
    if "messages" in normalized and normalized["messages"] is not None:
        normalized["messages"] = list(normalized["messages"])
    return normalized


def load_training_rows(path: Path, split: Optional[str] = None) -> list[dict[str, Any]]:
    rows = [normalize_training_row(row) for row in load_jsonl(path)]
    if split is not None:
        rows = [row for row in rows if row.get("split", "train") == split]
    return rows


def row_to_text(row: dict[str, Any], tokenizer: Any, system_prompt: str) -> str:
    if row.get("text"):
        return row["text"]

    messages = row.get("messages")
    if messages:
        chat = [dict(message) for message in messages]
        if not chat or chat[0].get("role") != "system":
            chat.insert(0, {"role": "system", "content": system_prompt})
        if chat[-1].get("role") != "assistant":
            chat.append({"role": "assistant", "content": row["response"]})
    else:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)

    return (
        "### Instruction:\n"
        f"{row['prompt']}\n\n"
        "### Response:\n"
        f"{row['response']}"
    )


def audit_juliet_rows(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    prompt_leaks = 0
    code_leaks = 0
    family_to_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        prompt = row.get("prompt", "")
        if "Filename:" in prompt or "Label Definition File:" in prompt or "Template File:" in prompt:
            prompt_leaks += 1
        if "S2SV" in prompt or CWE_ID_RE.search(prompt) or CWE_IDENT_RE.search(prompt):
            prompt_leaks += 1

        family = row.get("family_id") or row.get("template_family") or "unknown"
        split = row.get("split", "train")
        family_to_splits[family].add(split)

    for row in rows:
        for field in ("prompt",):
            text = row.get(field, "")
            if any(marker in text for marker in LEAK_MARKERS) or "S2SV" in text:
                code_leaks += 1

    split_leaks = sum(1 for splits in family_to_splits.values() if len(splits) > 1)
    return {
        "prompt_leaks": prompt_leaks,
        "code_leaks": code_leaks,
        "split_leaks": split_leaks,
    }
