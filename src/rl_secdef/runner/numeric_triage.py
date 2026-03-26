"""Grading helpers for narrow numeric vulnerability triage."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class NumericGrade:
    reward: float
    unit_pass_rate: float
    process_score: float
    details: dict[str, Any]


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _normalize_subtype(value: Any) -> str:
    if value is None:
        return "NONE"
    text = str(value).strip().upper()
    if not text or text in {"NONE", "NULL", "N/A"}:
        return "NONE"
    match = re.search(r"CWE-\d+", text)
    return match.group(0) if match else text


def _normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def grade_numeric_triage_task(task: dict[str, Any], response: str) -> NumericGrade:
    gold_vulnerable = bool(task.get("vulnerable") or task.get("metadata", {}).get("gold_vulnerable"))
    gold_subtype = _normalize_subtype(task.get("subtype") or task.get("metadata", {}).get("gold_subtype"))

    parsed = _extract_json(response)
    if parsed is None:
        return NumericGrade(
            reward=0.0,
            unit_pass_rate=0.0,
            process_score=0.0,
            details={
                "error": "invalid_json",
                "response": response[:500],
                "gold_vulnerable": gold_vulnerable,
                "gold_subtype": gold_subtype,
                "pred_vulnerable": None,
                "pred_subtype": "NONE",
                "binary_correct": False,
                "valid_json": False,
            },
        )

    pred_vulnerable = _normalize_bool(parsed.get("vulnerable"))
    pred_subtype = _normalize_subtype(parsed.get("subtype"))
    valid_json = set(parsed.keys()) >= {"vulnerable", "subtype", "location", "reason"}
    process_score = 1.0 if valid_json else 0.5

    binary_correct = pred_vulnerable is not None and pred_vulnerable == gold_vulnerable
    if gold_vulnerable:
        subtype_correct = pred_subtype == gold_subtype
        if binary_correct and subtype_correct:
            unit = 1.0
        elif binary_correct:
            unit = 0.5
        else:
            unit = 0.0
    else:
        exact_negative = binary_correct and pred_subtype == "NONE"
        unit = 1.0 if exact_negative else (0.5 if binary_correct else 0.0)

    reward = 0.7 * unit + 0.3 * process_score
    return NumericGrade(
        reward=reward,
        unit_pass_rate=unit,
        process_score=process_score,
        details={
            "gold_vulnerable": gold_vulnerable,
            "gold_subtype": gold_subtype,
            "pred_vulnerable": pred_vulnerable,
            "pred_subtype": pred_subtype,
            "binary_correct": binary_correct,
            "valid_json": valid_json,
        },
    )
