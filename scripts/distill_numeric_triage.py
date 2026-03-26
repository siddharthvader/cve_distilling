"""Distill teacher-written JSON targets for the numeric triage task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openai import OpenAI


SYSTEM_PROMPT = """You are generating supervised fine-tuning targets for a narrow vulnerability triage model.

Return strict JSON only with exactly these keys:
{"vulnerable": true|false, "subtype": "CWE-190"|"CWE-191"|"NONE", "location": "...", "reason": "..."}

Rules:
- Use the hidden label as ground truth.
- This task only cares about CWE-190 and CWE-191. If the code has another issue, return vulnerable=false and subtype=NONE.
- Keep location short and code-grounded.
- Keep reason to one sentence.
- Do not wrap the JSON in Markdown.
"""


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["task_id"]] = row
    return rows


def build_teacher_request(row: dict[str, Any]) -> str:
    hidden_label = {
        "vulnerable": bool(row.get("vulnerable") or row.get("metadata", {}).get("gold_vulnerable")),
        "subtype": row.get("subtype") or row.get("metadata", {}).get("gold_subtype") or "NONE",
    }
    return (
        f"{row['prompt']}\n\n"
        f"HIDDEN LABEL:\n{json.dumps(hidden_label, indent=2)}\n\n"
        "Write the final JSON only."
    )


def distill_row(client: OpenAI, model: str, row: dict[str, Any]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_teacher_request(row)},
        ],
        temperature=0.0,
        max_completion_tokens=300,
    )
    distilled = (response.choices[0].message.content or "").strip()
    out = dict(row)
    out["response"] = distilled
    metadata = dict(out.get("metadata", {}))
    metadata["teacher_model"] = model
    out["metadata"] = metadata
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--splits", nargs="*", default=["train", "valid"])
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set")

    input_path = Path(args.input)
    rows = [row for row in load_rows(input_path) if row.get("split") in set(args.splits)]
    output_path = Path(args.output)
    existing = load_existing(output_path)
    replacements = dict(existing)
    pending = [row for row in rows if row["task_id"] not in replacements]

    print(f"Loaded {len(rows)} rows, {len(existing)} already distilled, {len(pending)} pending")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(distill_row, client, args.model, row): row["task_id"] for row in pending}
        for i, future in enumerate(as_completed(futures), start=1):
            task_id = futures[future]
            replacements[task_id] = future.result()
            print(f"[{i}/{len(pending)}] distilled {task_id}")

    final_rows = []
    original_rows = load_rows(input_path)
    for row in original_rows:
        final_rows.append(replacements.get(row["task_id"], row))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in final_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote distilled dataset to {output_path}")


if __name__ == "__main__":
    main()
