"""Evaluate an OpenAI model on the numeric triage benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openai import OpenAI

from rl_secdef.runner.numeric_triage import grade_numeric_triage_task

SYSTEM_PROMPT = "You are a careful C/C++ security triage assistant. Return strict JSON only."


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def compute_metrics(results: list[dict]) -> dict:
    binary = [1.0 if row["grading_details"]["binary_correct"] else 0.0 for row in results]
    exact = [1.0 if row["unit_pass_rate"] == 1.0 else 0.0 for row in results]
    return {
        "overall": {
            "num_tasks": len(results),
            "avg_reward": sum(r["reward"] for r in results) / len(results),
            "avg_unit_pass_rate": sum(r["unit_pass_rate"] for r in results) / len(results),
            "avg_process_score": sum(r["process_score"] for r in results) / len(results),
            "binary_accuracy": sum(binary) / len(binary),
            "exact_accuracy": sum(exact) / len(exact),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-completion-tokens", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set")

    tasks = load_tasks(Path(args.tasks_file))
    if args.limit > 0:
        tasks = tasks[: args.limit]

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results = []
    start = time.time()
    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] {task['task_id']}")
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]},
            ],
            temperature=0.0,
            max_completion_tokens=args.max_completion_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        graded = grade_numeric_triage_task(task, text)
        results.append(
            {
                "task_id": task["task_id"],
                "reward": graded.reward,
                "unit_pass_rate": graded.unit_pass_rate,
                "process_score": graded.process_score,
                "response": text,
                "grading_details": graded.details,
            }
        )
        print(f"  Reward: {graded.reward:.4f} | Unit: {graded.unit_pass_rate:.2f} | Process: {graded.process_score:.2f}")

    output = {
        "metadata": {
            "model": args.model,
            "num_tasks": len(tasks),
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - start,
        },
        "metrics": compute_metrics(results),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
