"""Benchmark a model on the PrimeVul numeric triage task."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rl_secdef.runner.numeric_triage import grade_numeric_triage_task


def load_tasks(tasks_file: Path) -> List[dict]:
    tasks = []
    with open(tasks_file) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def select_tasks(tasks: List[dict], num_tasks: int, seed: int = 42) -> List[dict]:
    if num_tasks <= 0 or num_tasks >= len(tasks):
        return list(tasks)
    rng = random.Random(seed)
    return rng.sample(tasks, num_tasks)


def compute_metrics(results: List[dict]) -> Dict[str, Any]:
    if not results:
        return {}

    def detail(row: dict, key: str, default: Any) -> Any:
        return row.get("grading_details", {}).get(key, default)

    binary_correct = [1.0 if detail(r, "binary_correct", False) else 0.0 for r in results]
    exact = [r["unit_pass_rate"] == 1.0 for r in results]
    positive_rows = [r for r in results if detail(r, "gold_vulnerable", False)]
    negative_rows = [r for r in results if not detail(r, "gold_vulnerable", False)]
    metrics = {
        "overall": {
            "num_tasks": len(results),
            "avg_reward": sum(r["reward"] for r in results) / len(results),
            "avg_unit_pass_rate": sum(r["unit_pass_rate"] for r in results) / len(results),
            "avg_process_score": sum(r["process_score"] for r in results) / len(results),
            "binary_accuracy": sum(binary_correct) / len(binary_correct),
            "exact_accuracy": sum(1.0 for flag in exact if flag) / len(exact),
        },
        "positives": {
            "num_tasks": len(positive_rows),
            "avg_unit_pass_rate": sum(r["unit_pass_rate"] for r in positive_rows) / max(len(positive_rows), 1),
        },
        "negatives": {
            "num_tasks": len(negative_rows),
            "avg_unit_pass_rate": sum(r["unit_pass_rate"] for r in negative_rows) / max(len(negative_rows), 1),
        },
        "subtypes": dict(Counter(detail(r, "pred_subtype", "NONE") for r in results)),
    }
    return metrics


def run_numeric_benchmark(
    base_model: str,
    adapter_path: Optional[str],
    tasks_file: Path,
    output_file: Path,
    num_tasks: int = 0,
    seed: int = 42,
    max_new_tokens: int = 256,
) -> dict:
    from rl_secdef.benchmark_hf import load_model_and_tokenizer, query_hf_model, resolve_model_id

    model, tokenizer = load_model_and_tokenizer(base_model, adapter_path=adapter_path, load_in_4bit=True)
    all_tasks = load_tasks(tasks_file)
    tasks = select_tasks(all_tasks, num_tasks, seed=seed)

    results = []
    start = time.time()
    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] {task['task_id']}")
        response = query_hf_model(model, tokenizer, task["prompt"], max_new_tokens=max_new_tokens, temperature=0.0)
        graded = grade_numeric_triage_task(task, response)
        results.append(
            {
                "task_id": task["task_id"],
                "reward": graded.reward,
                "unit_pass_rate": graded.unit_pass_rate,
                "process_score": graded.process_score,
                "response": response,
                "grading_details": graded.details,
            }
        )
        print(f"  Reward: {graded.reward:.4f} | Unit: {graded.unit_pass_rate:.2f} | Process: {graded.process_score:.2f}")

    output = {
        "metadata": {
            "base_model": base_model,
            "resolved_model": resolve_model_id(base_model),
            "adapter_path": adapter_path,
            "num_tasks": len(tasks),
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - start,
            "task_ids": [task["task_id"] for task in tasks],
        },
        "metrics": compute_metrics(results),
        "results": results,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    run_numeric_benchmark(
        base_model=args.model,
        adapter_path=args.adapter,
        tasks_file=Path(args.tasks_file),
        output_file=Path(args.output),
        num_tasks=args.num_tasks,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
