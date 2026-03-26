"""Benchmark HuggingFace/PEFT causal LM models on security tasks."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from rl_secdef.runner.graders import grade_detect_task, grade_patch_task, grade_qa_task

SYSTEM_PROMPT = "You are a security expert analyzing code for vulnerabilities. Provide detailed, technical responses."

MODEL_SHORTCUTS = {
    "qwen-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen-coder-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
}


def resolve_model_id(model: str) -> str:
    return MODEL_SHORTCUTS.get(model, model)


def load_tasks(tasks_file: Path) -> List[dict]:
    tasks = []
    with open(tasks_file) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def sample_tasks(tasks: List[dict], num_tasks: int, seed: int = 42) -> List[dict]:
    random.seed(seed)
    task_types = ["detect", "patch", "qa"]
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for task in tasks:
        if task["type"] in task_types:
            by_type[task["type"]].append(task)

    available_types = [t for t in task_types if by_type[t]]
    if not available_types:
        return []

    per_type = num_tasks // len(available_types)
    remainder = num_tasks % len(available_types)
    sampled = []
    for i, task_type in enumerate(available_types):
        n = per_type + (1 if i < remainder else 0)
        sampled.extend(random.sample(by_type[task_type], min(n, len(by_type[task_type]))))
    random.shuffle(sampled)
    return sampled


def select_tasks(tasks: List[dict], num_tasks: int, seed: int = 42) -> List[dict]:
    if num_tasks <= 0 or num_tasks >= len(tasks):
        return list(tasks)
    return sample_tasks(tasks, num_tasks, seed=seed)


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = True,
):
    model_id = resolve_model_id(base_model)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def query_hf_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    inputs = tokenizer(formatted, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def grade_response(task: dict, response: str):
    task_type = task["type"]
    if task_type == "detect":
        return grade_detect_task(task, response)
    if task_type == "patch":
        return grade_patch_task(task, response, sandbox=None)
    if task_type == "qa":
        return grade_qa_task(task, response)
    raise ValueError(f"Unknown task type: {task_type}")


def compute_metrics(results: List[dict]) -> Dict[str, Any]:
    if not results:
        return {}
    metrics = {
        "overall": {
            "num_tasks": len(results),
            "avg_reward": sum(r["reward"] for r in results) / len(results),
            "avg_unit_pass_rate": sum(r["unit_pass_rate"] for r in results) / len(results),
            "avg_process_score": sum(r["process_score"] for r in results) / len(results),
        },
        "by_type": {},
    }
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for result in results:
        by_type[result["task_type"]].append(result)
    for task_type, rows in by_type.items():
        metrics["by_type"][task_type] = {
            "num_tasks": len(rows),
            "avg_reward": sum(r["reward"] for r in rows) / len(rows),
            "avg_unit_pass_rate": sum(r["unit_pass_rate"] for r in rows) / len(rows),
            "avg_process_score": sum(r["process_score"] for r in rows) / len(rows),
        }
    return metrics


def run_hf_benchmark(
    base_model: str,
    adapter_path: Optional[str],
    tasks_file: Path,
    output_file: Path,
    num_tasks: int = 0,
    seed: int = 42,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    load_in_4bit: bool = True,
) -> dict:
    model, tokenizer = load_model_and_tokenizer(base_model, adapter_path=adapter_path, load_in_4bit=load_in_4bit)
    all_tasks = load_tasks(tasks_file)
    tasks = select_tasks(all_tasks, num_tasks, seed=seed)

    type_counts = defaultdict(int)
    source_counts = defaultdict(int)
    for task in tasks:
        type_counts[task["type"]] += 1
        source_counts[task.get("metadata", {}).get("source", "unknown")] += 1

    results = []
    start = time.time()
    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] {task['task_id']} ({task['type']})")
        response = query_hf_model(
            model,
            tokenizer,
            task["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        graded = grade_response(task, response)
        results.append(
            {
                "task_id": task["task_id"],
                "task_type": task["type"],
                "reward": graded.reward,
                "unit_pass_rate": graded.unit_pass_rate,
                "process_score": graded.process_score,
                "response": response,
                "grading_details": graded.details,
            }
        )
        print(
            f"  Reward: {graded.reward:.4f} | Unit: {graded.unit_pass_rate:.2f} | Process: {graded.process_score:.2f}"
        )

    metrics = compute_metrics(results)
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
            "source_counts": dict(source_counts),
        },
        "metrics": metrics,
        "results": results,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model or shortcut")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path")
    parser.add_argument("--tasks-file", required=True, help="Task JSONL")
    parser.add_argument("--output", required=True, help="Benchmark output JSON")
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    args = parser.parse_args()

    run_hf_benchmark(
        base_model=args.model,
        adapter_path=args.adapter,
        tasks_file=Path(args.tasks_file),
        output_file=Path(args.output),
        num_tasks=args.num_tasks,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        load_in_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
