"""Evaluate a base or adapter-backed HF model on the numeric triage benchmark via Modal."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME_NAME = os.environ.get("RL_SECDEF_MODAL_VOLUME", "rl-secdef-numeric-artifacts")
REMOTE_ROOT = "/root/project"
REMOTE_SRC = f"{REMOTE_ROOT}/src"
REMOTE_DATA = f"{REMOTE_ROOT}/data"
REMOTE_ARTIFACTS = "/artifacts"

app = modal.App("rl-secdef-numeric-eval")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.48.0",
        "datasets>=3.2.0",
        "accelerate>=1.2.1",
        "bitsandbytes>=0.45.0",
        "peft>=0.14.0",
        "sentencepiece>=0.2.0",
        "rich>=13.0.0",
    )
    .add_local_dir(str(ROOT / "src"), remote_path=REMOTE_SRC)
    .add_local_dir(str(ROOT / "data"), remote_path=REMOTE_DATA)
)


def _upload_to_volume(local_path: Path, remote_path: str) -> None:
    with volume.batch_upload(force=True) as batch:
        batch.put_file(local_path, remote_path)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={REMOTE_ARTIFACTS: volume},
)
def eval_numeric_remote(
    tasks_file: str,
    run_name: str,
    output_name: str,
    base_model: str = "qwen-coder-7b",
    adapter_volume_subdir: str = "",
    max_new_tokens: int = 256,
    num_tasks: int = 0,
):
    sys.path.insert(0, REMOTE_SRC)
    from rl_secdef.benchmark_numeric import run_numeric_benchmark

    os.environ["HF_HOME"] = f"{REMOTE_ARTIFACTS}/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = f"{REMOTE_ARTIFACTS}/hf_cache"

    tasks_path = Path(tasks_file)
    if not tasks_path.is_absolute():
        tasks_path = Path(REMOTE_ARTIFACTS) / tasks_path

    adapter_path = ""
    if adapter_volume_subdir:
        adapter_path = str(Path(REMOTE_ARTIFACTS) / adapter_volume_subdir)

    output_path = Path(REMOTE_ARTIFACTS) / "benchmarks" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_numeric_benchmark(
        base_model=base_model,
        adapter_path=adapter_path or None,
        tasks_file=tasks_path,
        output_file=output_path,
        num_tasks=num_tasks,
        seed=42,
        max_new_tokens=max_new_tokens,
    )
    volume.commit()
    return {
        "run_name": run_name,
        "output_path": str(output_path),
        "avg_reward": result["metrics"]["overall"]["avg_reward"],
        "binary_accuracy": result["metrics"]["overall"]["binary_accuracy"],
        "exact_accuracy": result["metrics"]["overall"]["exact_accuracy"],
    }


@app.local_entrypoint()
def main(
    tasks_file: str = "data/primevul_numeric_triage_eval.jsonl",
    run_name: str = "qwen7b_primevul_numeric",
    output_name: str = "qwen7b_primevul_numeric_eval.json",
    base_model: str = "qwen-coder-7b",
    adapter: str = "",
    max_new_tokens: int = 256,
    num_tasks: int = 0,
    local_output_dir: str = "benchmarks",
):
    local_tasks = Path(tasks_file)
    if not local_tasks.exists():
        raise FileNotFoundError(local_tasks)

    remote_tasks_name = f"runs/{run_name}/{local_tasks.name}"
    _upload_to_volume(local_tasks, remote_tasks_name)

    remote_adapter_name = ""
    if adapter:
        adapter_path = Path(adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(adapter_path)
        if adapter_path.is_dir():
            remote_adapter_name = f"runs/{run_name}/{adapter_path.name}"
            with volume.batch_upload(force=True) as batch:
                batch.put_directory(adapter_path, remote_adapter_name)
        else:
            raise ValueError("adapter must be a directory containing a PEFT adapter")

    result = eval_numeric_remote.remote(
        tasks_file=remote_tasks_name,
        run_name=run_name,
        output_name=output_name,
        base_model=base_model,
        adapter_volume_subdir=remote_adapter_name,
        max_new_tokens=max_new_tokens,
        num_tasks=num_tasks,
    )
    print(result)

    local_dir = Path(local_output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_path = f"benchmarks/{output_name}"
    subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote_path, str(local_dir / output_name), "--force"], check=True)
    try:
        volume.remove_file(remote_tasks_name, recursive=True)
    except Exception:
        pass
    if remote_adapter_name:
        try:
            volume.remove_file(remote_adapter_name, recursive=True)
        except Exception:
            pass
    try:
        volume.remove_file(f"benchmarks/{output_name}", recursive=True)
    except Exception:
        pass
    print(f"Downloaded benchmark to {local_dir / output_name}")
