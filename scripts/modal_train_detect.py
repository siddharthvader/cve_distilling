"""Train a detect-only QLoRA adapter on cleaned data using Modal."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME_NAME = os.environ.get("RL_SECDEF_MODAL_VOLUME", "rl-secdef-detect-artifacts")
REMOTE_ROOT = "/root/project"
REMOTE_SRC = f"{REMOTE_ROOT}/src"
REMOTE_DATA = f"{REMOTE_ROOT}/data"
REMOTE_ARTIFACTS = "/artifacts"

app = modal.App("rl-secdef-detect-train")
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
        "trl>=0.13.0",
        "sentencepiece>=0.2.0",
        "rich>=13.0.0",
    )
    .add_local_dir(str(ROOT / "src"), remote_path=REMOTE_SRC)
    .add_local_dir(str(ROOT / "data"), remote_path=REMOTE_DATA)
)


def _upload_to_volume(local_path: Path, remote_path: str) -> None:
    with volume.batch_upload(force=True) as batch:
        batch.put_file(local_path, remote_path)


def _upload_directory_to_volume(local_dir: Path, remote_dir: str) -> None:
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(local_dir, remote_dir)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={REMOTE_ARTIFACTS: volume},
)
def train_detect_remote(
    dataset_path: str,
    run_name: str,
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    resume_adapter_subdir: str = "",
    num_epochs: float = 2.0,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_train_rows: int = 0,
):
    import json
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments

    sys.path.insert(0, REMOTE_SRC)
    from rl_secdef.data.juliet_clean import load_training_rows

    os.environ["HF_HOME"] = f"{REMOTE_ARTIFACTS}/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = f"{REMOTE_ARTIFACTS}/hf_cache"

    dataset_file = Path(dataset_path)
    if not dataset_file.is_absolute():
        dataset_file = Path(REMOTE_ARTIFACTS) / dataset_file

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_training_rows(dataset_file, split="train")
    valid_rows = load_training_rows(dataset_file, split="valid")
    if max_train_rows > 0:
        train_rows = train_rows[:max_train_rows]

    train_ds = Dataset.from_list(train_rows)
    valid_ds = Dataset.from_list(valid_rows) if valid_rows else None

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    if resume_adapter_subdir:
        resume_adapter_path = Path(REMOTE_ARTIFACTS) / resume_adapter_subdir
        model = PeftModel.from_pretrained(model, str(resume_adapter_path), is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def build_messages(row):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]

    def build_supervised_example(row, max_length: int = 2048):
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            prompt_text = tokenizer.apply_chat_template(build_messages(row)[:-1], tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(build_messages(row), tokenize=False, add_generation_prompt=False)
        else:
            prompt_text = (
                "### System:\n"
                f"{SYSTEM_PROMPT}\n\n"
                "### Instruction:\n"
                f"{row['prompt']}\n\n"
                "### Response:\n"
            )
            full_text = prompt_text + row["response"]

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(full_ids)
        for i in range(min(len(prompt_ids), len(full_ids)), len(full_ids)):
            labels[i] = full_ids[i]

        if len(full_ids) > max_length:
            overflow = len(full_ids) - max_length
            drop_from_prompt = min(overflow, len(prompt_ids))
            if drop_from_prompt:
                full_ids = full_ids[drop_from_prompt:]
                labels = labels[drop_from_prompt:]
                prompt_ids = prompt_ids[drop_from_prompt:]
            if len(full_ids) > max_length:
                full_ids = full_ids[-max_length:]
                labels = labels[-max_length:]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "supervised_tokens": sum(token != -100 for token in labels),
        }

    train_ds = train_ds.map(build_supervised_example, remove_columns=train_ds.column_names)
    train_ds = train_ds.filter(lambda row: row["supervised_tokens"] > 0)
    if valid_ds is not None:
        valid_ds = valid_ds.map(build_supervised_example, remove_columns=valid_ds.column_names)
        valid_ds = valid_ds.filter(lambda row: row["supervised_tokens"] > 0)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

    mean_supervised_tokens = sum(train_ds["supervised_tokens"]) / max(len(train_ds), 1)

    output_dir = Path(REMOTE_ARTIFACTS) / "runs" / run_name / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    args_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )
    args_kwargs["eval_strategy"] = "steps" if valid_ds is not None else "no"
    if valid_ds is not None:
        args_kwargs["eval_steps"] = 25
    args = TrainingArguments(**args_kwargs)
    model.config.use_cache = False

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        args=args,
        data_collator=collator,
    )
    train_result = trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "train_loss": float(train_result.training_loss),
        "num_train_rows": len(train_ds),
        "num_valid_rows": len(valid_ds) if valid_ds is not None else 0,
        "resume_adapter_subdir": resume_adapter_subdir or None,
        "mean_supervised_tokens": mean_supervised_tokens,
    }
    metrics_path = Path(REMOTE_ARTIFACTS) / "runs" / run_name / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    volume.commit()
    return {"run_name": run_name, "remote_adapter_dir": str(output_dir), "metrics": metrics}


SYSTEM_PROMPT = "You are a security expert analyzing code for vulnerabilities. Provide detailed, technical responses."


@app.local_entrypoint()
def main(
    dataset_path: str = "data/juliet_clean_detect.jsonl",
    run_name: str = "qwen7b_juliet_clean",
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    resume_adapter: str = "",
    num_epochs: float = 2.0,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_train_rows: int = 0,
    local_output_dir: str = "models/modal",
):
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(dataset_file)

    remote_dataset_path = f"runs/{run_name}/{dataset_file.name}"
    _upload_to_volume(dataset_file, remote_dataset_path)

    remote_resume_adapter = ""
    if resume_adapter:
        resume_adapter_path = Path(resume_adapter)
        if not resume_adapter_path.exists() or not resume_adapter_path.is_dir():
            raise FileNotFoundError(resume_adapter_path)
        remote_resume_adapter = f"runs/{run_name}/seed_adapter"
        _upload_directory_to_volume(resume_adapter_path, remote_resume_adapter)

    result = train_detect_remote.remote(
        dataset_path=remote_dataset_path,
        run_name=run_name,
        base_model=base_model,
        resume_adapter_subdir=remote_resume_adapter,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_train_rows=max_train_rows,
    )
    print(result)

    local_dir = Path(local_output_dir) / run_name
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = f"runs/{run_name}"
    subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote_dir, str(local_dir), "--force"], check=True)
    try:
        volume.remove_file(remote_dataset_path, recursive=True)
    except Exception:
        pass
    if remote_resume_adapter:
        try:
            volume.remove_file(remote_resume_adapter, recursive=True)
        except Exception:
            pass
    print(f"Downloaded artifacts to {local_dir}")
