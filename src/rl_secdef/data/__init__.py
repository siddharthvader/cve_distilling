"""Data loading and normalization."""

from __future__ import annotations

from importlib import import_module

from .eval_builder import build_clean_eval_set

__all__ = [
    "TaskSpec",
    "normalize_juliet",
    "normalize_qa_tasks",
    "normalize_all",
    "create_detect_prompt",
    "create_patch_prompt",
    "normalize_bigvul",
    "save_bigvul_tasks",
    "get_cwe_statistics",
    "create_detect_prompt_bigvul",
    "create_patch_prompt_bigvul",
    "normalize_cvefixes",
    "normalize_cvefixes_to_file",
    "download_cvefixes",
    "connect_database",
    "query_file_level_fixes",
    "query_method_level_fixes",
    "load_from_huggingface",
    "CVEFixRecord",
    "get_cvefixes_cwe_statistics",
    "build_clean_eval_set",
    "build_juliet_detect_jsonl",
    "build_juliet_detect_rows",
    "build_bigvul_detect_jsonl",
    "build_bigvul_detect_rows",
    "create_numeric_triage_prompt",
    "template_numeric_response",
    "write_primevul_numeric_files",
    "write_juliet_numeric_file",
    "build_primevul_numeric_rows",
    "build_juliet_numeric_rows",
    "load_training_rows",
    "load_response_overrides",
    "normalize_training_row",
    "row_to_text",
    "load_eval_identity_keys",
    "strip_juliet_metadata",
]

_LAZY_EXPORTS = {
    "TaskSpec": (".normalize", "TaskSpec"),
    "normalize_juliet": (".normalize", "normalize_juliet"),
    "normalize_qa_tasks": (".normalize", "normalize_qa_tasks"),
    "normalize_all": (".normalize", "normalize_all"),
    "create_detect_prompt": (".normalize", "create_detect_prompt"),
    "create_patch_prompt": (".normalize", "create_patch_prompt"),
    "normalize_bigvul": (".bigvul", "normalize_bigvul"),
    "save_bigvul_tasks": (".bigvul", "save_bigvul_tasks"),
    "get_cwe_statistics": (".bigvul", "get_cwe_statistics"),
    "create_detect_prompt_bigvul": (".bigvul", "create_detect_prompt_bigvul"),
    "create_patch_prompt_bigvul": (".bigvul", "create_patch_prompt_bigvul"),
    "normalize_cvefixes": (".cvefixes_loader", "normalize_cvefixes"),
    "normalize_cvefixes_to_file": (".cvefixes_loader", "normalize_cvefixes_to_file"),
    "download_cvefixes": (".cvefixes_loader", "download_cvefixes"),
    "connect_database": (".cvefixes_loader", "connect_database"),
    "query_file_level_fixes": (".cvefixes_loader", "query_file_level_fixes"),
    "query_method_level_fixes": (".cvefixes_loader", "query_method_level_fixes"),
    "load_from_huggingface": (".cvefixes_loader", "load_from_huggingface"),
    "CVEFixRecord": (".cvefixes_loader", "CVEFixRecord"),
    "get_cvefixes_cwe_statistics": (".cvefixes_loader", "get_cwe_statistics"),
    "build_juliet_detect_jsonl": (".juliet_clean", "build_juliet_detect_jsonl"),
    "build_juliet_detect_rows": (".juliet_clean", "build_juliet_detect_rows"),
    "build_bigvul_detect_jsonl": (".bigvul_clean", "build_bigvul_detect_jsonl"),
    "build_bigvul_detect_rows": (".bigvul_clean", "build_bigvul_detect_rows"),
    "create_numeric_triage_prompt": (".primevul_numeric", "create_numeric_triage_prompt"),
    "template_numeric_response": (".primevul_numeric", "template_numeric_response"),
    "write_primevul_numeric_files": (".primevul_numeric", "write_primevul_numeric_files"),
    "write_juliet_numeric_file": (".primevul_numeric", "write_juliet_numeric_file"),
    "build_primevul_numeric_rows": (".primevul_numeric", "build_primevul_numeric_rows"),
    "build_juliet_numeric_rows": (".primevul_numeric", "build_juliet_numeric_rows"),
    "load_training_rows": (".juliet_clean", "load_training_rows"),
    "load_response_overrides": (".juliet_clean", "load_response_overrides"),
    "normalize_training_row": (".juliet_clean", "normalize_training_row"),
    "row_to_text": (".juliet_clean", "row_to_text"),
    "load_eval_identity_keys": (".bigvul_clean", "load_eval_identity_keys"),
    "strip_juliet_metadata": (".juliet_clean", "strip_juliet_metadata"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
