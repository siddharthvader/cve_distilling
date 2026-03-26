"""Tests for PrimeVul numeric triage builders."""

from pathlib import Path

from rl_secdef.data.primevul_numeric import (
    TARGET_CWES,
    build_juliet_numeric_rows,
    build_primevul_numeric_rows,
)


def test_build_primevul_numeric_rows_with_synthetic_records():
    records = [
        {
            "idx": 1,
            "project": "projA",
            "commit_id": "a1",
            "cve": "CVE-1",
            "cwe": ["CWE-190"],
            "target": 1,
            "func": "int f(int n){ int x = n * 8; return x; }\n" * 4,
        },
        {
            "idx": 2,
            "project": "projB",
            "commit_id": "b2",
            "cve": "CVE-2",
            "cwe": ["CWE-190"],
            "target": 0,
            "func": "int g(int n){ if (n > 100) return -1; int x = n * 8; return x; }\n" * 4,
        },
        {
            "idx": 3,
            "project": "projC",
            "commit_id": "c3",
            "cve": "CVE-3",
            "cwe": ["CWE-787"],
            "target": 1,
            "func": "void h(char *p){ char buf[8]; strcpy(buf, p); }\n" * 4,
        },
    ]
    rows = build_primevul_numeric_rows(
        split="train",
        source_rows=records,
        hard_negative_ratio=2,
        distractor_ratio=1,
        include_response=True,
    )
    assert len(rows) == 3
    assert sum(1 for row in rows if row["vulnerable"]) == 1
    assert {row["metadata"]["source_category"] for row in rows} == {"positive", "hard_negative", "distractor_negative"}


def test_build_juliet_numeric_rows_filters_to_target_and_near_miss(tmp_path: Path):
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(
        "\n".join(
            [
                '{"task_id":"juliet_overflow","type":"detect","language":"c","grading":{"expected_cwe":"CWE-190"},"metadata":{"source":"juliet","filename":"CWE190_test.c","bad_code":"int f(int n){ int total = n * 8; int bytes = total + 16; return bytes; }","good_code":"int f(int n){ if(n>100) return -1; int total = n * 8; int bytes = total + 16; return bytes; }"}}',
                '{"task_id":"juliet_signedness","type":"detect","language":"c","grading":{"expected_cwe":"CWE-195"},"metadata":{"source":"juliet","filename":"CWE195_test.c","bad_code":"unsigned int f(int n){ unsigned x = (unsigned)n; unsigned y = x + 4; return y; }","good_code":"unsigned int f(int n){ if(n < 0) return 0; unsigned x = (unsigned)n; return x + 4; }"}}',
                '{"task_id":"juliet_other","type":"detect","language":"c","grading":{"expected_cwe":"CWE-78"},"metadata":{"source":"juliet","filename":"CWE78_test.c","bad_code":"system(cmd);","good_code":"execl(...);"}}',
            ]
        )
        + "\n"
    )
    rows = build_juliet_numeric_rows(tasks_file)
    assert any(row["vulnerable"] and row["subtype"] in TARGET_CWES for row in rows)
    assert any((not row["vulnerable"]) and row["metadata"]["source_category"] == "near_miss_negative" for row in rows)
    assert all("CWE-78" not in row["task_id"] for row in rows)
