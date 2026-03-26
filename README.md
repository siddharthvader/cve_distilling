# cve_distilling

Clean experiment artifacts for a narrow code-security distillation study.

This repo packages the subset of code, data, benchmark outputs, and notes from a larger research workspace that turned out to be worth keeping after the benchmark was cleaned up.

## Main Result

The original broad claim did **not** survive cleanup.

What *did* survive was narrower and more believable:

- a `7B` Qwen model can be made competitive with, and on one metric slightly better than, `GPT-5.2`
- on a **narrow real-world security task**
- using **public real-world data** plus **frontier-model distillation**

The task was:

- C/C++ numeric vulnerability triage
- only `CWE-190` and `CWE-191`
- strict structured output:
  - `vulnerable`
  - `subtype`
  - `location`
  - `reason`

This is **not** a claim about general vulnerability detection or patch generation.

## Headline Results

All models below were evaluated on the same frozen `140`-example PrimeVul test set:

- `20` true numeric vulnerabilities
- `120` negatives / distractors

Because the benchmark is negative-heavy, the most useful metric is:

`balanced binary accuracy = (positive recall + negative accuracy) / 2`

| Model | Balanced Binary Acc | Positive Recall | Negative Accuracy | Read |
| --- | ---: | ---: | ---: | --- |
| `Qwen + Juliet -> PrimeVul distilled` | **73.8%** | **95.0%** | 52.5% | Best recall, most aggressive |
| `GPT-5.2` | 70.8% | 85.0% | 56.7% | Strong frontier baseline |
| `Qwen + PrimeVul distilled` | 70.0% | 85.0% | 55.0% | Roughly tied with GPT-5.2 |
| `Qwen base` | 63.8% | 30.0% | **97.5%** | Very conservative |
| `Qwen + Juliet stage 1` | 50.0% | 0.0% | **100.0%** | Learned to always say `NONE` |

Short version:

- `Juliet alone` did not transfer
- `PrimeVul + GPT-5.2-distilled targets` created the real lift
- `Juliet` may provide a small warm-start benefit before the real-world distilled stage

## What This Repo Shows

The work here supports a narrower conclusion than the original broad story:

- **Synthetic-only security fine-tuning did not hold up**
- **Narrow, structured, real-world distillation worked**
- **A small model can compress frontier behavior on a repeated workflow**

That is closer to:

- "build a cheaper specialist for a narrow task"

than:

- "beat frontier models at security in general"

## Important Caveats

These matter.

1. This is a **narrow task**, not general vulnerability detection.

2. This is **distillation**, not independent general reasoning.
   - the best student models were trained on `GPT-5.2`-generated targets from the same task family

3. The benchmark has **no exact train/test overlap**:
   - task ID overlap: `0`
   - commit overlap: `0`
   - exact prompt overlap: `0`
   - exact code overlap: `0`

4. There are still **weaker overlap risks**:
   - same public corpus family (`PrimeVul`)
   - same-project overlap
   - some shared CVEs between train and eval

5. The best-performing student is **more aggressive** than the base model.
   - it catches more real positives
   - it also produces more false positives

So the right interpretation is:

- this is evidence that a small open model can become competitive on a narrow, structured workflow
- it is **not** evidence that a small model has better general reasoning than a frontier model

## Broad Benchmark Context

Before narrowing the task, the broader benchmark was rebuilt to remove leakage and make comparisons fair.

On that cleaned broad BigVul detection benchmark:

- `GPT-5.2 > Qwen base > old Qwen SFT`

That was the key reason to abandon the broad claim and focus on a narrower experiment that held up better.

## Repo Layout

- `EXPERIMENT_NOTES_2026-03-26.md`
  Full writeup with methodology, result tables, contamination checks, and interpretation.

- `benchmarks/`
  Saved benchmark outputs for the cleaned broad benchmark and the numeric-triage runs.

- `data/`
  Frozen eval sets, manifests, and numeric training datasets.

- `scripts/`
  Dataset builders, distillation scripts, Modal train/eval entrypoints, and the numeric rebalancer.

- `src/rl_secdef/`
  The subset of package code needed to build and benchmark the cleaned experiments.

- `tests/`
  Focused tests for the numeric-triage pipeline.

## Most Relevant Files

### Results

- `benchmarks/qwen7b_primevul_numeric_base_eval.json`
- `benchmarks/qwen7b_juliet_numeric_stage1_eval.json`
- `benchmarks/gpt52_primevul_numeric_eval.json`
- `benchmarks/qwen7b_primevul_numeric_distilled_eval.json`
- `benchmarks/qwen7b_juliet_primevul_numeric_distilled_eval.json`

### Data

- `data/primevul_numeric_triage_train.jsonl`
- `data/primevul_numeric_triage_train_distilled.jsonl`
- `data/primevul_numeric_triage_eval.jsonl`
- `data/primevul_numeric_triage_train.manifest.json`
- `data/juliet_numeric_triage.jsonl`
- `data/juliet_numeric_triage.manifest.json`

### Key code

- `scripts/build_primevul_numeric_triage.py`
- `scripts/build_juliet_numeric_triage.py`
- `scripts/distill_numeric_triage.py`
- `scripts/modal_train_detect.py`
- `scripts/modal_eval_numeric.py`
- `scripts/rebalance_numeric_triage.py`
- `src/rl_secdef/data/primevul_numeric.py`
- `src/rl_secdef/runner/numeric_triage.py`
- `src/rl_secdef/benchmark_numeric.py`

## Quick Verification

Focused tests:

```bash
python3 -m pytest tests/test_numeric_triage.py tests/test_primevul_numeric.py tests/test_rebalance_numeric.py -q
```

The extracted numeric experiment bundle is intentionally small. This repo is not meant to be a full mirror of the original workspace.

## If You Read One File

Read:

- [EXPERIMENT_NOTES_2026-03-26.md](./EXPERIMENT_NOTES_2026-03-26.md)

That file contains the full story:

- what failed
- what was fixed
- what actually worked
- and what claims are still too strong
