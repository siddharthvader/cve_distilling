# Experiment Notes: Repo Audit, Cleaned Eval, and Narrow Numeric Triage

Date: 2026-03-26

This note records the main findings from the repo audit, the cleaned broad evaluation, the narrowed numeric-triage experiments, the exact benchmark outputs worth remembering, and the caveats that matter if these results are referenced later.

## TL;DR

The original broad benchmark setup was not trustworthy enough to support the original claims. The main issues were leaked labels in prompts, artifact-laden CVEfixes preprocessing, patch grading that rewarded lexical similarity instead of behavior, and non-identical task sets across model comparisons.

After fixing the worst benchmark issues, the original broad Qwen SFT no longer beat the base model. On the cleaned blinded BigVul detection benchmark, the ordering was:

- `GPT-5.2 > Qwen base > old Qwen SFT`

That pushed the work toward a narrower task:

- C/C++ numeric triage for `CWE-190` and `CWE-191`

On that narrow task:

- `Juliet alone` did not help and actually collapsed into always predicting `NONE`
- `PrimeVul + GPT-5.2-distilled targets` created real lift over base Qwen
- `Juliet -> PrimeVul-distilled` slightly outperformed both `PrimeVul-distilled` and `GPT-5.2` on balanced accuracy, but by becoming more aggressive and accepting more false positives

The most honest summary is:

- Public data can be enough to make a 7B model competitive with a frontier model on a narrow, structured, distribution-matched task.
- Public synthetic data by itself, especially Juliet-only, did not transfer well to real-world vulnerability triage.
- Real-world matched distillation was the key lever.

## 1. Repo Audit: Why the Original Broad Claims Were Not Trustworthy

### Main flaws found

1. Detection labels were leaked in the prompt.
   - BigVul and CVEfixes detect prompts included the expected CWE directly in task text.
   - The detect grader gave unit credit for reproducing the expected CWE string.

2. CVEfixes preprocessing was contaminated by artifacts.
   - Tasks contained `S2SV_*` markers and fragmentary fix text.
   - This made both detection and patching partially solvable via artifact matching.

3. Patch grading was not behavioral.
   - Patch grading used lexical similarity and regex-based pseudo-static-analysis instead of compiling/running tests.
   - That made patch reward easy to hack by generating reference-like text.

4. Cross-model comparisons were not apples-to-apples.
   - Different benchmark runs sampled different task sets.
   - The repo compared models on overlapping but non-identical data.

5. Train/test separation was partly enforced post hoc.
   - Juliet contamination was fixed later by filtering results rather than by dataset construction.

6. The RL path was not mature enough to support strong claims.
   - The MLX RL path had no real optimizer step in the earlier implementation path examined.
   - PPO was still stub-like.

### Practical consequence

The original story looked much more like:

- "the model got better at producing benchmark-shaped outputs"

than:

- "the model genuinely improved at held-out vulnerability analysis"

That is why the broad benchmark was cleaned first, and why the work shifted toward a narrower task.

## 2. Cleaned Broad Benchmark

### Cleaned benchmark design

The broad benchmark was rebuilt as a blinded BigVul-only detect benchmark:

- dataset file: `data/eval_bigvul_detect_blind.jsonl`
- size: `44` tasks
- no leaked `CWE-*` labels in prompt
- no CVE or commit-hint leakage in prompt
- all models evaluated on the same frozen file

Detection grading was also tightened so it no longer gave full credit for simply echoing a leaked label.

### Broad cleaned results

Files:

- `benchmarks/qwen7b_base_blind_detect.json`
- `benchmarks/qwen7b_sft_blind_detect.json`
- `benchmarks/gpt52_blind_detect.json`

Summary:

| Model | Tasks | Avg Reward | Avg Unit Pass |
| --- | ---: | ---: | ---: |
| Qwen base | 44 | 0.3509 | 0.1136 |
| Old Qwen SFT | 44 | 0.3398 | 0.0795 |
| GPT-5.2 | 44 | 0.4023 | 0.2045 |

### Broad benchmark takeaway

The original broad fine-tune did not survive benchmark cleanup.

- The old SFT was slightly worse than base Qwen.
- GPT-5.2 remained clearly better.

This was the main reason to stop chasing a broad "general vulnerability detection" win and instead define a narrower task with better structure and better in-distribution data.

## 3. Narrow Task Chosen

The new task was:

- `C/C++ numeric vulnerability triage`

Limited to:

- `CWE-190` Integer Overflow or Wraparound
- `CWE-191` Integer Underflow

Prompt framing:

- only care about those two CWEs
- ignore unrelated vulnerability types
- return strict JSON with four keys:
  - `vulnerable`
  - `subtype`
  - `location`
  - `reason`

The point of the narrow task was not to beat a frontier model at "general reasoning". It was to see whether a smaller model could become competitive on a narrow, repeated, structured task with public but distribution-matched data.

## 4. Numeric Triage Data Pipeline

### Core scripts and modules

- dataset builder: `src/rl_secdef/data/primevul_numeric.py`
- PrimeVul build script: `scripts/build_primevul_numeric_triage.py`
- Juliet build script: `scripts/build_juliet_numeric_triage.py`
- distillation script: `scripts/distill_numeric_triage.py`
- benchmark runner: `src/rl_secdef/benchmark_numeric.py`
- grading logic: `src/rl_secdef/runner/numeric_triage.py`
- Modal eval wrapper: `scripts/modal_eval_numeric.py`
- data rebalancer prepared for follow-up: `scripts/rebalance_numeric_triage.py`

### PrimeVul numeric train/eval split

Source:

- HuggingFace dataset `colin/PrimeVul`

Construction:

- train data came from PrimeVul `train` + `validation`
- eval data came from PrimeVul `test`
- positives were functions labeled with exactly one target CWE in `{CWE-190, CWE-191}`
- hard negatives were `target == 0` rows with target numeric CWE labels
- distractor negatives were positive vulnerabilities from other CWE families, but labeled as `NONE` for this task

Manifest:

- `data/primevul_numeric_triage_train.manifest.json`

Counts:

- train rows: `876`
- train split counts:
  - `train: 764`
  - `valid: 112`
- train categories:
  - `positive: 219`
  - `hard_negative: 438`
  - `distractor_negative: 219`
- train positive subtypes:
  - `CWE-190: 211`
  - `CWE-191: 8`

- eval rows: `140`
- eval categories:
  - `positive: 20`
  - `hard_negative: 60`
  - `distractor_negative: 60`
- eval positive subtypes:
  - `CWE-190: 17`
  - `CWE-191: 3`

Important note:

- the eval set is negative-heavy
- because of that, `avg_reward` and raw binary accuracy are not the best summary metrics
- the most useful metric for this benchmark is:

`balanced binary accuracy = (positive recall + negative accuracy) / 2`

### Juliet numeric curriculum data

Source:

- cleaned Juliet tasks already present in `data/tasks.jsonl`

Construction:

- positives from Juliet bad variants for target CWEs
- clean negatives from corresponding good variants
- near-miss negatives from adjacent numeric CWEs such as `CWE-195`, `CWE-197`, `CWE-131`, etc.

Manifest:

- `data/juliet_numeric_triage.manifest.json`

Counts:

- total rows: `74`
- split counts:
  - `train: 64`
  - `valid: 10`
- categories:
  - `positive: 19`
  - `near_miss_negative: 18`
  - `clean_negative: 37`
- positive subtypes:
  - `CWE-190: 10`
  - `CWE-191: 9`

### Distilled supervision

The key stage-2 training file was:

- `data/primevul_numeric_triage_train_distilled.jsonl`

It was produced by `scripts/distill_numeric_triage.py`, which:

- uses `gpt-5.2` as the teacher
- passes the prompt plus a hidden label:
  - `vulnerable`
  - `subtype`
- asks the teacher to emit only the final JSON target

This means the student is not learning from raw dataset labels only. It is learning a teacher-produced response format grounded in hidden labels, which is a more realistic SFT target for this narrow task.

## 5. Training Methodology

### Model family

Base model:

- `Qwen/Qwen2.5-Coder-7B-Instruct`

Local benchmark shorthand:

- `qwen-coder-7b`

### Training infrastructure

Training and numeric eval were run on Modal.

At the end of the run:

- all Modal apps were stopped
- the temporary volume `rl-secdef-numeric-artifacts` was deleted

### Important training fix

One important bug had to be fixed before these experiments were meaningful:

- the earlier SFT path optimized over the whole chat transcript instead of masking loss to assistant tokens

The current training path uses assistant-only masking, which was necessary to avoid collapse in prior runs.

### Numeric training runs

1. `Qwen + Juliet stage 1`
   - dataset: `data/juliet_numeric_triage.jsonl`
   - run dir: `models/modal/qwen7b_juliet_numeric_stage1/qwen7b_juliet_numeric_stage1`
   - train rows: `64`
   - valid rows: `10`
   - mean supervised tokens: `49.109375`
   - train loss: `0.8608459929625193`

2. `Qwen + PrimeVul distilled`
   - dataset: `data/primevul_numeric_triage_train_distilled.jsonl`
   - run dir: `models/modal/qwen7b_primevul_numeric_distilled/qwen7b_primevul_numeric_distilled`
   - train rows: `764`
   - valid rows: `112`
   - mean supervised tokens: `53.94502617801047`
   - train loss: `0.8596245317409436`

3. `Qwen + Juliet -> PrimeVul distilled`
   - same distilled PrimeVul dataset as above
   - warm-started from the Juliet stage-1 adapter
   - run dir: `models/modal/qwen7b_juliet_primevul_numeric_distilled/qwen7b_juliet_primevul_numeric_distilled`
   - train rows: `764`
   - valid rows: `112`
   - mean supervised tokens: `53.94502617801047`
   - train loss: `0.8071326787273089`

### Rebalanced follow-up data

A rebalanced distilled training file was also prepared for future work:

- `data/primevul_numeric_triage_train_distilled_rebalanced.jsonl`

Manifest:

- `data/primevul_numeric_triage_train_distilled_rebalanced.jsonl.manifest.json`

Key properties:

- train positives rebalanced to match hard negatives
- distractor negatives removed from train
- positive subtypes balanced

Counts:

- train positives: `382`
- train hard negatives: `382`
- positive subtypes:
  - `CWE-190: 190`
  - `CWE-191: 192`

This rebalanced set was prepared but not run during this session.

## 6. Numeric Benchmark Methodology

The numeric grader in `src/rl_secdef/runner/numeric_triage.py` assigns:

- exact positive hit: `1.0`
- right binary call but wrong subtype on positives: `0.5`
- exact negative hit: `1.0`
- right binary call but wrong subtype on negatives: `0.5`

Process score:

- `1.0` if valid JSON with all required keys
- `0.5` if parsed JSON but missing required fields
- `0.0` for invalid JSON

Reward:

- `0.7 * unit_pass_rate + 0.3 * process_score`

For this benchmark, the metrics worth watching are:

- positive binary recall
- negative binary accuracy
- balanced binary accuracy
- exact positive recall

This matters because:

- base Qwen can look very strong on reward/accuracy simply by saying `NONE` almost all the time
- that behavior is not useful if it misses most true positives

## 7. Numeric Benchmark Results

Benchmark files:

- `benchmarks/qwen7b_primevul_numeric_base_eval.json`
- `benchmarks/qwen7b_juliet_numeric_stage1_eval.json`
- `benchmarks/gpt52_primevul_numeric_eval.json`
- `benchmarks/qwen7b_primevul_numeric_distilled_eval.json`
- `benchmarks/qwen7b_juliet_primevul_numeric_distilled_eval.json`

### Main comparison table

| Model | Avg Reward | Binary Acc | Exact Acc | Positive Recall | Positive Exact Recall | Negative Acc | Balanced Binary Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen base | 0.9100 | 0.8786 | 0.8643 | 0.3000 | 0.2000 | 0.9750 | 0.6375 |
| Qwen + Juliet stage 1 | 0.9000 | 0.8571 | 0.8571 | 0.0000 | 0.0000 | 1.0000 | 0.5000 |
| GPT-5.2 | 0.7200 | 0.6071 | 0.5929 | 0.8500 | 0.7500 | 0.5667 | 0.7083 |
| Qwen + PrimeVul distilled | 0.7054 | 0.5929 | 0.5714 | 0.8500 | 0.7000 | 0.5500 | 0.7000 |
| Qwen + Juliet -> PrimeVul distilled | 0.7025 | 0.5857 | 0.5643 | 0.9500 | 0.8000 | 0.5250 | 0.7375 |

### Hard-negative vs distractor-negative breakdown

| Model | Hard Negative Acc | Distractor Negative Acc |
| --- | ---: | ---: |
| Qwen base | 1.0000 | 0.9500 |
| Qwen + Juliet stage 1 | 1.0000 | 1.0000 |
| GPT-5.2 | 0.7500 | 0.3833 |
| Qwen + PrimeVul distilled | 0.6833 | 0.4167 |
| Qwen + Juliet -> PrimeVul distilled | 0.6667 | 0.3833 |

### What these numbers mean

1. Base Qwen is highly conservative.
   - It is very strong on negatives.
   - It misses most real positives.

2. Juliet-only training is bad for real-world transfer.
   - It learned to always output `NONE`.
   - Balanced accuracy collapsed to `0.5000`, which is effectively useless here.

3. PrimeVul-distilled training is the main source of lift.
   - It raises positive recall from `0.30` to `0.85`.
   - It gives up a lot of specificity to do that.

4. The Juliet warm start may provide a small additional lift on top of PrimeVul-distilled training.
   - It pushes positive recall from `0.85` to `0.95`
   - It gives up additional negative accuracy
   - Balanced accuracy still improves from `0.7000` to `0.7375`

## 8. Did Juliet Actually Help?

Short answer:

- `Juliet alone`: no
- `Juliet warm start before PrimeVul-distilled`: maybe a small real lift

### Paired comparison: PrimeVul-distilled vs Juliet -> PrimeVul-distilled

Comparing those two models on the exact same 140 eval tasks:

- `2` positive examples where Juliet warm start was right and PrimeVul-only was wrong
- `0` positive examples where PrimeVul-only was right and Juliet warm start was wrong
- `3` negative examples where PrimeVul-only was right and Juliet warm start was wrong
- `135` examples where they behaved the same

That means the warm start likely changed behavior on only `5` examples total.

Interpretation:

- there is a small recall-biased lift
- it is not a large effect
- it is plausible, but not definitive enough to make a strong standalone claim about Juliet

The safest formulation is:

- Juliet did not transfer by itself
- Juliet may provide a small warm-start benefit before the real-world distilled stage

## 9. Contamination / Leakage Check

### Exact contamination checks

Direct overlap checks between:

- `data/primevul_numeric_triage_train_distilled.jsonl`
- `data/primevul_numeric_triage_eval.jsonl`

Results:

- task ID overlap: `0`
- commit ID overlap: `0`
- exact prompt overlap: `0`
- exact extracted-code overlap: `0`
- row index overlap: `0`

So there is no exact train/test contamination in the obvious sense.

### Weaker overlap risks

There are still weaker sources of optimism:

- same dataset family: both train and eval come from PrimeVul
- same prompt format and grading format
- same-project overlap: `32` projects appear in both train and eval
- shared CVEs: `4`

Shared CVEs:

- `CVE-2010-4345`
- `CVE-2020-10726`
- `CVE-2021-41133`
- `CVE-2022-32545`

Those account for:

- `16 / 140` eval rows

This is not exact task leakage, but it is a weaker benchmark-contamination risk because train and eval can still come from the same broader vulnerability clusters.

### Shared-CVE exclusion check

After excluding the `16` eval rows whose CVEs also appear in the distilled training set:

| Model | Rows | Positive Recall | Positive Exact Recall | Negative Acc | Balanced Binary Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen base | 124 | 0.3333 | 0.2222 | 0.9717 | 0.6525 |
| GPT-5.2 | 124 | 0.8889 | 0.8333 | 0.5849 | 0.7369 |
| Qwen + PrimeVul distilled | 124 | 0.8889 | 0.7222 | 0.5849 | 0.7369 |
| Qwen + Juliet -> PrimeVul distilled | 124 | 1.0000 | 0.8333 | 0.5566 | 0.7783 |

Interpretation:

- the narrow-task lift does not disappear when shared-CVE rows are removed
- the best run still remains strong on the stricter slice

So the result is not obviously explained away by direct contamination.

## 10. Important Caveats

These are the caveats that should travel with any future reference to these results.

1. This is a narrow task.
   - The result is not "small model beats frontier on vulnerability detection."
   - The result is only about a narrow structured task:
     - numeric vulnerability triage for `CWE-190` / `CWE-191`

2. This is distillation, not independent general reasoning.
   - The student was trained on `GPT-5.2`-generated targets from the same task family.
   - A win here means:
     - "the small model can imitate or compress the teacher well on this distribution"
   - It does not mean:
     - "the small model has better general reasoning than the teacher"

3. The eval is still from the same public corpus family as the training data.
   - There is no exact overlap.
   - But there is same-dataset, same-project, and some shared-CVE overlap.

4. The current best model wins mainly on recall.
   - It is more aggressive than base Qwen.
   - It catches more true positives.
   - It also creates more false positives, especially on distractor negatives.

5. One PrimeVul-distilled run emitted one invalid JSON output.
   - The benchmark now handles this correctly as a miss.
   - Invalid JSON count in the saved benchmark:
     - `qwen_primevul_distilled`: `1`
     - `qwen_juliet_primevul_distilled`: `0`

6. The rebalanced distilled dataset was built but not trained yet.
   - That means there is still an obvious next experiment not yet executed:
     - try to keep the recall lift while recovering specificity

## 11. Best Current Interpretation

The best interpretation of the work so far is:

1. The original broad benchmark claims were not reliable.

2. After cleaning the benchmark, the original broad SFT did not help.

3. Juliet by itself is not a good standalone training source for real-world transfer on this task.

4. Public real-world data plus frontier-teacher distillation can produce meaningful specialization on a narrow task.

5. On this narrow numeric triage benchmark:
   - `Qwen + PrimeVul distilled` is roughly competitive with `GPT-5.2`
   - `Qwen + Juliet -> PrimeVul distilled` slightly outperforms `GPT-5.2` on balanced accuracy and positive recall
   - but it does so by becoming more aggressive, not by being uniformly cleaner

If this work is described externally, the safest honest claim is something like:

- "A 7B Qwen model, fine-tuned on a narrow real-world numeric vulnerability triage task using frontier-model distilled targets, became competitive with or slightly better than GPT-5.2 on a small frozen benchmark for that task."

That is much more defensible than any broad security-modeling claim.

## 12. Suggested Next Steps

If the goal is to turn this into a stronger internal result, the next steps should be:

1. Train on the rebalanced distilled dataset.
   - Goal: recover specificity without giving up too much recall.

2. Build a stricter eval split.
   - no shared CVEs
   - optionally no shared projects

3. Add a second dataset family if available.
   - The current result is narrow and promising, but still tied to PrimeVul.

4. Keep the output structured.
   - JSON classification/triage appears much more reliable than open-ended prose for this use case.

5. Resist broadening too early.
   - The current positive signal came from narrowing the task.

## 13. Artifact Index

### Broad cleaned benchmark

- benchmark file: `data/eval_bigvul_detect_blind.jsonl`
- base result: `benchmarks/qwen7b_base_blind_detect.json`
- old SFT result: `benchmarks/qwen7b_sft_blind_detect.json`
- GPT result: `benchmarks/gpt52_blind_detect.json`

### Numeric datasets

- PrimeVul train: `data/primevul_numeric_triage_train.jsonl`
- PrimeVul eval: `data/primevul_numeric_triage_eval.jsonl`
- PrimeVul manifest: `data/primevul_numeric_triage_train.manifest.json`
- Juliet numeric train: `data/juliet_numeric_triage.jsonl`
- Juliet manifest: `data/juliet_numeric_triage.manifest.json`
- PrimeVul distilled train: `data/primevul_numeric_triage_train_distilled.jsonl`
- Rebalanced distilled train: `data/primevul_numeric_triage_train_distilled_rebalanced.jsonl`
- Rebalanced manifest: `data/primevul_numeric_triage_train_distilled_rebalanced.jsonl.manifest.json`

### Numeric model artifacts

- Juliet stage 1 adapter: `models/modal/qwen7b_juliet_numeric_stage1/qwen7b_juliet_numeric_stage1`
- PrimeVul distilled adapter: `models/modal/qwen7b_primevul_numeric_distilled/qwen7b_primevul_numeric_distilled`
- Juliet -> PrimeVul distilled adapter: `models/modal/qwen7b_juliet_primevul_numeric_distilled/qwen7b_juliet_primevul_numeric_distilled`

### Numeric benchmark outputs

- base Qwen: `benchmarks/qwen7b_primevul_numeric_base_eval.json`
- Juliet stage 1: `benchmarks/qwen7b_juliet_numeric_stage1_eval.json`
- GPT-5.2: `benchmarks/gpt52_primevul_numeric_eval.json`
- PrimeVul distilled: `benchmarks/qwen7b_primevul_numeric_distilled_eval.json`
- Juliet -> PrimeVul distilled: `benchmarks/qwen7b_juliet_primevul_numeric_distilled_eval.json`

## 14. Final Bottom Line

The earlier benchmark was reward-hackable enough that its headline results should not be trusted.

The narrowed numeric-triage work is much more credible.

What actually seems to work is:

- narrow the task
- use real-world matched data
- use structured teacher-distilled targets
- evaluate on a frozen benchmark with no exact overlap

What did not work is:

- broad public synthetic security fine-tuning
- Juliet-only transfer
- trusting lexical or leakage-prone benchmarks

