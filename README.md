# cve_distilling

Clean experiment artifacts extracted from the larger `rl_security` workspace.

This repo contains the subset of code, data, benchmark outputs, and notes for the cleaned experiments completed on 2026-03-26:

- broad cleaned BigVul detection benchmark
- narrow C/C++ numeric triage benchmark for `CWE-190` / `CWE-191`
- Juliet curriculum experiments
- PrimeVul + `GPT-5.2` distillation experiments

The main writeup is:

- [EXPERIMENT_NOTES_2026-03-26.md](./EXPERIMENT_NOTES_2026-03-26.md)

## Contents

- `benchmarks/`
  Saved benchmark outputs referenced in the note.

- `data/`
  Frozen eval files, training manifests, and the numeric training datasets used in the experiments.

- `scripts/`
  Builders, distillation scripts, Modal train/eval entrypoints, and the numeric data rebalancer.

- `src/rl_secdef/`
  The key package modules required to build and benchmark the cleaned experiments.

- `tests/`
  Focused tests for the numeric-triage pipeline.

## Important Context

This is not a full mirror of the original research workspace.

It is an experiment bundle intended to preserve:

- methodology
- benchmark outputs
- relevant code paths
- main caveats

The broad conclusion from these experiments is:

- the original broad benchmark claims did not survive cleanup
- narrow, structured, real-world distillation worked substantially better than synthetic-only fine-tuning

## Reproduction Notes

The most relevant benchmark outputs are:

- `benchmarks/qwen7b_primevul_numeric_base_eval.json`
- `benchmarks/qwen7b_juliet_numeric_stage1_eval.json`
- `benchmarks/gpt52_primevul_numeric_eval.json`
- `benchmarks/qwen7b_primevul_numeric_distilled_eval.json`
- `benchmarks/qwen7b_juliet_primevul_numeric_distilled_eval.json`

The core note explains which metrics matter and why `balanced binary accuracy` is the right headline metric for the numeric task.
