"""Build PrimeVul integer overflow/underflow triage train and eval files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_secdef.data import write_primevul_numeric_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-output", type=Path, default=Path("data/primevul_numeric_triage_train.jsonl"))
    parser.add_argument("--eval-output", type=Path, default=Path("data/primevul_numeric_triage_eval.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-neg-train", type=int, default=2)
    parser.add_argument("--distractor-train", type=int, default=1)
    parser.add_argument("--hard-neg-eval", type=int, default=3)
    parser.add_argument("--distractor-eval", type=int, default=3)
    args = parser.parse_args()

    stats = write_primevul_numeric_files(
        train_output=args.train_output,
        eval_output=args.eval_output,
        seed=args.seed,
        hard_negative_ratio_train=args.hard_neg_train,
        distractor_ratio_train=args.distractor_train,
        hard_negative_ratio_eval=args.hard_neg_eval,
        distractor_ratio_eval=args.distractor_eval,
    )

    manifest = args.train_output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
