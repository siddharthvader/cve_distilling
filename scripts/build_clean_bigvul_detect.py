"""Build a cleaned non-eval BigVul detect corpus for real-world calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_secdef.data import build_bigvul_detect_jsonl, load_response_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/bigvul_clean_detect.jsonl"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/eval_bigvul_detect_blind.jsonl"))
    parser.add_argument("--split", default="train", help="BigVul HF split")
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=120, help="Positive examples before optional negatives")
    parser.add_argument("--max-per-cwe", type=int, default=12)
    parser.add_argument("--include-negative", action="store_true")
    parser.add_argument("--responses", type=Path, default=None, help="Optional JSONL with response overrides")
    args = parser.parse_args()

    stats = build_bigvul_detect_jsonl(
        output_path=args.output,
        eval_file=args.eval_file,
        split=args.split,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
        max_samples=args.max_samples,
        max_per_cwe=args.max_per_cwe,
        include_negative=args.include_negative,
        response_overrides=load_response_overrides(args.responses),
    )

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
