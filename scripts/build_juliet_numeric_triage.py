"""Build Juliet curriculum data for PrimeVul numeric triage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_secdef.data import write_juliet_numeric_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", type=Path, default=Path("data/tasks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/juliet_numeric_triage.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = write_juliet_numeric_file(tasks_file=args.tasks_file, output_path=args.output, seed=args.seed)
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
