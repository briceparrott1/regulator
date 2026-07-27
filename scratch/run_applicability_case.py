"""Run only the applicability-truth case (skips every other case)."""

import json
from pathlib import Path

from dotenv import load_dotenv

from regulator.evaluation import _run_applicability_truth_case

load_dotenv("/Users/briceparrott/coding/projects/regulator/.env")
records = [
    json.loads(line)
    for line in Path("tests/cases/applicability_truth.jsonl").read_text().splitlines()
    if line.strip()
]
failures = _run_applicability_truth_case(records)
print(f"\n==== FAILURES ({len(failures)}) ====")
for f in failures:
    print(" -", f)
print("\nCASE:", "PASS" if not failures else "FAIL")
