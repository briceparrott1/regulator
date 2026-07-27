"""Run only the profile-truth case (skips the 3 expensive parse cases)."""

import json
from pathlib import Path

from dotenv import load_dotenv

from regulator.evaluation import _run_profile_truth_case

load_dotenv("/Users/briceparrott/coding/projects/regulator/.env")
truth = json.loads(Path("tests/cases/profiles_truth.json").read_text())
failures = _run_profile_truth_case(truth)
print(f"\n==== FAILURES ({len(failures)}) ====")
for f in failures:
    print(" -", f)
print("\nCASE:", "PASS" if not failures else "FAIL")
