"""Standalone evaluation harness for the Regulator pipeline.

Run with ``python -m regulator.evaluation``. This is deliberately separate from
``main.py`` and is not invoked by it.

Stub implementation. Will (once implemented) load the human-readable YAML test
cases from ``tests/cases/`` and run each against the pipeline, reporting
pass/fail results.
"""

from __future__ import annotations

from pathlib import Path

CASES_DIR = Path(__file__).resolve().parent.parent / "tests" / "cases"


def main() -> None:
    """Load and run the YAML test cases in ``tests/cases/``.

    Will (once implemented) parse each YAML case, execute the pipeline against
    it, and report pass/fail results. For now it reports that no runnable test
    cases exist yet.
    """
    case_files = sorted(CASES_DIR.glob("*.yaml")) + sorted(CASES_DIR.glob("*.yml"))
    if not case_files:
        print(f"No test cases found in {CASES_DIR} (stub).")
        return

    print(f"Found {len(case_files)} test case file(s) in {CASES_DIR} (stub; not run).")


if __name__ == "__main__":
    main()
