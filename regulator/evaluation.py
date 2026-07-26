"""Standalone evaluation harness for the Regulator pipeline.

Run with ``python -m regulator.evaluation``. It loads the human-readable YAML
test cases in ``tests/cases/``, runs the real pipeline against each, and reports
pass/fail. Exit code is 1 if any case fails.

Normalization semantics (the truth files assume no synthetic document root; our
parser adds one, so these checks account for it):

* ``node_count``: raw ``len(doc.nodes)`` vs an int (exact) or ``{min, max}``
  (inclusive range).
* ``contains_nodes``: an entry matches a node iff EVERY provided field matches —
  ``node_id`` / ``is_leaf`` exact; ``parent_id`` truth ``null`` matches a parsed
  parent that is the synthetic root OR ``None``, otherwise exact;
  ``body_contains`` is a case-insensitive substring of the body;
  ``section_lineage`` compares the truth list against the parsed lineage AFTER
  stripping the leading synthetic-root id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from regulator.models import OperatingProcedure, RegulatoryDocument, RegulatoryNode
from regulator.parsing import parse_operating_procedure, parse_regulatory_document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = PROJECT_ROOT / "tests" / "cases"

# The repo carries exactly one SOP; a bare-list truth case is always its full
# parsed tree, so the SOP source path is a constant rather than a case field.
SOP_PATH = PROJECT_ROOT / "data" / "sop" / "original.docx"


def _find_root_id(doc: RegulatoryDocument) -> str | None:
    """The synthetic root is the node whose parent is the document id itself."""
    for node in doc.nodes:
        if node.parent_id == doc.doc_id:
            return node.node_id
    return None


def _stripped_lineage(node: RegulatoryNode, root_id: str | None) -> list[str]:
    lineage = list(node.section_lineage)
    if root_id is not None and lineage and lineage[0] == root_id:
        return lineage[1:]
    return lineage


def _node_matches(
    entry: dict[str, Any], node: RegulatoryNode, root_id: str | None
) -> bool:
    for key, expected in entry.items():
        if key == "node_id":
            if node.node_id != expected:
                return False
        elif key == "is_leaf":
            if node.is_leaf != expected:
                return False
        elif key == "parent_id":
            if expected is None:
                if node.parent_id not in (None, root_id):
                    return False
            elif node.parent_id != expected:
                return False
        elif key == "body_contains":
            if str(expected).lower() not in node.body.lower():
                return False
        elif key == "section_lineage" and _stripped_lineage(node, root_id) != list(
            expected
        ):
            return False
        # Unknown keys are ignored.
    return True


def _check_node_count(expected: Any, actual: int, failures: list[str]) -> None:
    if isinstance(expected, dict):
        low, high = expected.get("min"), expected.get("max")
        if (low is not None and actual < low) or (high is not None and actual > high):
            failures.append(f"node_count {actual} outside range [{low}, {high}]")
    elif actual != expected:
        failures.append(f"node_count expected {expected}, got {actual}")


def _candidate_report(
    entry: dict[str, Any], doc: RegulatoryDocument, root_id: str | None
) -> list[str]:
    """Describe the parsed nodes closest to an unmatched contains_nodes entry."""
    target = entry.get("node_id")
    candidates = [n for n in doc.nodes if n.node_id == target]
    if not candidates:
        return [f"      (no parsed node has node_id {target!r})"]
    lines: list[str] = []
    for node in candidates:
        snippet = node.body[:90].replace("\n", " ")
        lines.append(
            f"      candidate node_id={node.node_id!r} "
            f"parent_id={node.parent_id!r} is_leaf={node.is_leaf} "
            f"lineage={_stripped_lineage(node, root_id)} "
            f"body[:90]={snippet!r}"
        )
    return lines


def _check_contains_nodes(
    entries: list[dict[str, Any]],
    doc: RegulatoryDocument,
    root_id: str | None,
    failures: list[str],
) -> None:
    for entry in entries:
        if any(_node_matches(entry, node, root_id) for node in doc.nodes):
            continue
        failures.append(f"contains_nodes entry unmatched: {entry}")
        failures.extend(_candidate_report(entry, doc, root_id))


def _body_diff(expected: str, got: str, width: int = 80) -> str:
    """Render expected vs got bodies, windowed around the first difference.

    Whitespace-only diffs (double spaces, trailing tabs) survive the repr, so a
    reviewer can see exactly where the strings part ways.
    """
    limit = min(len(expected), len(got))
    first = next((i for i in range(limit) if expected[i] != got[i]), limit)
    start = max(0, first - width // 2)
    exp_window = expected[start : start + width]
    got_window = got[start : start + width]
    return (
        f"body differs at index {first}\n"
        f"        expected {exp_window!r}\n"
        f"        got      {got_window!r}"
    )


# Fields compared verbatim on every SOP truth node (body handled separately so
# whitespace diffs get a windowed report).
_SOP_SCALAR_FIELDS = ("parent_id", "is_leaf", "order")


def _compare_sop_node(entry: dict[str, Any], node) -> list[str]:
    """Compare one truth entry against its parsed :class:`SopNode`."""
    problems: list[str] = []
    if "body" in entry and node.body != entry["body"]:
        problems.append(_body_diff(str(entry["body"]), node.body))
    if "section_lineage" in entry and list(node.section_lineage) != list(
        entry["section_lineage"]
    ):
        problems.append(
            f"section_lineage expected {list(entry['section_lineage'])!r}, "
            f"got {list(node.section_lineage)!r}"
        )
    for field in _SOP_SCALAR_FIELDS:
        if field in entry and getattr(node, field) != entry[field]:
            problems.append(
                f"{field} expected {entry[field]!r}, got {getattr(node, field)!r}"
            )
    return problems


def _run_sop_full_tree_case(truth: list[dict[str, Any]]) -> list[str]:
    """Compare the full parsed SOP tree against an enumerated truth list."""
    failures: list[str] = []
    doc: OperatingProcedure = parse_operating_procedure(SOP_PATH)

    if len(doc.nodes) != len(truth):
        failures.append(f"node_count expected {len(truth)}, got {len(doc.nodes)}")

    by_id = {node.node_id: node for node in doc.nodes}
    truth_ids = {entry["node_id"] for entry in truth}

    extra = sorted(set(by_id) - truth_ids)
    if extra:
        failures.append(f"extra node_ids not in truth: {extra}")

    for entry in truth:
        node = by_id.get(entry["node_id"])
        if node is None:
            failures.append(f"missing node_id {entry['node_id']!r}")
            continue
        for problem in _compare_sop_node(entry, node):
            failures.append(f"{entry['node_id']}: {problem}")
    return failures


def _run_parse_case(case: dict[str, Any]) -> list[str]:
    """Run a parse_regulatory_document case; return a list of failure strings."""
    failures: list[str] = []
    document_path = (PROJECT_ROOT / case["document"]).resolve()
    doc = parse_regulatory_document(document_path)
    root_id = _find_root_id(doc)

    expect = case.get("expect", {})
    if "node_count" in expect:
        _check_node_count(expect["node_count"], len(doc.nodes), failures)
    if "contains_nodes" in expect:
        _check_contains_nodes(expect["contains_nodes"], doc, root_id, failures)
    return failures


# Maps a case's ``task`` to its handler. Unknown tasks are reported as skips.
_HANDLERS = {"parse_regulatory_document": _run_parse_case}


def main() -> int:
    """Load and run the YAML test cases; return the process exit code."""
    load_dotenv()

    case_files = sorted(CASES_DIR.glob("*.yaml")) + sorted(CASES_DIR.glob("*.yml"))
    if not case_files:
        print(f"No test cases found in {CASES_DIR}.")
        return 0

    passed = 0
    failed = 0
    skipped = 0

    for case_file in case_files:
        case = yaml.safe_load(case_file.read_text(encoding="utf-8"))
        name = case_file.name

        # A bare list of node dicts (no task/document wrapper) is a full-tree
        # SOP truth case; anything else is a task-keyed regulatory case.
        if isinstance(case, list):
            failures = _run_sop_full_tree_case(case)
        else:
            task = case.get("task")
            handler = _HANDLERS.get(task)
            if handler is None:
                print(f"SKIP {name}: unknown task {task!r}")
                skipped += 1
                continue
            failures = handler(case)

        if failures:
            failed += 1
            print(f"FAIL {name}")
            for line in failures:
                print(f"  - {line}")
        else:
            passed += 1
            print(f"PASS {name}")

    print(
        f"\nSummary: {passed} passed, {failed} failed, "
        f"{skipped} skipped, {len(case_files)} total"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
