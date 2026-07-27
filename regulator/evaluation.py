"""Standalone evaluation harness for the Regulator pipeline.

Run with ``python -m regulator.evaluation``. It loads the human-readable test
cases in ``tests/cases/``, runs the real pipeline against each, and reports
pass/fail. Exit code is 1 if any case fails.

Case formats, each recognized by its shape rather than its filename:

* ``.yaml``/``.yml`` with a ``task`` key — a regulatory parse case;
* ``.yaml``/``.yml`` holding a bare list — the SOP's full parsed tree, compared
  exactly;
* ``.json`` with ``sop_profile``/``reg_profiles`` — the profile-truth set;
* ``.jsonl`` whose records carry ``sop_id``/``doc_id``/``label`` — the
  document-level applicability truth set.

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

import json
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from regulator.applicability import (
    document_text,
    judge_document_applicability,
    quote_is_verbatim,
)
from regulator.llm import StructureLLM
from regulator.models import (
    DocApplicabilityVerdict,
    OperatingProcedure,
    RegulatoryDocument,
    RegulatoryNode,
)
from regulator.parse_cli import _available_regulations
from regulator.parsing import (
    _slug,
    parse_operating_procedure,
    parse_regulatory_document,
)
from regulator.profiles import (
    _profile_llm,
    extract_regulatory_profile,
    extract_sop_profile,
    load_profile,
)

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


# ── Profile-truth case (tests/cases/profiles_truth.json) ───────────────
# The truth file keys its 10 regulatory documents by ids that differ from our
# filename slugs for two of them; map truth id -> our slug here. The truth file
# is frozen, so the mapping lives on our side.
_DOC_ID_ALIASES = {
    "reg-pa-dep-air-quality": "reg-pa-dep-of-air-quality",
    "reg-saso-iec-60051-1-2020": "reg-saso-iec-60051-1-2020-e",
}

# Judge model for semantic field equivalence (reuses llm.py).
JUDGE_MODEL = "claude-sonnet-5"

# Reg fields judged semantically. doc_kind is an exact match (handled
# separately); _notes / _meta are never compared.
_REG_JUDGED_FIELDS = (
    "jurisdiction",
    "activities",
    "substances",
    "equipment",
    "industries",
    "addressee_types",
)
# SOP fields judged semantically. internal_references is a deterministic
# set-equality check (handled separately).
_SOP_JUDGED_FIELDS = (
    "jurisdiction",
    "industry",
    "activities",
    "substances",
    "equipment",
)

_PROFILE_JUDGE_SYSTEM_PROMPT = """\
You compare two lists describing the same facet of one document: a ground-truth
list and a generated list. Decide whether they are SEMANTICALLY EQUIVALENT.

COVERAGE (be strict here): every concept in the truth list must be covered by
some item in the generated list. An item covers a truth concept when it is the
same concept, a synonym or paraphrase, the same concept at a different
granularity, or a broader/narrower term that clearly stands for it. A truth
concept is MISSING only when no generated item reasonably expresses it.

ADDITIONS (be tolerant here): the generated list may hold extra items not in the
truth list. An extra item is SPURIOUS only if it is factually WRONG about this
document or clearly OFF-TOPIC / irrelevant to it. Extra items that are true and
on-topic — finer detail, related sub-items, reasonable audiences, or items the
truth list simply did not bother to enumerate — are ACCEPTABLE and must NOT be
reported as spurious.

Equivalent = (nothing missing) AND (nothing spurious). Judge by meaning, not
wording. When it is genuinely arguable whether some generated item expresses a
truth concept, treat the concept as COVERED.

Output contract — follow exactly:
- "missing" holds ONLY strings copied VERBATIM from the truth list, and only for
  concepts no generated item expresses. Never invent entries, never annotate
  them, never add commentary or parentheticals. If you would note that an item is
  present, it does NOT belong in "missing".
- "spurious" holds ONLY strings copied verbatim from the generated list.
- Set "equivalent" to true if and only if both arrays are empty.

Return ONLY this JSON object:
{"equivalent": true or false,
 "missing": [verbatim truth items no generated item expresses],
 "spurious": [verbatim generated items that are wrong about, or off-topic to, this document],
 "note": "one short sentence"}
"""


def _print_field(field: str, passed: bool, detail: str) -> None:
    """Print one per-field PASS/FAIL row of the profile table."""
    status = "PASS" if passed else "FAIL"
    line = f"  {field:<24} {status}"
    if not passed and detail:
        line += f"  {detail}"
    print(line)


def _judge_field(
    judge: StructureLLM,
    field: str,
    truth: list[Any],
    generated: list[Any],
    system_prompt: str = _PROFILE_JUDGE_SYSTEM_PROMPT,
) -> tuple[bool, str]:
    """Judge semantic equivalence of one field; return (passed, detail).

    Trivial cases short-circuit without an LLM call: both empty passes, and a
    non-empty truth against an empty generated list fails outright.
    ``system_prompt`` selects the rubric — profile facets by default, other
    rubrics (e.g. applicability rationales) pass their own.
    """
    truth_list = [str(x) for x in truth]
    gen_list = [str(x) for x in generated]
    if not truth_list and not gen_list:
        return True, ""
    if truth_list and not gen_list:
        return False, f"missing={truth_list} spurious=[]"

    user = (
        f"Facet: {field}\n"
        f"Truth list: {json.dumps(truth_list, ensure_ascii=False)}\n"
        f"Generated list: {json.dumps(gen_list, ensure_ascii=False)}"
    )
    try:
        result = judge.propose_json(
            system_prompt,
            user,
            max_tokens=1500,
            thinking={"type": "disabled"},
        )
    except Exception as exc:  # noqa: BLE001 — a judge failure fails the field
        return False, f"judge error: {type(exc).__name__}"

    equivalent = bool(result.get("equivalent"))
    missing = result.get("missing") or []
    spurious = result.get("spurious") or []
    return equivalent, f"missing={missing} spurious={spurious}"


def _norm_ref(value: Any) -> str:
    """Light normalization for internal-reference set equality (case/space)."""
    return " ".join(str(value).split()).lower()


def _check_internal_references(
    truth: list[Any], generated: list[Any]
) -> tuple[bool, str]:
    """Deterministic set equality of SOP internal references after normalizing."""
    truth_set = {_norm_ref(x) for x in truth}
    gen_set = {_norm_ref(x) for x in generated}
    if truth_set == gen_set:
        return True, ""
    missing = sorted(truth_set - gen_set)
    spurious = sorted(gen_set - truth_set)
    return False, f"missing={missing} spurious={spurious}"


def _run_sop_profile(
    judge: StructureLLM, sop_truth: dict[str, Any], failures: list[str]
) -> None:
    """Run SOP profile extraction live and compare it to the truth."""
    print("PROFILE sop (original)")
    profile = extract_sop_profile().profile
    for field in _SOP_JUDGED_FIELDS:
        truth_val = sop_truth.get(field, [])
        gen_val = getattr(profile, field)
        if field == "industry":  # a single string on both sides
            truth_list = [truth_val] if truth_val else []
            gen_list = [gen_val] if gen_val else []
        else:
            truth_list = list(truth_val)
            gen_list = list(gen_val)
        passed, detail = _judge_field(judge, field, truth_list, gen_list)
        _print_field(field, passed, detail)
        if not passed:
            failures.append(f"sop.{field}: {detail}")

    passed, detail = _check_internal_references(
        sop_truth.get("internal_references", []), profile.internal_references
    )
    _print_field("internal_references", passed, detail)
    if not passed:
        failures.append(f"sop.internal_references: {detail}")


def _run_reg_profile(
    judge: StructureLLM,
    reg_truth: dict[str, Any],
    slug_to_path: dict[str, Path],
    failures: list[str],
) -> None:
    """Run one regulatory-document profile extraction live and compare to truth."""
    truth_id = reg_truth["doc_id"]
    slug = _DOC_ID_ALIASES.get(truth_id, truth_id)
    print(f"PROFILE {truth_id}")
    path = slug_to_path.get(slug)
    if path is None:
        msg = f"no source PDF for slug {slug!r} (truth doc_id {truth_id!r})"
        print(f"  (skipped: {msg})")
        failures.append(msg)
        return

    profile = extract_regulatory_profile(path).profile

    dk_pass = profile.doc_kind == reg_truth["doc_kind"]
    dk_detail = (
        ""
        if dk_pass
        else f"expected {reg_truth['doc_kind']!r}, got {profile.doc_kind!r}"
    )
    _print_field("doc_kind", dk_pass, dk_detail)
    if not dk_pass:
        failures.append(f"{truth_id}.doc_kind: {dk_detail}")

    for field in _REG_JUDGED_FIELDS:
        passed, detail = _judge_field(
            judge, field, reg_truth.get(field, []), list(getattr(profile, field))
        )
        _print_field(field, passed, detail)
        if not passed:
            failures.append(f"{truth_id}.{field}: {detail}")


def _run_profile_truth_case(truth: dict[str, Any]) -> list[str]:
    """Run the profile-truth case: extract all 11 profiles live and compare.

    Prints a per-document per-field PASS/FAIL table and returns a list of
    failure summaries. The case passes only when every field of every profile
    passes. doc_kind is exact, internal_references is set-equality, and all other
    fields are judged for semantic equivalence by :data:`JUDGE_MODEL`.
    """
    failures: list[str] = []
    judge = StructureLLM(model=JUDGE_MODEL)

    sop_truth = truth.get("sop_profile")
    if sop_truth is not None:
        _run_sop_profile(judge, sop_truth, failures)

    slug_to_path = {_slug(p.stem): p for p in _available_regulations()}
    for reg_truth in truth.get("reg_profiles", []):
        _run_reg_profile(judge, reg_truth, slug_to_path, failures)

    return failures


# ── Applicability-truth case (tests/cases/applicability_truth.jsonl) ───
# The truth file names the SOP "sop-original"; our slug for data/sop/original.docx
# is "original". Same frozen-truth situation as _DOC_ID_ALIASES.
_SOP_ID_ALIASES = {"sop-original": "original"}

# Which truth fields gate the case. `label` is the verdict itself and is
# compared EXACTLY; the quote and the two rationale lists are judged. Everything
# else in a truth record is bookkeeping (id, labeled_by, verified, date,
# judgment_call, expected_resolution) or advisory (confidence) and is reported
# without gating.
_APPLICABILITY_RATIONALE_FIELDS = ("reasons", "missing_facts")

_APPLICABILITY_JUDGE_SYSTEM_PROMPT = """\
You compare two lists of short statements explaining why one regulatory document
does or does not reach one Standard Operating Procedure: a ground-truth list
written by a human reviewer and a generated list. Decide whether they are
SEMANTICALLY EQUIVALENT.

COVERAGE (be strict): every substantive claim in the truth list must be made by
some statement in the generated list. A generated statement covers a truth claim
when it asserts the same thing — same scope gate, same fact about the procedure,
same reason the document is or is not an operative rule here — in different
words or at a different level of detail. A truth claim is MISSING only when no
generated statement asserts it.

IGNORE truth entries that comment on the LABELLING PROCESS rather than on the
documents: notes about why the reviewer chose one label over another, remarks
about what the row is testing, and other meta-commentary. These describe the
ground-truth exercise, not the documents, so a generated answer cannot be
expected to reproduce them and they are never "missing".

ADDITIONS (be tolerant): extra generated statements are SPURIOUS only when they
are factually WRONG about these documents or clearly irrelevant to whether the
document reaches this procedure. Additional true, on-point observations are
ACCEPTABLE.

Equivalent = (nothing missing) AND (nothing spurious). Judge by meaning, not
wording. When it is genuinely arguable whether a generated statement makes a
truth claim, treat the claim as COVERED.

Return ONLY this JSON object:
{"equivalent": true or false,
 "missing": [verbatim truth statements no generated statement makes],
 "spurious": [verbatim generated statements that are wrong or irrelevant],
 "note": "one short sentence"}
"""

_QUOTE_JUDGE_SYSTEM_PROMPT = """\
You compare two quotations taken from the SAME regulatory document: a
ground-truth quote a human picked as the scope/applicability statement that
drives whether the document reaches a given procedure, and a generated quote.
Decide whether they point at the SAME statement of reach.

They AGREE when they are the same sentence or heading, overlapping parts of one
passage, or two passages stating the same scope gate. The truth quote may be
abridged (fragments joined with "..."), lightly trimmed, or start mid-sentence;
none of that is disagreement.

They DISAGREE when the generated quote points at a different gate, at a
requirement rather than a statement of reach, or at unrelated material — or when
one side quotes nothing while the other quotes a real statement of reach.

Return ONLY this JSON object:
{"agrees": true or false, "note": "one short sentence"}
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a ``.jsonl`` case file into records, ignoring blank lines."""
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _is_applicability_record(record: dict[str, Any]) -> bool:
    """True when a JSONL record is an applicability-truth row (by shape)."""
    return all(key in record for key in ("sop_id", "doc_id", "label"))


def _print_info(field: str, detail: str) -> None:
    """Print one non-gating (informational) row of the applicability table."""
    print(f"  {field:<24} INFO  {detail}")


def _judge_quote(
    judge: StructureLLM, truth_quote: str, generated_quote: str
) -> tuple[bool, str]:
    """LLM-judge whether two trigger quotes point at the same scope statement."""
    user = (
        f"Truth quote: {json.dumps(truth_quote, ensure_ascii=False)}\n"
        f"Generated quote: {json.dumps(generated_quote, ensure_ascii=False)}"
    )
    try:
        result = judge.propose_json(
            _QUOTE_JUDGE_SYSTEM_PROMPT,
            user,
            max_tokens=800,
            thinking={"type": "disabled"},
        )
    except Exception as exc:  # noqa: BLE001 — a judge failure fails the field
        return False, f"judge error: {type(exc).__name__}"
    agrees = bool(result.get("agrees"))
    return agrees, "" if agrees else f"note={result.get('note')!r}"


def _check_trigger_quote(
    judge: StructureLLM,
    truth_quote: str | None,
    verdict: DocApplicabilityVerdict,
    source_text: str,
) -> list[tuple[str, bool, str]]:
    """Check the generated trigger quote; return (field, passed, detail) rows.

    Two independent checks. VERBATIM: the quote must be findable in the source
    document (whitespace-normalized), because a quote a reviewer cannot locate is
    worthless whatever it says. SEMANTIC: it must point at the same statement of
    reach the truth quote does.

    A truth quote of ``null`` means the document states no scope to quote, so the
    expected generated quote is also empty; both checks then hinge on that.
    """
    generated = verdict.trigger_quote

    if truth_quote is None:
        passed = generated is None
        detail = "" if passed else f"expected no quote, got {generated!r}"
        return [
            ("trigger_quote (verbatim)", passed, detail),
            ("trigger_quote (semantic)", passed, detail),
        ]

    if generated is None:
        detail = f"no quote produced; truth quotes {truth_quote[:60]!r}"
        return [
            ("trigger_quote (verbatim)", False, detail),
            ("trigger_quote (semantic)", False, detail),
        ]

    verbatim = quote_is_verbatim(generated, source_text)
    verbatim_detail = "" if verbatim else f"not found in source: {generated[:80]!r}"
    agrees, agree_detail = _judge_quote(judge, truth_quote, generated)
    return [
        ("trigger_quote (verbatim)", verbatim, verbatim_detail),
        ("trigger_quote (semantic)", agrees, agree_detail),
    ]


def _load_profile_document(doc_id: str) -> RegulatoryDocument | OperatingProcedure:
    """Load one profile-only artifact, or raise a ValueError explaining why not.

    Collapses "not generated yet" and "predates the current schema" into one
    exception type, since the case reports both the same way.
    """
    try:
        return load_profile(doc_id)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def _run_applicability_pair(
    judge: StructureLLM,
    matcher_llm: StructureLLM,
    record: dict[str, Any],
    sop: OperatingProcedure,
    failures: list[str],
) -> None:
    """Run the matcher for one truth row and print its per-field table."""
    truth_id = record["doc_id"]
    slug = _DOC_ID_ALIASES.get(truth_id, truth_id)
    print(f"APPLICABILITY {record.get('id', truth_id)} {truth_id}")

    try:
        doc = _load_profile_document(slug)
    except ValueError as exc:
        print(f"  (skipped: {exc})")
        failures.append(f"{truth_id}: {exc}")
        return
    if not isinstance(doc, RegulatoryDocument):
        msg = f"profile artifact for {slug!r} is not a regulatory document"
        print(f"  (skipped: {msg})")
        failures.append(f"{truth_id}: {msg}")
        return

    verdict = judge_document_applicability(sop, doc, llm=matcher_llm)

    label_pass = verdict.verdict == record["label"]
    label_detail = (
        "" if label_pass else f"expected {record['label']!r}, got {verdict.verdict!r}"
    )
    _print_field("label", label_pass, label_detail)
    if not label_pass:
        failures.append(f"{truth_id}.label: {label_detail}")

    for field, passed, detail in _check_trigger_quote(
        judge, record.get("trigger_quote"), verdict, document_text(doc)
    ):
        _print_field(field, passed, detail)
        if not passed:
            failures.append(f"{truth_id}.{field}: {detail}")

    for field in _APPLICABILITY_RATIONALE_FIELDS:
        passed, detail = _judge_field(
            judge,
            field,
            record.get(field, []) or [],
            list(getattr(verdict, field)),
            system_prompt=_APPLICABILITY_JUDGE_SYSTEM_PROMPT,
        )
        _print_field(field, passed, detail)
        if not passed:
            failures.append(f"{truth_id}.{field}: {detail}")

    # Not gated: reported so a systematic over/under-confidence shows up.
    truth_confidence = record.get("confidence")
    if truth_confidence != verdict.confidence:
        _print_info(
            "confidence",
            f"truth={truth_confidence!r} got={verdict.confidence!r} (not gated)",
        )


def _run_applicability_truth_case(records: list[dict[str, Any]]) -> list[str]:
    """Run the applicability-truth case: judge every (SOP, document) pair live.

    Profiles come from the ``data/profiles/`` artifacts (no re-parsing), and the
    scope excerpts the matcher quotes from are read straight off the source PDFs.
    Prints a per-pair per-field PASS/FAIL table and returns failure summaries;
    the case passes only when every gated field of every pair passes.
    """
    failures: list[str] = []
    judge = StructureLLM(model=JUDGE_MODEL)
    matcher_llm = _profile_llm()

    sops: dict[str, OperatingProcedure] = {}
    for record in records:
        truth_sop_id = record.get("sop_id", "")
        sop_slug = _SOP_ID_ALIASES.get(truth_sop_id, truth_sop_id)
        sop = sops.get(sop_slug)
        if sop is None:
            try:
                loaded = _load_profile_document(sop_slug)
            except ValueError as exc:
                return [f"could not load SOP profile {sop_slug!r}: {exc}"]
            if not isinstance(loaded, OperatingProcedure):
                return [f"profile artifact {sop_slug!r} is not an operating procedure"]
            sop = loaded
            sops[sop_slug] = sop
        _run_applicability_pair(judge, matcher_llm, record, sop, failures)
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

    case_files = (
        sorted(CASES_DIR.glob("*.yaml"))
        + sorted(CASES_DIR.glob("*.yml"))
        + sorted(CASES_DIR.glob("*.json"))
        + sorted(CASES_DIR.glob("*.jsonl"))
    )
    if not case_files:
        print(f"No test cases found in {CASES_DIR}.")
        return 0

    passed = 0
    failed = 0
    skipped = 0

    for case_file in case_files:
        name = case_file.name

        if case_file.suffix == ".json":
            # A JSON case is the profile-truth set: a dict carrying a
            # "sop_profile" and/or "reg_profiles". Anything else is skipped.
            data = json.loads(case_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and (
                "reg_profiles" in data or "sop_profile" in data
            ):
                failures = _run_profile_truth_case(data)
            else:
                print(f"SKIP {name}: unrecognized JSON case")
                skipped += 1
                continue
        elif case_file.suffix == ".jsonl":
            # One JSON record per line. A record carrying sop_id / doc_id /
            # label is an applicability-truth row; anything else is skipped.
            records = _read_jsonl(case_file)
            if records and _is_applicability_record(records[0]):
                failures = _run_applicability_truth_case(records)
            else:
                print(f"SKIP {name}: unrecognized JSONL case")
                skipped += 1
                continue
        else:
            case = yaml.safe_load(case_file.read_text(encoding="utf-8"))
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
