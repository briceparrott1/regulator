"""Shared data models for the Regulator pipeline.

Pydantic v2 models spanning three layers: an immutable parse layer produced by
document parsing, a per-run verdict layer that references IDs only, and an output
layer assembled into the final report. These are thin first-iteration models;
nested structures stay plain dicts rather than dedicated sub-models.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Constrained-string aliases used across the schema. The doc_kind taxonomy
# mirrors the captain's truth set:
#   regulation           — the codified text of a binding law/rule (statute,
#                          CFR part, state regulation)
#   compliance_directive — an agency's enforcement/compliance instruction or
#                          guidance about a regulation (e.g. an archived OSHA
#                          CPL directive), NOT the regulation's own text
#   industry_standard    — a voluntary consensus standard from an SDO/trade body
#                          (ASME, API, ASTM, ANSI, NFPA, IEEE, ISO, ...)
#   national_standard    — a single nation's adoption/transposition of an
#                          international standard (e.g. SASO adopting IEC)
#   reference_package    — a compiled reference/informational collection, not an
#                          operative rule (e.g. a state SIP reference package)
#   non_regulatory       — neither regulation nor standard (lecture notes,
#                          tutorials, descriptive commentary)
DocKind = Literal[
    "regulation",
    "compliance_directive",
    "industry_standard",
    "national_standard",
    "reference_package",
    "non_regulatory",
]
CoverageStatus = Literal["covered", "partial", "absent", "contradicted"]
FindingType = Literal["gap", "partial", "contradiction"]
ValidationStatus = Literal["passed", "quarantined"]

# Document-level applicability vocabulary. The three verdicts are DESCRIPTIVE —
# they record what the document's own scope says about the SOP, not what the
# pipeline should do about it (that decision is
# :func:`regulator.applicability.should_audit`):
#   applicable     — the scope covers the SOP on facts the SOP itself states
#   not_applicable — a scope gate fails on facts already known (wrong activity,
#                    wrong jurisdiction, wrong article), or the document states
#                    no scope at all
#   conditional    — the subject matter matches, but applicability turns on one
#                    or more facts the SOP does not state (a threshold quantity,
#                    a source classification, an equipment type, an adoption).
#                    Those unknowns are listed in ``missing_facts``.
Verdict = Literal["applicable", "not_applicable", "conditional"]
# How sure the reader is of the verdict, on the evidence it was given.
Confidence = Literal["high", "medium", "low"]


# ── Parse layer (immutable) ─────────────────────────────
class RegProfile(BaseModel):
    """Structured facets describing what a regulatory document governs."""

    model_config = ConfigDict(frozen=True)

    doc_id: str  # slug of the source document this profile describes
    # Where the document applies, using the truth vocabulary: "US-federal",
    # a US state as "US-<STATE>" (e.g. "US-PA"), country-wide "US", other
    # country names ("Saudi Arabia"), "adopted-by-AHJ" for standards binding
    # only where an authority adopts them, phrases like
    # "industry-adopted (jurisdiction-dependent)" or
    # "international (WTO TBT-aligned)", and empty when non-regulatory.
    jurisdiction: list[str]
    doc_kind: DocKind
    activities: list[str]
    substances: list[str]
    equipment: list[str]
    industries: list[str]
    addressee_types: list[str]  # employer, facility, procedure, manufacturer...


class RegulatoryNode(BaseModel):
    """A node in a regulatory document's parsed hierarchy.

    Documents are parsed into a tree; each node sits at some level of that tree.
    There are two kinds. Internal nodes group their descendants and carry an
    intro paragraph or header in ``body``. Leaf nodes carry a single clause in
    ``body`` and are where the normative facets (``is_normative``, ``addressee``)
    are meaningful; on internal nodes those fields keep their defaults.
    """

    model_config = ConfigDict(frozen=True)

    node_id: str  # citation path: "1910.119(f)(1)(i)"
    parent_id: str | None
    section_lineage: list[str]  # node_ids of this node's ancestors, root first
    body: str  # leaf: a clause; internal: an intro paragraph or header
    is_leaf: bool  # two node types: internal and leaf
    provenance: dict[str, Any]  # {page, span}
    is_normative: bool = False  # only meaningful on leaves
    addressee: str | None = None  # only meaningful on leaves


class RegulatoryDocument(BaseModel):
    """A parsed regulatory document with its profile and extracted nodes."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    source_path: str
    file_hash: str
    title: str
    framework: str  # e.g. "OSHA 29 CFR"
    edition: str  # e.g. "2022"
    parse_accounting: dict[str, Any]  # {pages_total, pages_parsed, warnings[]}
    profile: RegProfile
    # Flat list; the tree shape is encoded via each node's parent_id and
    # section_lineage rather than nesting.
    nodes: list[RegulatoryNode]


class SopProfile(BaseModel):
    """Structured facets describing an SOP; mirror-shaped vs RegProfile."""

    model_config = ConfigDict(frozen=True)

    # Same jurisdiction vocabulary as RegProfile (e.g. "US", "US-PA").
    jurisdiction: list[str]
    industry: str
    activities: list[str]
    substances: list[str]
    equipment: list[str]
    internal_references: list[str]


class SopNode(BaseModel):
    """A node in an SOP's parsed hierarchy; mirror of :class:`RegulatoryNode`.

    SOPs are parsed into a tree; each node sits at some level of that tree.
    There are two kinds. Internal nodes group their descendants and carry a
    heading or intro paragraph in ``body``. Leaf nodes carry a single
    procedural step or clause in ``body``.
    """

    model_config = ConfigDict(frozen=True)

    node_id: str  # "S-047"; the parser will assign ids
    parent_id: str | None
    section_lineage: list[str]  # node_ids of this node's ancestors, root first
    body: str  # leaf: a step/clause; internal: a heading or intro paragraph
    is_leaf: bool  # two node types: internal and leaf
    order: int  # document order (SOP-specific)


class OperatingProcedure(BaseModel):
    """A parsed Standard Operating Procedure with its profile and nodes."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    source_path: str
    file_hash: str
    title: str
    parse_accounting: dict[str, Any]  # {pages_total, pages_parsed, warnings[]}
    profile: SopProfile
    # Flat list; the tree shape is encoded via each node's parent_id and
    # section_lineage rather than nesting. A synthetic doc-root node anchors
    # the tree, same convention as RegulatoryDocument.
    nodes: list[SopNode]


# ── Verdict layer (per-run, references IDs only) ───────
class DocApplicabilityVerdict(BaseModel):
    """What one regulatory document's scope says about the run's SOP.

    Deliberately DESCRIPTIVE, not decisive: it records an observation (the
    scope sentence that drives the call, why it lands where it does, and which
    facts are still unknown) without saying whether the pipeline should keep
    auditing the document. That call is made by
    :func:`regulator.applicability.should_audit`, so the policy can change
    without the record changing meaning.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    sop_id: str
    verdict: Verdict
    # The scope/applicability sentence driving the verdict, copied VERBATIM from
    # the source document so a reviewer can find it. ``None`` when the document
    # states no scope to quote (e.g. non-regulatory material).
    trigger_quote: str | None = None
    reasons: list[str] = Field(default_factory=list)
    # Facts the SOP does not state that GATE applicability (threshold
    # quantities, source classifications, equipment types). Populated mainly on
    # a "conditional" verdict; empty when the verdict rests on known facts.
    missing_facts: list[str] = Field(default_factory=list)
    confidence: Confidence


class NodeApplicabilityVerdict(BaseModel):
    """Whether a single regulatory node binds and applies to the SOP."""

    node_id: str
    binds_context: bool  # jurisdiction/activity/substance match
    sop_is_instrument: bool  # is an SOP what satisfies this clause?
    applicable: bool  # AND of above
    reasoning: str


class CoverageVerdict(BaseModel):
    """How well the SOP covers a single regulatory node."""

    node_id: str
    status: CoverageStatus
    sop_node_ids: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    reasoning: str


# ── Output layer ────────────────────────────────────────
class Finding(BaseModel):
    """A reportable gap, partial coverage, or contradiction."""

    finding_id: str
    type: FindingType
    reg_ref: dict[str, Any]  # {node_id, quote}
    sop_anchors: list[dict[str, Any]] = Field(
        default_factory=list
    )  # [{node_id, quote}]; empty for pure gaps
    explanation: str
    suggested_adjustment: str
    confidence: float
    validation: ValidationStatus  # citation checker result


class Report(BaseModel):
    """The final compliance report, with run metadata and findings."""

    run_meta: dict[str, Any]  # {timestamp, models, config, input_hashes}
    transparency: dict[str, Any]  # docs_excluded, parse_accounting_summary, ...
    findings: list[Finding] = Field(default_factory=list)  # grouped by SOP section
