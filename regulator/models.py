"""Shared data models for the Regulator pipeline.

Pydantic v2 models spanning three layers: an immutable parse layer produced by
document parsing, a per-run verdict layer that references IDs only, and an output
layer assembled into the final report. These are thin first-iteration models;
nested structures stay plain dicts rather than dedicated sub-models.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Constrained-string aliases used across the schema.
DocKind = Literal["regulation", "standard", "permit"]
CoverageStatus = Literal["covered", "partial", "absent", "contradicted"]
FindingType = Literal["gap", "partial", "contradiction"]
ValidationStatus = Literal["passed", "quarantined"]


# ── Parse layer (immutable) ─────────────────────────────
class RegProfile(BaseModel):
    """Structured facets describing what a regulatory document governs."""

    model_config = ConfigDict(frozen=True)

    jurisdiction: list[str]  # federal / state / country
    doc_kind: DocKind
    activities: list[str]
    substances: list[str]
    equipment: list[str]
    industries: list[str]
    addressee_types: list[str]  # employer, facility, procedure, manufacturer...


class RegulatoryAtom(BaseModel):
    """A single atomic regulatory clause with citation and provenance."""

    model_config = ConfigDict(frozen=True)

    atom_id: str  # citation path: "1910.119(f)(1)(i)"
    parent_id: str | None
    section_lineage: list[str]
    text: str
    provenance: dict[str, Any]  # {page, span}
    is_normative: bool
    addressee: str


class RegulatoryDocument(BaseModel):
    """A parsed regulatory document with its profile and extracted atoms."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    source_path: str
    file_hash: str
    title: str
    framework: str  # e.g. "OSHA 29 CFR"
    edition: str  # e.g. "2022"
    parse_accounting: dict[str, Any]  # {pages_total, pages_parsed, warnings[]}
    profile: RegProfile
    atoms: list[RegulatoryAtom]


class SopProfile(BaseModel):
    """Structured facets describing an SOP; mirror-shaped vs RegProfile."""

    model_config = ConfigDict(frozen=True)

    jurisdiction: list[str]
    industry: str
    activities: list[str]
    substances: list[str]
    equipment: list[str]
    internal_references: list[str]


class SopAtom(BaseModel):
    """A single ordered passage extracted from an SOP."""

    model_config = ConfigDict(frozen=True)

    atom_id: str  # "S-047"
    section_lineage: list[str]
    order: int
    text: str


class OperatingProcedure(BaseModel):
    """A parsed Standard Operating Procedure with its profile and atoms."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    source_path: str
    file_hash: str
    title: str
    parse_accounting: dict[str, Any]  # {pages_total, pages_parsed, warnings[]}
    profile: SopProfile
    atoms: list[SopAtom]


# ── Verdict layer (per-run, references IDs only) ───────
class DocApplicabilityVerdict(BaseModel):
    """Whether a regulatory document applies to the run's SOP."""

    doc_id: str
    applicable: bool
    reasons: list[str] = Field(default_factory=list)
    confidence: float


class AtomApplicabilityVerdict(BaseModel):
    """Whether a single regulatory atom binds and applies to the SOP."""

    atom_id: str
    binds_context: bool  # jurisdiction/activity/substance match
    sop_is_instrument: bool  # is an SOP what satisfies this clause?
    applicable: bool  # AND of above
    reasoning: str


class CoverageVerdict(BaseModel):
    """How well the SOP covers a single regulatory atom."""

    atom_id: str
    status: CoverageStatus
    sop_atom_ids: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    reasoning: str


# ── Output layer ────────────────────────────────────────
class Finding(BaseModel):
    """A reportable gap, partial coverage, or contradiction."""

    finding_id: str
    type: FindingType
    reg_ref: dict[str, Any]  # {atom_id, quote}
    sop_anchors: list[dict[str, Any]] = Field(
        default_factory=list
    )  # [{atom_id, quote}]; empty for pure gaps
    explanation: str
    suggested_adjustment: str
    confidence: float
    validation: ValidationStatus  # citation checker result


class Report(BaseModel):
    """The final compliance report, with run metadata and findings."""

    run_meta: dict[str, Any]  # {timestamp, models, config, input_hashes}
    transparency: dict[str, Any]  # docs_excluded, parse_accounting_summary, ...
    findings: list[Finding] = Field(default_factory=list)  # grouped by SOP section
