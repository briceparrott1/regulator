"""Shared data models for the Regulator pipeline.

These dataclasses are intentionally minimal placeholders. They capture the
handful of fields the stubbed pipeline needs today; each will grow additional
fields as the parsing, applicability, coverage, and reporting stages are
implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Atom:
    """A single, atomic regulatory requirement extracted from a clause.

    An "atom" is the smallest checkable unit of obligation (for example, a
    single "shall" statement). It is what the coverage stage will ultimately
    test the SOP against.
    """

    text: str = ""


@dataclass
class Clause:
    """A regulatory clause: a coherent passage from a regulatory document.

    A clause is parsed out of a regulatory document and may decompose into one
    or more :class:`Atom` requirements.
    """

    identifier: str = ""
    text: str = ""
    atoms: list[Atom] = field(default_factory=list)


@dataclass
class RegulatoryDocument:
    """A parsed regulatory document (e.g. a single regulation PDF).

    Holds the source path, a human-readable title, and the clauses extracted
    from the document.
    """

    path: Path | None = None
    title: str = ""
    clauses: list[Clause] = field(default_factory=list)


@dataclass
class OperatingProcedure:
    """A parsed Standard Operating Procedure (SOP) document.

    Holds the source path, a title, and the SOP body text that will be checked
    against applicable regulatory requirements.
    """

    path: Path | None = None
    title: str = ""
    text: str = ""


@dataclass
class CoverageResult:
    """The outcome of checking the SOP against one regulatory atom.

    Records which atom was checked, whether the SOP appears to cover it, and a
    short note explaining the judgement.
    """

    atom: Atom | None = None
    covered: bool = False
    note: str = ""


@dataclass
class Finding:
    """A noteworthy result promoted from coverage analysis.

    Represents a gap or discrepancy worth surfacing in the report, along with a
    suggested SOP adjustment.
    """

    summary: str = ""
    suggestion: str = ""


@dataclass
class Report:
    """The final compliance-analysis report.

    ``text`` holds the rendered Markdown body that the entry point writes to
    ``output/report.md``.
    """

    text: str = ""
    findings: list[Finding] = field(default_factory=list)
