"""Regulator: a regulatory-compliance document processor.

Checks a Standard Operating Procedure (SOP) against a set of regulatory
documents and reports gaps and suggested adjustments.
"""

from __future__ import annotations

from regulator.applicability import (
    get_applicable_atoms,
    get_applicable_regulatory_procedures,
)
from regulator.coverage import get_batch_coverage
from regulator.findings import promote_findings
from regulator.parsing import parse_operating_procedure, parse_regulatory_document
from regulator.report import get_report

__all__ = [
    "get_applicable_atoms",
    "get_applicable_regulatory_procedures",
    "get_batch_coverage",
    "get_report",
    "parse_operating_procedure",
    "parse_regulatory_document",
    "promote_findings",
]
