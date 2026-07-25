"""Findings promotion: turn coverage results into reportable findings.

Stub implementation. Will (once implemented) select the coverage gaps and
discrepancies worth surfacing to the user.
"""

from __future__ import annotations

from regulator.models import CoverageResult, Finding


def promote_findings(coverage: list[CoverageResult]) -> list[Finding]:
    """Promote noteworthy coverage results into :class:`Finding` objects.

    Will (once implemented) inspect ``coverage``, keep the uncovered or
    conflicting requirements, and attach a suggested SOP adjustment to each.
    For now it returns an empty list.
    """
    return []
