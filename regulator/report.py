"""Report rendering: assemble the final compliance report.

Stub implementation. Will (once implemented) render the findings into a
structured Markdown compliance report.
"""

from __future__ import annotations

from regulator.models import Finding, OperatingProcedure, Report


def get_report(sop: OperatingProcedure, findings: list[Finding]) -> Report:
    """Build the compliance :class:`Report` for ``sop`` from ``findings``.

    Will (once implemented) render each finding into a structured Markdown
    report describing discrepancies and suggested SOP adjustments. For now it
    returns a placeholder report noting that the pipeline is still stubbed.
    """
    title = sop.title or "Standard Operating Procedure"
    text = (
        f"# Compliance Report: {title}\n\n"
        "This is a placeholder report. The Regulator pipeline is currently "
        "stubbed, so no regulatory findings have been generated yet.\n\n"
        f"Findings: {len(findings)} (stub)\n"
    )
    return Report(text=text, findings=findings)
