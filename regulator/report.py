"""Report rendering: assemble the final compliance report.

Stub implementation. Will (once implemented) render the findings into a
structured Markdown compliance report.
"""

from __future__ import annotations

from regulator.models import Finding, OperatingProcedure, Report


def get_report(sop: OperatingProcedure, findings: list[Finding]) -> Report:
    """Build the compliance :class:`Report` for ``sop`` from ``findings``.

    Will (once implemented) assemble run metadata, transparency accounting, and
    the grouped findings. For now it returns a placeholder report with empty
    metadata that simply carries ``findings`` through.
    """
    return Report(
        run_meta={
            "timestamp": "",
            "models": {},
            "config": {},
            "input_hashes": {"sop": sop.file_hash},
        },
        transparency={
            "docs_excluded": [],
            "parse_accounting_summary": {},
            "atoms_assessed": 0,
            "findings_quarantined": 0,
        },
        findings=findings,
    )


def render_report_markdown(report: Report) -> str:
    """Render ``report`` into a Markdown string for writing to disk.

    Will (once implemented) format each finding into a structured Markdown
    report grouped by SOP section. For now it returns placeholder markdown
    noting that the pipeline is still stubbed.
    """
    return (
        "# Compliance Report\n\n"
        "This is a placeholder report. The Regulator pipeline is currently "
        "stubbed, so no regulatory findings have been generated yet.\n\n"
        f"Findings: {len(report.findings)} (stub)\n"
    )
