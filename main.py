"""Entry point for the Regulator compliance pipeline.

Wires the pipeline stages together end to end. Every stage is currently a stub,
so running ``python main.py`` completes without external dependencies and writes
a placeholder report to ``output/report.md``.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from regulator.applicability import get_applicable_regulatory_procedures
from regulator.coverage import get_batch_coverage
from regulator.findings import promote_findings
from regulator.parsing import parse_operating_procedure, parse_regulatory_document
from regulator.report import get_report, render_report_markdown

PROJECT_ROOT = Path(__file__).resolve().parent
REGULATIONS_DIR = PROJECT_ROOT / "data" / "regulations"
SOP_PATH = PROJECT_ROOT / "data" / "sop" / "original.docx"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORT_PATH = OUTPUT_DIR / "report.md"


def main() -> None:
    """Run the full compliance pipeline and write the report to disk."""
    load_dotenv()

    # Stage 1: parse every regulatory document.
    regulatory_paths = sorted(REGULATIONS_DIR.glob("*.pdf"))
    regulatory_docs = [parse_regulatory_document(path) for path in regulatory_paths]
    total_nodes = sum(len(doc.nodes) for doc in regulatory_docs)
    print(
        f"Parsed {len(regulatory_docs)} regulatory documents "
        f"({total_nodes} nodes total)"
    )

    # Stage 2: parse the SOP.
    sop = parse_operating_procedure(SOP_PATH)
    print(f"Parsed SOP '{sop.title}' (stub)")

    # Stage 3: narrow to the applicable regulatory documents.
    applicable = get_applicable_regulatory_procedures(sop, regulatory_docs)
    print(f"Selected {len(applicable)} applicable regulatory documents (stub)")

    # Stage 4: check SOP coverage against applicable requirements.
    coverage = get_batch_coverage(sop, applicable)
    print(f"Computed coverage for {len(coverage)} requirements (stub)")

    # Stage 5: promote coverage results into findings.
    findings = promote_findings(coverage)
    print(f"Promoted {len(findings)} findings (stub)")

    # Stage 6: render the report.
    report = get_report(sop, findings)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report_markdown(report), encoding="utf-8")
    print(f"Wrote report to {REPORT_PATH} (stub)")


if __name__ == "__main__":
    main()
