"""Document parsing for regulatory PDFs and the SOP.

Stub implementations. Once implemented, these functions will extract text and
structure from real documents (PDF regulations and a DOCX SOP) and return
populated model instances.
"""

from __future__ import annotations

from pathlib import Path

from regulator.models import (
    OperatingProcedure,
    RegProfile,
    RegulatoryDocument,
    SopProfile,
)


def parse_regulatory_document(path: Path) -> RegulatoryDocument:
    """Parse a single regulatory document into a :class:`RegulatoryDocument`.

    Will (once implemented) read the PDF at ``path``, extract its text, and
    segment it into atomic requirements. For now it returns a placeholder
    document with empty accounting, an empty profile, and no atoms.
    """
    return RegulatoryDocument(
        doc_id=path.stem,
        source_path=str(path),
        file_hash="",
        title=path.stem,
        framework="",
        edition="",
        parse_accounting={"pages_total": 0, "pages_parsed": 0, "warnings": []},
        profile=RegProfile(
            jurisdiction=[],
            doc_kind="regulation",
            activities=[],
            substances=[],
            equipment=[],
            industries=[],
            addressee_types=[],
        ),
        atoms=[],
    )


def parse_operating_procedure(path: Path) -> OperatingProcedure:
    """Parse a Standard Operating Procedure into an :class:`OperatingProcedure`.

    Will (once implemented) read the DOCX at ``path`` and segment it into
    ordered atoms. For now it returns a placeholder procedure with empty
    accounting, an empty profile, and no atoms.
    """
    return OperatingProcedure(
        doc_id=path.stem,
        source_path=str(path),
        file_hash="",
        title=path.stem,
        parse_accounting={"pages_total": 0, "pages_parsed": 0, "warnings": []},
        profile=SopProfile(
            jurisdiction=[],
            industry="",
            activities=[],
            substances=[],
            equipment=[],
            internal_references=[],
        ),
        atoms=[],
    )
