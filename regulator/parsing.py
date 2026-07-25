"""Document parsing for regulatory PDFs and the SOP.

Stub implementations. Once implemented, these functions will extract text and
structure from real documents (PDF regulations and a DOCX SOP) and return
populated model instances.
"""

from __future__ import annotations

from pathlib import Path

from regulator.models import OperatingProcedure, RegulatoryDocument


def parse_regulatory_document(path: Path) -> RegulatoryDocument:
    """Parse a single regulatory document into a :class:`RegulatoryDocument`.

    Will (once implemented) read the PDF at ``path``, extract its text, and
    segment it into clauses and atomic requirements. For now it returns a
    placeholder document that records the source path and a title derived from
    the filename.
    """
    return RegulatoryDocument(path=path, title=path.stem)


def parse_operating_procedure(path: Path) -> OperatingProcedure:
    """Parse a Standard Operating Procedure into an :class:`OperatingProcedure`.

    Will (once implemented) read the DOCX at ``path`` and extract its title and
    body text. For now it returns a placeholder procedure that records the
    source path and a title derived from the filename.
    """
    return OperatingProcedure(path=path, title=path.stem)
