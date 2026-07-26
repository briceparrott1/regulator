"""Deterministic paragraph extraction from the SOP DOCX via ``python-docx``.

Unlike the regulatory PDFs, an SOP is a well-structured Word document whose
custom paragraph styles (``CNXL1``, ``CNXL3``, ``CNXL4``, ``CNXL5`` and their
``*Body`` variants) already encode the document hierarchy. This module's only
job is to hand back the ordered, non-empty paragraphs with their style names,
exactly as Word stored them, so the parser can rebuild the tree from the style
ladder. Body text is returned VERBATIM (double spaces, tabs, curly quotes and
inch marks preserved) — no whitespace normalization happens here or downstream.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import docx

# docProps/app.xml namespace for the extended properties (Pages, TitlesOfParts).
_APP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"


@dataclass(frozen=True)
class ParagraphRecord:
    """One non-empty document paragraph, kept verbatim with its style name."""

    text: str  # paragraph text exactly as python-docx returns it
    style: str  # Word style name, e.g. "CNXL1", "CNXL4", "CNXL1Body"


@dataclass(frozen=True)
class SopExtraction:
    """Ordered SOP paragraphs plus enough counts for parse accounting."""

    paragraphs: list[ParagraphRecord]  # non-empty, document order
    total_paragraphs: int  # every body paragraph, including empties
    skipped_empty: int  # paragraphs dropped for having no text
    pages_total: int | None  # from docProps/app.xml, if present
    title: str | None  # document title from docProps/app.xml, if present


def _read_app_properties(path: Path) -> tuple[int | None, str | None]:
    """Return ``(pages, title)`` from the DOCX ``docProps/app.xml``.

    A DOCX is a zip archive; the extended properties part records the page
    count Word last rendered and the document title. Both are best-effort: any
    missing part or malformed value yields ``None`` rather than raising.
    """
    pages: int | None = None
    title: str | None = None
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("docProps/app.xml")
    except (KeyError, zipfile.BadZipFile, OSError):
        return None, None
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return None, None

    pages_el = root.find(f"{{{_APP_NS}}}Pages")
    if pages_el is not None and pages_el.text and pages_el.text.strip().isdigit():
        pages = int(pages_el.text.strip())

    titles = root.find(f"{{{_APP_NS}}}TitlesOfParts")
    if titles is not None:
        # TitlesOfParts wraps a vt:vector of vt:lpstr entries; the first is the
        # document title. Match by local tag to stay namespace-agnostic.
        for element in titles.iter():
            if element.tag.endswith("}lpstr") and element.text:
                title = element.text.strip() or None
                break

    return pages, title


def extract_sop(path: Path) -> SopExtraction:
    """Extract ordered, non-empty paragraph records from the SOP DOCX.

    Paragraphs whose text is blank once stripped are skipped (their count is
    reported for accounting). Surviving paragraphs keep their text verbatim.
    """
    document = docx.Document(str(path))

    records: list[ParagraphRecord] = []
    total = 0
    skipped = 0
    for paragraph in document.paragraphs:
        total += 1
        if not paragraph.text.strip():
            skipped += 1
            continue
        style = paragraph.style.name if paragraph.style else ""
        records.append(ParagraphRecord(text=paragraph.text, style=style))

    pages, title = _read_app_properties(Path(path))
    return SopExtraction(
        paragraphs=records,
        total_paragraphs=total,
        skipped_empty=skipped,
        pages_total=pages,
        title=title,
    )
