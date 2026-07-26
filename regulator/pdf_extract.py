"""Deterministic text extraction from regulatory PDFs via ``pdfplumber``.

The LLM is only ever asked to propose *structure*. Every character of body
text a node ends up carrying is sliced from the output of this module, so the
extraction has to be deterministic and stable across runs.

Regulatory PDFs are frequently laid out in two columns. ``pdfplumber``'s
default ``extract_text`` reads across the full page width line by line, which
interleaves the two columns and scrambles the reading order (a sentence in the
left column gets cut in half by unrelated right-column text). To recover a
sane reading order we detect the column count from the horizontal distribution
of words and, when a page is two-column, extract each column separately and
concatenate them left-to-right.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber

# Lines containing any of these markers are publisher watermark/banner noise
# (e.g. the iTeh "Document Preview" overlay) and never belong in a node body.
WATERMARK_MARKERS = ("iTeh", "Document Preview", "standards.iteh.ai")

# Glyph-level watermark: the iTeh "Document Preview" diagonal overlay is set in
# large (24pt) bold type in a distinctive pink. Its individual glyphs are laid
# out diagonally *across* the body text, so they physically interleave with the
# 10pt body glyphs during word assembly and garble the extracted words (e.g.
# "5.1.1 Disintegration" comes out as "Pr5.1e.1vDiiseintwegration"). These have
# to be dropped BEFORE word assembly, which the line-based pass cannot do. The
# predicate is deliberately conservative: large AND pink. Across all 10 source
# PDFs the only large pink glyphs are this watermark — every legitimate large
# heading (page titles, section banners) is black or grayscale — so this cannot
# eat real headings.
WATERMARK_COLOR = (0.8784, 0.102, 0.349)  # iTeh pink, RGB non_stroking_color
WATERMARK_COLOR_TOL = 0.05
WATERMARK_MIN_SIZE = 18.0

# The corpus's PDFs carry no space glyphs, so pdfplumber must infer word breaks
# from inter-character gaps. Its default x_tolerance (3.0pt) drops spaces where
# justified text compresses word gaps below 3.0pt (measured ~2.9pt on ASTM),
# merging words. Intra-word gaps run <=0.5pt and the smallest real word space is
# ~1.75pt, so 1.5pt safely splits words without breaking tokens like "4.1".
WORD_SPLIT_X_TOLERANCE = 1.5


def _is_watermark_color(color: object) -> bool:
    """True when ``color`` is (approximately) the iTeh watermark pink (RGB)."""
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        return False
    return all(
        abs(float(c) - t) <= WATERMARK_COLOR_TOL for c, t in zip(color, WATERMARK_COLOR)
    )


def _is_watermark_glyph(obj: dict) -> bool:
    """True for a large pink glyph belonging to the iTeh preview watermark.

    Non-text objects (lines, rects, images) carry no ``size`` and are kept.
    """
    size = obj.get("size") or 0
    return size > WATERMARK_MIN_SIZE and _is_watermark_color(
        obj.get("non_stroking_color")
    )


@dataclass(frozen=True)
class PageRecord:
    """Deterministically extracted text for a single PDF page."""

    page_number: int  # 1-based
    text: str  # column-ordered, watermark-stripped page text
    lines: list[str]  # ``text`` split into individual lines


def _column_gutter(words: list[dict], page_width: float) -> float | None:
    """Detect a two-column gutter and return its x-coordinate (or ``None``).

    A two-column page has a vertical whitespace gutter: a near-vertical line in
    the central band that almost no *body* word crosses. Full-width banners,
    titles, and footnotes do cross the centre, so they are excluded (by width)
    before scanning — otherwise they mask the gutter. If even the least-crossed
    candidate line in the central band is crossed by more than a small fraction
    of body words, the page is single-column.

    Returns the x-coordinate of the detected gutter, or ``None`` for a
    single-column page.
    """
    body = [w for w in words if (w["x1"] - w["x0"]) < 0.45 * page_width]
    if len(body) < 20:
        return None
    total = len(body)
    best_x: float | None = None
    best_frac = 1.0
    # Scan candidate gutter positions across the central 35%–65% band.
    for percent in range(35, 66):
        x = page_width * percent / 100.0
        crossing = sum(1 for w in body if w["x0"] < x < w["x1"])
        frac = crossing / total
        if frac < best_frac:
            best_frac, best_x = frac, x
    if best_frac > 0.025 or best_x is None:
        return None
    # Require both sides to be substantially populated (a real split, not a
    # lone indented block).
    left = sum(1 for w in body if w["x1"] <= best_x) / total
    right = sum(1 for w in body if w["x0"] >= best_x) / total
    if left < 0.20 or right < 0.20:
        return None
    return best_x


def _extract_page_text(page: pdfplumber.page.Page) -> str:
    """Return page text in a sensible reading order (column-aware).

    Watermark glyphs are removed *before* word assembly so their diagonal
    overlay cannot interleave with and garble the body text.
    """
    page = page.filter(lambda obj: not _is_watermark_glyph(obj))
    words = page.extract_words()
    gutter_x = _column_gutter(words, page.width)
    if gutter_x is not None:
        left = (
            page.crop((0, 0, gutter_x, page.height)).extract_text(
                x_tolerance=WORD_SPLIT_X_TOLERANCE
            )
            or ""
        )
        right = (
            page.crop((gutter_x, 0, page.width, page.height)).extract_text(
                x_tolerance=WORD_SPLIT_X_TOLERANCE
            )
            or ""
        )
        return f"{left}\n{right}"
    return page.extract_text(x_tolerance=WORD_SPLIT_X_TOLERANCE) or ""


def _strip_watermarks(text: str) -> list[str]:
    """Drop whole lines that are publisher watermark/banner noise."""
    kept: list[str] = []
    for line in text.splitlines():
        if any(marker in line for marker in WATERMARK_MARKERS):
            continue
        kept.append(line)
    return kept


def extract_pages(path: Path) -> list[PageRecord]:
    """Extract per-page text records from the PDF at ``path``.

    Pages that yield no extractable text still appear in the result with empty
    ``text``/``lines`` so the caller can account for them (e.g. scanned pages
    that would need OCR) in ``parse_accounting``.
    """
    records: list[PageRecord] = []
    with pdfplumber.open(str(path)) as pdf:
        for index, page in enumerate(pdf.pages):
            raw = _extract_page_text(page)
            lines = _strip_watermarks(raw)
            text = "\n".join(lines)
            records.append(PageRecord(page_number=index + 1, text=text, lines=lines))
    return records
