"""Document parsing for regulatory PDFs and the SOP.

``parse_regulatory_document`` is the implemented path. Its contract: the LLM
proposes *structure only* (a set of citation-path node ids, each with a
verbatim start anchor), and every character of body text is sliced
deterministically from PDF text extracted by :mod:`regulator.pdf_extract`. No
LLM-generated prose is ever stored as a node body.

Parent links, ``is_leaf``, and ``section_lineage`` are derived deterministically
from the citation paths themselves (e.g. the parent of ``6.3.1.1`` is ``6.3.1``,
and a node is a leaf iff no other node names it as parent). A synthetic document
root node is prepended so the flat node list always forms a single tree.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from regulator.llm import StructureLLM
from regulator.models import (
    OperatingProcedure,
    RegProfile,
    RegulatoryDocument,
    RegulatoryNode,
    SopProfile,
)
from regulator.parse_store import write_parsed_document
from regulator.pdf_extract import PageRecord, extract_pages

# Where parsed documents are persisted as flat JSONL.
PARSED_DIR = Path(__file__).resolve().parent.parent / "data" / "parsed"

# Chunking: overlapping page windows sized for reliable structure proposals on
# Haiku. Small documents collapse to a single chunk.
CHUNK_PAGES = 6
CHUNK_OVERLAP = 2

# Each window is proposed more than once and the node sets are unioned. Haiku
# under-enumerates dense, tightly-set clause text on any single pass; a second
# independent pass reliably recovers clauses the first missed.
STRUCTURE_PASSES = 2

_STRUCTURE_SYSTEM_PROMPT = """\
You extract the STRUCTURE of a regulatory / standards document. You are given
raw text from one or more pages. Return the enumerated hierarchy of the
document as JSON. You propose structure ONLY — never rewrite or summarize body
text.

Return a single JSON object of this exact shape:
{
  "nodes": [
    {
      "node_id": "citation path exactly as printed",
      "anchor": "a verbatim substring copied EXACTLY from the page text that
                 marks where this node's text begins (include the citation
                 label and the next few words, ~4-10 words)",
      "page": <1-based page number where the node starts>,
      "is_normative": <true if a leaf clause states a requirement, else false>,
      "addressee": "<who the clause binds: employer, facility, manufacturer,
                     procedure, product... or null>"
    }
  ]
}

Completeness (CRITICAL): enumerate EVERY numbered/labelled unit you can see,
at every depth. Do not stop at the top two levels. If the text contains
"5.1.1", "5.1.2", "5.1.3", "6.3", "6.3.1", "6.3.1.1", "6.4.1", "6.4.2",
"6.4.3", each of those is its own node. Every numbered definition
("3.1.1", "3.1.2", ... "3.1.7") is its own node. Missing a clause is a worse
error than including a borderline one. Walk the whole document top to bottom.
- The text may be tightly set with few spaces (e.g. "3.1.5 plastic—a
  material..."). A clause label still starts a node even when it is jammed
  against the surrounding words.
- A clause defined immediately after a cross-reference RANGE is still its own
  node. If you see "characteristics found in 5.1.1 – 5.1.3" and then the text
  goes on to define "5.1.1 ...", "5.1.2 ...", "5.1.3 ...", emit all three.
- Emit the intermediate grouping levels too: if "6.3.1.1" exists, then "6.3"
  and "6.3.1" must also appear as nodes.

Rules for node_id (CRITICAL):
- Use the citation path exactly as printed in the document
  (e.g. "1910.119(f)(1)(i)", "6.3.1.1", "I.1.2.a").
- For lettered subsections whose labels restart under each parent (a), b),
  c)...), the node_id MUST be the FULL path so ids never collide. Example:
  an "a)" under section I.1.1 is "I.1.1.a"; a different "a)" under I.1.2 is
  "I.1.2.a". Never emit a bare "a" — always prefix the full parent path.
- Include BOTH internal (grouping) sections and their leaf clauses. An internal
  section that has sub-clauses is still a node; it carries its own intro text.

What is NOT a node (these fold into the nearest clause's body, never their own
node):
- NOTE / footnote paragraphs
- bullet / dash lists
- figure and table captions
- running headers, page numbers, watermark or banner lines

Anchor rules:
- Copy the anchor characters EXACTLY as they appear in the provided text,
  starting at the node's first character. Do not paraphrase or fix spacing.
- The anchor must be long and specific enough to be unique (include the words
  that follow the citation label).

Output ONLY the JSON object. No prose, no code fences.
"""

_PROFILE_SYSTEM_PROMPT = """\
You read the front matter / scope of a regulatory or standards document and
return a compact structured profile as JSON, of this exact shape:
{
  "title": "the document's title",
  "framework": "issuing body / framework, e.g. 'OSHA 29 CFR', 'ASTM', 'IEEE'",
  "edition": "edition or year, e.g. '2023' (empty string if unknown)",
  "doc_kind": "one of: regulation | standard | permit",
  "jurisdiction": ["federal", "state", or country names that apply"],
  "activities": ["regulated activities"],
  "substances": ["regulated substances / materials"],
  "equipment": ["regulated equipment"],
  "industries": ["industries governed"],
  "addressee_types": ["who it binds: employer, facility, manufacturer, ..."]
}
Use [] for lists you cannot determine and "" for unknown strings. Output ONLY
the JSON object, no prose and no code fences.
"""

_WS_RE = re.compile(r"\s+")
_BARE_LETTER_RE = re.compile(r"[A-Za-z]")


def _specificity(node_id: str) -> tuple[int, int]:
    """Rank a citation path by depth then length (more specific = larger)."""
    return (node_id.count("."), len(node_id))


def _slug(text: str) -> str:
    """Filesystem/id-friendly slug of a document name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "document"


def _norm_ws(text: str) -> str:
    """Collapse all runs of whitespace to single spaces."""
    return _WS_RE.sub(" ", text).strip()


def _normalize_node_id(raw: str) -> str:
    """Clean a proposed citation path: trim and drop trailing separators."""
    nid = _norm_ws(raw)
    # Drop stray trailing punctuation like "I.1." or "6.2)" without touching
    # internal separators.
    return nid.rstrip(" .)").strip()


def _is_numeric_id(node_id: str) -> bool:
    """True for pure numeric citation paths like ``6.3.1`` (not ``I.1.2.a``)."""
    return all(seg.isdigit() for seg in node_id.split("."))


def _locate_clause_start(text: str, label: str) -> int | None:
    """Offset of ``label`` used as a clause START in ``text``, else ``None``.

    A clause start is the label not preceded by a word char or dot (so it is not
    the tail of a longer number), not followed by a dot/digit (so ``6.3`` does
    not match inside ``6.3.1``), and followed by whitespace then a letter (so
    cross-references like ``5.1.1 - 5.1.3`` and bare figures like ``84 days``
    are not mistaken for clause starts).
    """
    pattern = re.compile(r"(?<![\w.])" + re.escape(label) + r"(?![\d.])\s+[A-Za-z]")
    match = pattern.search(text)
    return match.start() if match else None


def _augment_numeric_gaps(
    located: dict[str, dict[str, Any]], doc_text: _DocumentText
) -> None:
    """Deterministically fill missing numeric clauses the LLM under-enumerated.

    For every numeric node id the LLM proposed, add any missing ANCESTOR or
    SIBLING clause that actually appears as a clause start in the extracted
    text. This is dialect-safe: lettered paths like ``I.1.2.a`` are ignored,
    and only labels genuinely present in the text are added, so spurious nodes
    (cross-references, figure numbers) are not introduced.
    """
    text = doc_text.text

    def try_add(label: str) -> bool:
        if label in located:
            return False
        offset = _locate_clause_start(text, label)
        if offset is None:
            return False
        located[label] = {
            "node_id": label,
            "offset": offset,
            "page": doc_text.page_for_offset(offset),
            "is_normative": False,
            "addressee": None,
        }
        return True

    changed = True
    while changed:
        changed = False
        for node_id in [n for n in located if _is_numeric_id(n)]:
            segments = node_id.split(".")
            # Ancestors: 6.3.1.1 -> 6.3.1, 6.3, 6
            for depth in range(1, len(segments)):
                changed |= try_add(".".join(segments[:depth]))
            # Siblings: walk the last numeric segment up and down until a gap.
            last = int(segments[-1])
            for direction in (1, -1):
                value = last + direction
                while value >= 1:
                    sibling = ".".join(segments[:-1] + [str(value)])
                    if sibling in located:
                        value += direction
                        continue
                    if not try_add(sibling):
                        break
                    changed = True
                    value += direction
            # Children: a lone or leading child under a childless located parent
            # (e.g. "4.1" under located "4") is reached by neither the ancestor
            # nor the sibling walk. Probe "N.1", "N.2", ... until the first gap.
            child = 1
            while True:
                candidate = f"{node_id}.{child}"
                if candidate in located:
                    child += 1
                    continue
                if not try_add(candidate):
                    break
                changed = True
                child += 1
            if changed:
                break  # restart the scan with the enlarged id set


def _structural_parent(node_id: str, all_ids: set[str]) -> str | None:
    """Parent citation path derived structurally from ``node_id``.

    The parent is ``node_id`` with its final ``.``-delimited segment removed,
    but only if that shortened path is itself a real node. Otherwise the node
    is treated as top-level (a roman-numeral prefix like ``I`` in ``I.1`` is
    not a real node, so ``I.1`` becomes top-level).
    """
    if "." not in node_id:
        return None
    candidate = node_id.rsplit(".", 1)[0]
    return candidate if candidate in all_ids else None


def _chunk_windows(pages: list[PageRecord]) -> list[list[PageRecord]]:
    """Split pages into overlapping windows for structure proposals."""
    if len(pages) <= CHUNK_PAGES:
        return [pages]
    step = CHUNK_PAGES - CHUNK_OVERLAP
    windows: list[list[PageRecord]] = []
    start = 0
    while start < len(pages):
        windows.append(pages[start : start + CHUNK_PAGES])
        if start + CHUNK_PAGES >= len(pages):
            break
        start += step
    return windows


def _chunk_text(window: list[PageRecord]) -> str:
    """Render a page window as prompt text with page markers."""
    return "\n".join(f"[PAGE {p.page_number}]\n{p.text}" for p in window)


class _DocumentText:
    """Whitespace-normalized full-document text with page offset tracking."""

    def __init__(self, pages: list[PageRecord]) -> None:
        self._parts: list[str] = []
        self.page_span: dict[int, tuple[int, int]] = {}
        buf: list[str] = []
        cursor = 0
        for page in pages:
            norm = _norm_ws(page.text)
            if buf:
                buf.append(" ")
                cursor += 1
            start = cursor
            buf.append(norm)
            cursor += len(norm)
            self.page_span[page.page_number] = (start, cursor)
        self.text = "".join(buf)

    def page_for_offset(self, offset: int) -> int:
        for page_number, (start, end) in self.page_span.items():
            if start <= offset < end:
                return page_number
        return next(iter(self.page_span), 1)

    def locate(self, anchor: str, page_hint: int | None) -> int | None:
        """Return the global start offset of ``anchor``, or ``None``.

        Exact (whitespace-normalized) match is tried first, biased to the
        hinted page, then a case-insensitive fallback.
        """
        needle = _norm_ws(anchor)
        if not needle:
            return None
        if page_hint in self.page_span:
            start, end = self.page_span[page_hint]
            local = self.text.find(needle, start, end)
            if local != -1:
                return local
        idx = self.text.find(needle)
        if idx != -1:
            return idx
        idx = self.text.lower().find(needle.lower())
        return idx if idx != -1 else None


def _collect_proposals(
    llm: StructureLLM, windows: list[list[PageRecord]]
) -> list[dict[str, Any]]:
    """Run the structure LLM over every page window; return raw node dicts."""
    proposals: list[dict[str, Any]] = []
    for window in windows:
        text = _chunk_text(window)
        for _ in range(STRUCTURE_PASSES):
            result = llm.propose_json(_STRUCTURE_SYSTEM_PROMPT, text)
            for node in result.get("nodes", []):
                if isinstance(node, dict) and node.get("node_id"):
                    proposals.append(node)
    return proposals


def _build_profile(
    llm: StructureLLM, pages: list[PageRecord], doc_id: str
) -> tuple[RegProfile, str, str, str]:
    """Derive a RegProfile plus (title, framework, edition) from front matter."""
    front = _chunk_text(pages[:2])
    title, framework, edition = doc_id, "", ""
    try:
        data = llm.propose_json(_PROFILE_SYSTEM_PROMPT, front, max_tokens=1500)
        title = str(data.get("title") or doc_id)
        framework = str(data.get("framework") or "")
        edition = str(data.get("edition") or "")
        doc_kind = data.get("doc_kind")
        if doc_kind not in ("regulation", "standard", "permit"):
            doc_kind = "standard"
        profile = RegProfile(
            doc_id=doc_id,
            jurisdiction=[str(x) for x in data.get("jurisdiction", [])],
            doc_kind=doc_kind,
            activities=[str(x) for x in data.get("activities", [])],
            substances=[str(x) for x in data.get("substances", [])],
            equipment=[str(x) for x in data.get("equipment", [])],
            industries=[str(x) for x in data.get("industries", [])],
            addressee_types=[str(x) for x in data.get("addressee_types", [])],
        )
    except Exception:  # noqa: BLE001 — profiling is best-effort, never fatal
        profile = RegProfile(
            doc_id=doc_id,
            jurisdiction=[],
            doc_kind="standard",
            activities=[],
            substances=[],
            equipment=[],
            industries=[],
            addressee_types=[],
        )
    return profile, title, framework, edition


def parse_regulatory_document(
    path: Path, out_dir: Path | None = None
) -> RegulatoryDocument:
    """Parse a regulatory PDF into a :class:`RegulatoryDocument`.

    Pipeline: deterministic text extraction, LLM structure proposal over page
    windows, verbatim anchor transcription, deterministic tree assembly with a
    synthetic root, best-effort profiling, then persistence to JSONL.

    The parsed document is written as flat JSONL to ``out_dir`` when given,
    otherwise to the default :data:`PARSED_DIR`.
    """
    path = Path(path)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    doc_id = _slug(path.stem)

    pages = extract_pages(path)
    warnings: list[str] = []
    empty_pages = [p.page_number for p in pages if not p.text.strip()]
    for page_number in empty_pages:
        warnings.append(f"page {page_number}: no extractable text")

    llm = StructureLLM()
    windows = _chunk_windows(pages)
    proposals = _collect_proposals(llm, windows)

    doc_text = _DocumentText(pages)

    # Deduplicate by node_id (first locatable proposal wins). Track ids that
    # were proposed but whose anchor never located, so we can warn once.
    located: dict[str, dict[str, Any]] = {}
    never_found: set[str] = set()
    for prop in proposals:
        node_id = _normalize_node_id(str(prop["node_id"]))
        if not node_id or node_id in located:
            continue
        # A bare single letter (e.g. "c") is an orphaned lettered subsection the
        # model forgot to give a full path; drop it — real lettered nodes always
        # carry their parent path (e.g. "I.1.2.c").
        if _BARE_LETTER_RE.fullmatch(node_id):
            continue
        offset = doc_text.locate(str(prop.get("anchor", "")), prop.get("page"))
        if offset is None:
            never_found.add(node_id)
            continue
        located[node_id] = {
            "node_id": node_id,
            "offset": offset,
            "page": doc_text.page_for_offset(offset),
            "is_normative": bool(prop.get("is_normative", False)),
            "addressee": prop.get("addressee") or None,
        }
    # NB: warnings for these unlocated anchors are deferred until AFTER the
    # gap-filler runs — a node it later recovers must not carry a "dropped"
    # warning (see the truthful-accounting pass below).

    # Deterministically recover numeric clauses the LLM missed (ancestors and
    # siblings that genuinely appear as clause starts in the extracted text).
    _augment_numeric_gaps(located, doc_text)

    # Two node ids can resolve to the same start offset when the model labelled
    # one heading twice (e.g. "I.1.2.c" and a bare "c"). Keep the most specific
    # citation path per offset and drop the rest, so bodies never come out empty.
    best_at_offset: dict[int, str] = {}
    for node_id, info in located.items():
        offset = info["offset"]
        incumbent = best_at_offset.get(offset)
        if incumbent is None or _specificity(node_id) > _specificity(incumbent):
            best_at_offset[offset] = node_id
    kept_ids = set(best_at_offset.values())
    for node_id in set(located) - kept_ids:
        warnings.append(f"node {node_id}: duplicate of another node's anchor; dropped")
        located.pop(node_id)

    # Truthful accounting for anchors the LLM proposed but that never located.
    # Recompute now that the gap-filler (and offset dedup) have run so every
    # warning describes the node's FINAL fate: recovered nodes are not "dropped".
    for node_id in sorted(never_found):
        if node_id in located:
            warnings.append(
                f"node {node_id}: LLM anchor not found; recovered by gap-filler"
            )
        else:
            warnings.append(f"node {node_id}: start anchor not found; dropped")

    all_ids = set(located)
    # Which ids are named as a structural parent → those are internal nodes.
    parent_of = {nid: _structural_parent(nid, all_ids) for nid in all_ids}
    has_child = {parent for parent in parent_of.values() if parent is not None}

    root_id = f"{doc_id}-root"

    # Document order is the located offset order; bodies run from each anchor to
    # the next node's anchor.
    ordered = sorted(located.values(), key=lambda n: n["offset"])
    boundaries = [n["offset"] for n in ordered] + [len(doc_text.text)]

    def lineage_of(node_id: str) -> list[str]:
        ancestors: list[str] = []
        cur = parent_of[node_id]
        while cur is not None:
            ancestors.append(cur)
            cur = parent_of[cur]
        ancestors.reverse()
        return [root_id, *ancestors]

    profile, title, framework, edition = _build_profile(llm, pages, doc_id)

    nodes: list[RegulatoryNode] = [
        RegulatoryNode(
            node_id=root_id,
            parent_id=doc_id,
            section_lineage=[],
            body=title,
            is_leaf=False,
            provenance={"page": 1, "span": [0, 0]},
        )
    ]
    for index, node in enumerate(ordered):
        node_id = node["node_id"]
        start, end = node["offset"], boundaries[index + 1]
        body = doc_text.text[start:end].strip()
        is_leaf = node_id not in has_child
        struct_parent = parent_of[node_id]
        nodes.append(
            RegulatoryNode(
                node_id=node_id,
                parent_id=struct_parent if struct_parent is not None else root_id,
                section_lineage=lineage_of(node_id),
                body=body,
                is_leaf=is_leaf,
                provenance={"page": node["page"], "span": [start, end]},
                is_normative=node["is_normative"] if is_leaf else False,
                addressee=node["addressee"] if is_leaf else None,
            )
        )

    parse_accounting = {
        "pages_total": len(pages),
        "pages_parsed": len(pages) - len(empty_pages),
        "warnings": warnings,
    }

    document = RegulatoryDocument(
        doc_id=doc_id,
        source_path=str(path),
        file_hash=file_hash,
        title=title,
        framework=framework,
        edition=edition,
        parse_accounting=parse_accounting,
        profile=profile,
        nodes=nodes,
    )

    write_parsed_document(document, out_dir if out_dir is not None else PARSED_DIR)
    return document


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
