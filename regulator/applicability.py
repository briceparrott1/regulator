"""Applicability filtering: narrow regulations and nodes to the SOP.

Document-level applicability is implemented here; node-level is still a stub.

The document-level matcher asks one question per (SOP, regulatory document)
pair: *does this document's own scope reach this procedure?* One LLM call
answers it, and the answer is recorded as a descriptive
:class:`~regulator.models.DocApplicabilityVerdict` — an observation, not a
decision. What the pipeline DOES with that observation is
:func:`should_audit`'s job, kept separate so the policy can be re-tuned without
touching the record or the prompt.

The call is given three things:

1. the SOP's profile (what the procedure does, with what, where),
2. the document's profile (what it governs), and
3. a deterministically selected slice of the document's OWN text — its scope,
   applicability, purpose and exemption passages (see :func:`scope_slice`).

The third input is not redundant. A verdict has to cite the scope sentence that
drives it, verbatim, and a profile is a summary: there is nothing in it to copy.
Handing the model the real sentences is also what lets it distinguish a scope
gate it can settle from one that turns on a fact the SOP never states.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from regulator.llm import StructureLLM
from regulator.models import (
    Confidence,
    DocApplicabilityVerdict,
    OperatingProcedure,
    RegulatoryDocument,
    RegulatoryNode,
    Verdict,
)
from regulator.parse_cli import _available_regulations
from regulator.parsing import _PROFILE_SIGNAL_PHRASES, _slug
from regulator.pdf_extract import PageRecord, extract_pages

# The matcher reuses the profile LLM (Sonnet by default, PROFILE_MODEL-
# overridable). Applicability is the same kind of semantic judgement call as
# document classification, and importing the factory — rather than building a
# second one — keeps the two stages on one model knob. Same rationale as
# profiles.py importing parsing's private helpers.
from regulator.profiles import _profile_llm

# ── Scope-slice selection (deterministic, no LLM) ───────
# Which passages of a regulatory document decide whether it reaches an SOP? The
# ones where the document states its own reach: a Scope / Applicability /
# Purpose section, a "this part applies to ..." sentence, an exemption. Those
# are also the only sentences a trigger_quote may legitimately come from, so the
# slice is built by finding them and taking a few lines of context around each.
#
# The patterns are deliberately narrow. Broad words ("requirements", "shall",
# "covered") match most pages of a regulatory PDF, which would turn the slice
# into the whole document — expensive, and (as the profile stage found the hard
# way) enough incidental body text to argue a model into a match it should not
# have made.
_SCOPE_HEADING_RE = re.compile(
    r"^\s*(?:appendix\s+[a-z0-9]+\b[.:\s-]*)?"  # optional "APPENDIX A:" prefix
    r"(?:\d+(?:\.\d+)*\s*[.)-]?\s*)?"  # optional clause number "1.2.1"
    r"(scope|applicability|application|field of application|purpose|general|"
    r"introduction|foreword|exemptions?|exclusions?|limitations?)\b",
    re.IGNORECASE,
)
_SCOPE_SENTENCE_RES = (
    # "This standard applies to ...", "The subpart shall not apply to ..."
    re.compile(
        r"\b(?:this|the)\s+\w+(?:\s+\w+)?\s+"
        r"(?:applies|shall\s+apply|does\s+not\s+apply|shall\s+not\s+apply|"
        r"covers|is\s+applicable|is\s+not\s+applicable)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:applies|shall\s+apply|does\s+not\s+apply)\s+to\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are)\s+(?:not\s+)?applicable\s+to\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are)\s+exempt\b|\bexempt\s+from\b", re.IGNORECASE),
)

# Lines of context kept around a matching line. A scope sentence usually runs on
# for several lines (the enumerated categories it covers), and the line before it
# is often the heading that names it.
_SCOPE_LINES_BEFORE = 1
_SCOPE_LINES_AFTER = 6

# Front matter is always included: it carries the title, issuing body, edition
# and any archival / "reference purposes only" marking — the facts that decide
# whether the document is a binding instrument at all. The char cap is PER
# front page, so a dense cover page cannot eat the whole slice budget.
_SCOPE_FRONT_PAGES = 2
_SCOPE_FRONT_CHAR_CAP = 6000
# Total slice budget. Roughly 6k tokens per pair, so all ten pairs of an eval
# run stay cheap.
_SCOPE_SLICE_CHAR_CAP = 24000

# Extracted pages are cached per PDF: one applicability run reads each document
# for the slice and (in the eval harness) again for the verbatim-quote check,
# and pdfplumber extraction is by far the slowest deterministic step we have.
_PAGE_CACHE: dict[Path, list[PageRecord]] = {}


def source_pdf_path(doc: RegulatoryDocument) -> Path | None:
    """Locate the PDF behind ``doc``, or ``None`` if it cannot be found.

    ``source_path`` is recorded absolutely at parse time, so an artifact copied
    between checkouts can point somewhere that no longer exists. Fall back to
    matching this checkout's ``data/regulations/`` by doc_id slug.
    """
    path = Path(doc.source_path)
    if path.exists():
        return path
    for candidate in _available_regulations():
        if _slug(candidate.stem) == doc.doc_id:
            return candidate
    return None


def document_pages(doc: RegulatoryDocument) -> list[PageRecord]:
    """Extracted page text for ``doc`` (cached); empty when the PDF is missing."""
    path = source_pdf_path(doc)
    if path is None:
        return []
    pages = _PAGE_CACHE.get(path)
    if pages is None:
        pages = extract_pages(path)
        _PAGE_CACHE[path] = pages
    return pages


def document_text(doc: RegulatoryDocument) -> str:
    """The full extracted text of ``doc``, pages joined in order."""
    return "\n".join(page.text for page in document_pages(doc))


def _is_scope_line(line: str) -> bool:
    """True when ``line`` states, or heads, a statement of the document's reach."""
    if _SCOPE_HEADING_RE.match(line):
        return True
    if any(pattern.search(line) for pattern in _SCOPE_SENTENCE_RES):
        return True
    lowered = line.lower()
    # Re-used from the profile stage: generic markings that reveal what a
    # document IS ("for reference purposes only", "archived", "lecture notes").
    # A document's nature gates applicability just as its subject matter does.
    return any(phrase in lowered for phrase in _PROFILE_SIGNAL_PHRASES)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent half-open line ranges, keeping document order."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _page_scope_blocks(page: PageRecord) -> list[str]:
    """Context windows around every scope-stating line on one page."""
    hits = [i for i, line in enumerate(page.lines) if _is_scope_line(line)]
    if not hits:
        return []
    windows = [
        (
            max(0, i - _SCOPE_LINES_BEFORE),
            min(len(page.lines), i + 1 + _SCOPE_LINES_AFTER),
        )
        for i in hits
    ]
    return [
        "\n".join(page.lines[start:end]).strip()
        for start, end in _merge_ranges(windows)
    ]


def scope_slice(pages: list[PageRecord]) -> str:
    """Select the document text the applicability matcher sees.

    Deterministic and document-agnostic: front matter first (identity and any
    archival/reference marking), then every scope / applicability / exemption
    passage in document order, each tagged with its page so a reviewer can find
    it. Blocks are added until :data:`_SCOPE_SLICE_CHAR_CAP` is reached; a
    document that states its reach in ten places is truncated at the cap rather
    than dropping its front matter.

    A document with no scope-stating passage at all (teaching material, say)
    yields front matter only — which is the right input for concluding that
    there is no scope sentence to quote.
    """
    if not pages:
        return ""

    parts: list[str] = []
    budget = _SCOPE_SLICE_CHAR_CAP

    for page in pages[:_SCOPE_FRONT_PAGES]:
        if not page.text.strip():
            continue
        block = f"[PAGE {page.page_number} — front matter]\n{page.text}"[
            :_SCOPE_FRONT_CHAR_CAP
        ]
        parts.append(block)
        budget -= len(block)

    # Front pages are already included in full, so scan only what follows them.
    seen: set[str] = set()
    for page in pages[_SCOPE_FRONT_PAGES:]:
        for block in _page_scope_blocks(page):
            if not block or block in seen:
                continue
            seen.add(block)
            tagged = f"[PAGE {page.page_number} — scope]\n{block}"
            if len(tagged) > budget:
                return "\n\n".join(parts)
            parts.append(tagged)
            budget -= len(tagged)
    return "\n\n".join(parts)


def normalize_quote(text: str) -> str:
    """Collapse whitespace so quotes survive PDF line wrapping."""
    return " ".join(text.split())


def quote_is_verbatim(quote: str | None, source_text: str) -> bool:
    """True when ``quote`` appears in ``source_text`` up to whitespace.

    The matcher promises a copy-pasteable ``trigger_quote``; this is that
    promise, checkable. Whitespace is normalized on both sides because the
    extractor breaks lines wherever the PDF did, so a quote spanning a line
    break carries a newline the source text renders differently.
    """
    if not quote:
        return False
    return normalize_quote(quote) in normalize_quote(source_text)


# ── Document-level applicability call ───────────────────
_VERDICTS: tuple[Verdict, ...] = ("applicable", "not_applicable", "conditional")
_CONFIDENCES: tuple[Confidence, ...] = ("high", "medium", "low")

_APPLICABILITY_SYSTEM_PROMPT = """\
You decide whether ONE regulatory / standards document reaches ONE Standard
Operating Procedure (SOP), and report what you observed. You are given the SOP's
profile, the document's profile, and VERBATIM EXCERPTS of the document — its
front matter plus every passage where it states its own reach (scope,
applicability, purpose, exemptions).

Reason from the DOCUMENT'S OWN SCOPE outward, never from topical resemblance:
1. What does the document say it applies to — which activity, article,
   substance, place, and addressee does its scope name?
2. Does the facility and work described by the SOP fall inside every one of
   those gates?
3. For any gate you cannot settle, is it because the SOP simply does not state
   the deciding fact — or because the gate plainly fails?

Verdicts — choose exactly one:
- "applicable": every scope gate you can check is satisfied on facts the SOP
  itself states, and none of the remaining unknowns could flip it.
- "conditional": the document's scope reaches this SOP's subject matter, but
  whether it binds turns on a specific fact the SOP does not state — a threshold
  quantity, a classification or status determination, an equipment type, a
  formal adoption. Name each such fact in "missing_facts".
- "not_applicable": a scope gate fails on facts already known (the activity is
  absent from the SOP, the place is wrong, the regulated article is something
  else, the document binds a party this procedure is not), or the document is
  not a normative instrument at all and states no scope to apply.

Discipline that decides the hard cases:
- Topical overlap is NOT applicability. A document about equipment the SOP
  happens to use does not apply if its jurisdiction excludes the facility, or if
  it is a product specification addressed to whoever makes that equipment rather
  than to whoever operates it.
- "conditional" requires a TEXTUAL HOOK in the document's scope plus a specific
  unstated fact that would decide the matter. A merely conceivable connection —
  the SOP does not mention the regulated activity at all, and nothing in the
  scope reaches what it does mention — is "not_applicable". If any absence could
  be called conditional, the verdict means nothing.
- Judge the document by what it IS as well as what it discusses. Teaching
  material, lecture notes, or a collection explicitly provided for reference
  only is not an operative rule; say so plainly rather than treating its subject
  matter as its scope.
- The SOP is a written procedure. A document that regulates the facility's work
  can still reach it through the procedure; a document that regulates the design
  or manufacture of an article generally does not.

Field rules:
- "trigger_quote": ONE contiguous passage copied CHARACTER FOR CHARACTER from
  the excerpts — the sentence or heading that drives your verdict. Never
  paraphrase, never join fragments with "...", never repair spelling or spacing.
  Drop the "[PAGE n — ...]" tag; quote only the document's own words. Use null
  ONLY when the excerpts contain no statement of reach at all.
- "reasons": one to four short sentences, each naming the concrete fact that
  drives it (an activity, a substance, a place, an addressee, a gate). No
  restatement of the verdict.
- "missing_facts": ONLY unstated facts that would DECIDE applicability. Use []
  when your verdict rests on facts already known — do not pad it with things
  that would merely be nice to know.
- "confidence": "high", "medium", or "low", for the verdict on this evidence.

Return ONLY this JSON object:
{"verdict": "applicable" | "not_applicable" | "conditional",
 "trigger_quote": "verbatim passage" or null,
 "reasons": ["..."],
 "missing_facts": ["..."],
 "confidence": "high" | "medium" | "low"}
"""


def _sop_summary(sop: OperatingProcedure) -> dict[str, Any]:
    """The SOP facts the matcher reasons over."""
    profile = sop.profile
    return {
        "title": sop.title,
        "jurisdiction": list(profile.jurisdiction),
        "industry": profile.industry,
        "activities": list(profile.activities),
        "substances": list(profile.substances),
        "equipment": list(profile.equipment),
    }


def _doc_summary(doc: RegulatoryDocument) -> dict[str, Any]:
    """The regulatory-document facts the matcher reasons over."""
    profile = doc.profile
    return {
        "title": doc.title,
        "framework": doc.framework,
        "edition": doc.edition,
        "doc_kind": profile.doc_kind,
        "jurisdiction": list(profile.jurisdiction),
        "activities": list(profile.activities),
        "substances": list(profile.substances),
        "equipment": list(profile.equipment),
        "industries": list(profile.industries),
        "addressee_types": list(profile.addressee_types),
    }


def _matcher_user_prompt(
    sop: OperatingProcedure, doc: RegulatoryDocument, excerpts: str
) -> str:
    """Assemble the volatile half of the applicability call."""
    return (
        "SOP PROFILE\n"
        + json.dumps(_sop_summary(sop), ensure_ascii=False, indent=2)
        + "\n\nREGULATORY DOCUMENT PROFILE\n"
        + json.dumps(_doc_summary(doc), ensure_ascii=False, indent=2)
        + "\n\nDOCUMENT EXCERPTS (verbatim; quote from these and nowhere else)\n"
        + (excerpts or "(no text could be extracted from this document)")
    )


def _coerce_verdict(value: Any) -> Verdict:
    """Map the model's verdict onto the vocabulary; unknown falls back safely."""
    text = str(value).strip().lower()
    return text if text in _VERDICTS else "conditional"  # type: ignore[return-value]


def _coerce_confidence(value: Any) -> Confidence:
    """Map the model's confidence onto the vocabulary; unknown becomes 'low'."""
    text = str(value).strip().lower()
    return text if text in _CONFIDENCES else "low"  # type: ignore[return-value]


def _coerce_quote(value: Any) -> str | None:
    """Normalize the trigger quote to a non-empty string or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    """Coerce a JSON field to a list of non-empty strings."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def judge_document_applicability(
    sop: OperatingProcedure,
    doc: RegulatoryDocument,
    llm: StructureLLM | None = None,
) -> DocApplicabilityVerdict:
    """Decide, in one LLM call, what ``doc``'s scope says about ``sop``.

    Thinking is disabled (as on the profile calls) so the whole token budget
    goes to the JSON answer, and sampling is left at the API default —
    ``temperature`` is a hard 400 on the Sonnet model these calls use.

    Best-effort, like the profile stage: if the call or its JSON fails, the pair
    comes back as a low-confidence "conditional" whose reason names the failure.
    That fails OPEN — a transient error keeps the document in the audit rather
    than silently dropping it, which is the safer error for a compliance run.
    """
    llm = llm or _profile_llm()
    excerpts = scope_slice(document_pages(doc))
    try:
        data = llm.propose_json(
            _APPLICABILITY_SYSTEM_PROMPT,
            _matcher_user_prompt(sop, doc, excerpts),
            max_tokens=2000,
            thinking={"type": "disabled"},
        )
    except Exception as exc:  # noqa: BLE001 — matching is best-effort, fails open
        return DocApplicabilityVerdict(
            doc_id=doc.doc_id,
            sop_id=sop.doc_id,
            verdict="conditional",
            trigger_quote=None,
            reasons=[f"applicability call failed: {type(exc).__name__}"],
            missing_facts=[],
            confidence="low",
        )

    return DocApplicabilityVerdict(
        doc_id=doc.doc_id,
        sop_id=sop.doc_id,
        verdict=_coerce_verdict(data.get("verdict")),
        trigger_quote=_coerce_quote(data.get("trigger_quote")),
        reasons=_string_list(data.get("reasons")),
        missing_facts=_string_list(data.get("missing_facts")),
        confidence=_coerce_confidence(data.get("confidence")),
    )


def judge_documents(
    sop: OperatingProcedure,
    docs: list[RegulatoryDocument],
    llm: StructureLLM | None = None,
) -> list[DocApplicabilityVerdict]:
    """Run :func:`judge_document_applicability` over every document, in order.

    One LLM instance is shared across the pairs so the cached system prompt is
    reused. Returned verdicts line up with ``docs`` index for index; the report
    stage will want them for its transparency section.
    """
    llm = llm or _profile_llm()
    return [judge_document_applicability(sop, doc, llm=llm) for doc in docs]


def should_audit(verdict: DocApplicabilityVerdict) -> bool:
    """Decide whether a judged document continues down the pipeline.

    This is the one decisive step in an otherwise descriptive stage. Only a
    "not_applicable" document is dropped: "applicable" obviously continues, and
    so does "conditional" — a conditional verdict means the document plausibly
    binds and we simply lack a fact, so dropping it would hide real findings.
    Its ``missing_facts`` are exactly what the report should surface for a human
    to settle.
    """
    return verdict.verdict != "not_applicable"


def get_applicable_regulatory_procedures(
    sop: OperatingProcedure,
    docs: list[RegulatoryDocument],
) -> list[RegulatoryDocument]:
    """Return the regulatory documents worth auditing ``sop`` against.

    Judges each document against the SOP (one LLM call per document) and keeps
    those :func:`should_audit` accepts, in the input order.
    """
    verdicts = judge_documents(sop, docs)
    return [doc for doc, verdict in zip(docs, verdicts) if should_audit(verdict)]


def get_applicable_nodes(
    sop: OperatingProcedure,
    doc: RegulatoryDocument,
) -> list[RegulatoryNode]:
    """Return the requirement nodes in ``doc`` that apply to ``sop``.

    Will (once implemented) select the individual :class:`RegulatoryNode`
    requirements from ``doc`` that are relevant to the SOP. This is not yet
    wired into the pipeline; it will be called from the coverage stage to focus
    checks on applicable requirements. For now it returns an empty list.
    """
    return []
