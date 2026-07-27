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
import json
import re
from pathlib import Path
from typing import Any

import anthropic

from regulator.llm import StructureLLM
from regulator.models import (
    OperatingProcedure,
    RegProfile,
    RegulatoryDocument,
    RegulatoryNode,
    SopNode,
    SopProfile,
)
from regulator.parse_store import write_parsed_document
from regulator.pdf_extract import PageRecord, extract_pages
from regulator.sop_extract import ParagraphRecord, extract_sop

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

# Profile facets are unioned across passes for the same reason (see
# _merge_profile_passes): a single pass drops a real facet item often enough to
# matter, and which item it drops varies between runs.
PROFILE_PASSES = 2

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
You read excerpts of a regulatory / standards / reference document (front
matter, table of contents, scope, and sampled sections) and return a compact
structured profile as JSON. Judge the document by what it ACTUALLY IS, not by
the subject it discusses — a teaching document about a standard is not the
standard, and an agency directive about a rule is not the rule.

IMPORTANT — profile ONE document. The excerpts are sampled from across a PDF and
may contain adjacent, appended, or unrelated material (a neighbouring subpart,
another standard, a bundled appendix). Identify the single document named in the
front matter / title and profile only THAT document. Ignore any facet that
belongs to neighbouring material rather than to this document's own scope.

Return a single JSON object of this exact shape:
{
  "title": "the document's title",
  "framework": "issuing body / framework, e.g. 'OSHA 29 CFR', 'API', 'NFPA'",
  "edition": "edition or year, e.g. '2022' (empty string if unknown)",
  "doc_kind": "one of the six kinds below",
  "jurisdiction": ["where it applies, using the vocabulary below"],
  "activities": ["regulated / described activities as verb-phrases"],
  "substances": ["substances / materials in scope"],
  "equipment": ["equipment / articles in scope"],
  "industries": ["industries or sectors in scope"],
  "addressee_types": ["who it binds or is written for"]
}

doc_kind — choose exactly one, by one-line criteria:
- "regulation": the codified text of a binding law or rule issued by a
  government body (a statute, a CFR part, a state regulation). This is the
  enforceable rule itself, in regulatory language ("§", "shall", parts/subparts).
- "compliance_directive": a government agency's OWN internal instruction,
  directive, policy, or enforcement/inspection guidance about how to implement
  or enforce a regulation — e.g. an enforcement directive / CPL / program
  instruction addressed to the agency's officers or inspectors. It is ABOUT a
  regulation; it is not the regulation's codified text. Watch for titles like
  "Directive"/"Instruction"/"CPL", an issuing agency office, and markings such
  as archived / superseded / dated guidance.
- "industry_standard": a voluntary consensus standard published by a
  standards-developing organization or trade body (ASME, API, ASTM, ANSI,
  NFPA, IEEE, ISO, UL, ...). It becomes mandatory only when an authority or
  contract adopts it.
- "national_standard": a standard issued or adopted by a single nation's
  standards body — typically that nation's adoption / transposition of an
  international standard (e.g. a national body adopting an IEC or ISO standard).
  Watch for a national standards-body identifier plus "adoption of IEC/ISO ...".
- "reference_package": a compiled collection assembled for information or
  reference rather than as an operative rule — e.g. a state implementation-plan
  reference compilation, a bundle of statutes/plans gathered for context, or an
  appendix explicitly provided "for reference purposes only" / "not to be
  approved". It is not itself the operative binding instrument.
- "non_regulatory": material that is neither a regulation nor a standard nor a
  reference package — e.g. university/lecture notes, an educational overview or
  introduction, a tutorial, slides, marketing, or descriptive commentary about
  a topic. Watch for an academic author/affiliation, "Introduction to ...", and
  the ABSENCE of any scope / applicability / requirements section.

jurisdiction — use this vocabulary (a list; may hold more than one):
- "US-federal" for a federal US law/rule;
- "US-<STATE>" for a US state, e.g. "US-PA" for Pennsylvania;
- "US" for something US-wide but not tied to a government level (a US industry
  standard);
- a country name ("Saudi Arabia") for a national document;
- "adopted-by-AHJ" ONLY for a code/standard that becomes binding where an
  authority having jurisdiction adopts it (fire/building/safety codes). A
  general consensus standard with no such adoption mechanism is just "US"
  (optionally also "international ..."); a federal rule is "US-federal", never
  "adopted-by-AHJ". "adopted-by-AHJ" names an adoption MECHANISM, not a place,
  so it never stands alone: always pair it with the country-level entry for the
  body that published the standard ("US" for a US standards body);
- descriptive phrases are fine when apt, e.g.
  "industry-adopted (jurisdiction-dependent)" or "international (WTO TBT-aligned)";
- use [] (empty) when the document is non-regulatory or states no jurisdiction;
- a STATE (or other sub-national) document that implements, submits to, or is
  approved under a federal programme still has only its own jurisdiction. Do not
  add the federal level merely because the federal programme it answers to is
  referenced throughout.

Field guidance — aim for COMPLETENESS on what the document actually governs
(omitting a governed concept is the worst error), while staying in scope:
- activities: verb-phrases naming what the document governs or describes
  ("in-service inspection of pressure vessels", "hot work permitting").
  START with the primary purpose or use that the document exists to govern —
  what the regulated articles are actually used to DO, or what outcome the rule
  is written to control — before any administrative, recordkeeping, or testing
  activity. Then list EVERY distinct requirement area, named program element,
  or subprogram as its own phrase, reusing the document's own names for them;
  never collapse several named elements into one umbrella phrase. Cross-check
  this list against the equipment list: whenever the document sets a recurring
  inspection, testing, or maintenance duty for an item in scope — including the
  protective and relief devices attached to the main article — emit an activity
  phrase naming that duty for that item. An activity list that covers only the
  primary article is incomplete. If the document
  enumerates its required elements (in a contents list, a paragraph-by-paragraph
  structure, or a set of named subprograms), walk that enumeration and emit one
  phrase per element — this list should be thorough rather than summarized. For a
  document describing a government program, include the administration of the
  program itself at its own level of government, and any formal delegation or
  agreement with another level of government (e.g. agreements with counties or
  local programs) as separate activities.
- equipment: the machines, devices, and installed systems the requirements act
  upon. Include BOTH the umbrella / family term AND its notable specific members
  — a reader must be able to find the general class and the examples. Always
  include the document's own CATCH-ALL scope category verbatim (the generic class
  term it names alongside its specific examples, often "any other ..." or a
  collective noun for the whole family), since that term carries the scope. Where
  the document ENUMERATES the categories of equipment it covers (a definition of
  covered equipment, or a list under an integrity/inspection requirement), emit
  one entry per enumerated category, including the systems-level ones — and
  prefer that enumeration over specific items picked out of illustrative appendix
  or example lists. A requirement to compile "process safety information" (or an
  equivalent design-information dossier) enumerates the equipment classes in
  scope — vessels, piping, relief and vent systems, controls, ventilation — so
  emit each such class. NEVER list a category the document EXEMPTS or excludes,
  and never list example FACILITY types (tank farms, named plants, sector-specific
  installations) as equipment — an enforcement or guidance document illustrates
  its reach with example sites, and those examples are not its equipment scope.
  Treat these as equipment too, and include each one the document names:
  * its own generic term for the regulated UNIT or POINT at which the phenomenon
    it controls occurs — a release / emission / discharge point, an affected or
    emission unit, a covered source. These belong in equipment even though they
    name a locus rather than a machine; a document that regulates emissions is
    largely ABOUT such points, so omitting the term loses its core scope;
  * TEMPORARY, MOBILE, and construction-phase objects it names as covered
    ("apparatus of a permanent or temporary character", "equipment or materials
    used therein") — not only finished permanent installations. Do not stop at
    the abstract phrase: name the concrete erection and construction machinery
    that class denotes on a real work site (cranes, derricks, rigs, hoists),
    since those are what a reader must match against;
  * when applicability turns on a physical THRESHOLD (a height, size, or
    capacity), the general class of object that threshold selects, in the
    document's own terms.
  Exclude: apparatus used only to carry out a test or laboratory method. If the
  regulated article is a MATERIAL or an item defined by the material it is made
  of, it belongs in substances and equipment may be [].
  CRITICAL exclusion — the PROTECTED side. Where the document regulates an
  activity because it might endanger or interfere with something else, that
  protected something is NOT equipment: not the facilities the rule shields, not
  the natural features or terrain it measures against, and not the third-party
  infrastructure whose safe operation is the rule's purpose. Equipment is ONLY
  what the regulated party itself proposes to build, erect, install, or operate.
  Ask of every candidate: "does the regulated party build or run this, or is it
  the thing being protected FROM them?" — and drop it if it is the latter.
- substances: the process / hazardous materials or material classes the
  requirements govern, including the material class a specification-style
  document defines, tests, or labels. Substances are what FLOWS THROUGH, is
  PROCESSED BY, or is HANDLED BY the regulated equipment — never what that
  equipment is MADE OF. So NEVER list materials of construction (steels, alloys,
  stainless grades, plastics an article is fabricated from) even when the
  document devotes whole sections to their corrosion, damage mechanisms, or
  material selection. Also exclude incidental mentions, and exclude analytes,
  residues, contaminants, impurity classes, or measured parameters that appear
  only as quantities a test method reports or as pass/fail limits a product must
  stay under — those are acceptance criteria, not substances in scope.
- industries: the sectors the scope names or clearly implies, in the document's
  own sector wording. Work through these four questions in turn and emit an entry
  for every answer the document supports — the common failure is answering only
  the first:
  1. which sectors MAKE or produce the regulated article?
  2. which sectors OPERATE or USE it in their own work?
  3. which sectors SERVICE, MAINTAIN, or INSPECT it — and, if the article is
     itself a tool for doing maintenance or service work, that service sector?
  4. which named sub-sectors or SERVICE CATEGORIES does the scope spell out
     ("... including X service, Y, and Z")? Emit each spelled-out one as its own
     entry, in the document's own wording, rather than folding it into the
     umbrella sector.
  Also include the broad umbrella sector for the field the document belongs to.
  Prefer the sectors named in the SCOPE over ones inferred from illustrative
  appendix or example lists, and skip the long tail of niche examples such lists
  contain. Never list a sector the document EXEMPTS, nor one that appears only as
  a single worked example of an inspection or enforcement case.
- addressee_types: every party the document assigns a ROLE or RESPONSIBILITY to,
  named at the level of a PARTY TYPE rather than an individual office — usually
  two to four entries. Work through the roles it names and include each one:
  * the party that must comply (e.g. the employer, the owner-operator, the
    facility, the manufacturer);
  * the END USERS / owners / operators of the regulated article — include these
    even when the document's requirements fall mainly on its maker;
  * for a specification or labeling standard, the party making the claim or
    applying the label — list this party as its OWN entry even when it is
    usually the same organization as the manufacturer;
  * any authority the document's own CLAUSES give an act to perform (an
    authority having jurisdiction that approves, accepts, or grants exceptions).
    Do NOT include an authority that appears only in adoption or reference
    boilerplate — a foreword noting that administrative or regulatory bodies may
    reference, adopt, or enforce the standard gives them no role in it;
  * the issuing agency's OWN officers, but ONLY when the document is written TO
    them (an internal directive or inspection guidance whose audience is those
    officers). For a rule addressed to the public, do NOT list the agency, its
    regional offices, or the officials who run its internal review and hearing
    process — they administer the rule, they are not its addressee types;
  * for a document about a government program, the government bodies that
    administer it (the state agency, the county/local programs).
  Do NOT infer stakeholders the document does not itself put under a duty — ask
  of each candidate "does THIS document tell them to do something?" and drop it
  if not. In particular EXCLUDE: other agencies or bodies the document merely
  mentions, cross-references, or notes as running their own parallel programs;
  enforcement or regulatory authorities it does not itself empower; laboratories
  or other parties that merely perform a referenced test method; and bodies that
  do nothing but receive reports or submittals. Above all, when a document sets
  requirements a product must meet for some DOWNSTREAM ENVIRONMENT or process,
  the operators of that downstream environment are NOT addressees — the document
  describes their setting, it does not bind them, however central that setting is
  to the document's subject. Nor are the product's eventual end users, nor the
  laboratories that run its test methods, unless the document states a duty they
  must discharge. For a product specification the answer is usually just the two
  parties that make the product and make the claim about it.
  Add the literal entry "procedure" when the document requires WRITTEN OPERATING
  PROCEDURES for running the process as one of its own named required program
  elements (look for a paragraph obliging the regulated party to develop and
  maintain written operating procedures); for most standards this does NOT apply.
- Use [] for a facet the document genuinely does not govern (e.g. a document
  about radio protocols or aerial devices governs no substances; purely
  educational material has no addressee).

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


# A clause start is a dotted-numeric label (e.g. ``6.3.1``) that opens a clause
# rather than appearing mid-sentence. It is: not preceded by a word char or dot
# (so it is not the tail of a longer number); captured MAXIMALLY via the greedy
# ``(?:\.\d+)*`` (so ``6.3`` is never captured inside ``6.3.1`` — the whole path
# is consumed); not followed by a dot/digit; and followed by whitespace then a
# letter (so cross-references like ``5.1.1 - 5.1.3`` and bare figures like
# ``84 days`` are not mistaken for clause starts).
#
# Running this once with ``finditer`` and keeping the first occurrence of each
# captured id is exactly equivalent to the old per-label ``re.search`` (same
# lookarounds, same maximal-path capture, same first-match-wins) run for every
# label — but O(len(text)) total instead of O(labels x len(text)).
_CLAUSE_START_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)*)(?![\d.])\s+[A-Za-z]")


def _clause_start_offsets(text: str) -> dict[str, int]:
    """Map every dotted-numeric clause-start label in ``text`` to its offset.

    Single regex pass; the first occurrence of each label wins, matching a
    per-label :func:`re.search`. See :data:`_CLAUSE_START_RE` for the exact
    clause-start semantics.
    """
    offsets: dict[str, int] = {}
    for match in _CLAUSE_START_RE.finditer(text):
        offsets.setdefault(match.group(1), match.start(1))
    return offsets


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
    # Precompute every clause-start offset ONCE so the probes below are O(1)
    # dict lookups instead of full-text regex scans (the old per-label search
    # made this routine quadratic on large, clause-dense documents).
    candidates = _clause_start_offsets(text)

    def try_add(label: str) -> bool:
        if label in located:
            return False
        offset = candidates.get(label)
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
    llm: StructureLLM,
    windows: list[list[PageRecord]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Run the structure LLM over every page window; return raw node dicts.

    A single window/pass that fails (malformed JSON surviving propose_json's
    retry, or an Anthropic API error) must not abort the whole run — one bad
    window out of hundreds would otherwise discard the entire spend. Each
    failure is logged to ``warnings`` and skipped; the remaining windows and
    passes still contribute their nodes.
    """
    proposals: list[dict[str, Any]] = []
    for window in windows:
        text = _chunk_text(window)
        first_page, last_page = window[0].page_number, window[-1].page_number
        for pass_index in range(STRUCTURE_PASSES):
            try:
                # Structure windows can be dense; the default 8000-token budget
                # risks truncated JSON, so give them a larger ceiling. Profile
                # calls keep their small explicit budget.
                result = llm.propose_json(
                    _STRUCTURE_SYSTEM_PROMPT, text, max_tokens=16000
                )
            except (json.JSONDecodeError, anthropic.AnthropicError) as exc:
                warnings.append(
                    f"window pages {first_page}-{last_page}: structure pass "
                    f"{pass_index + 1} failed: {type(exc).__name__}; skipped"
                )
                continue
            for node in result.get("nodes", []):
                if isinstance(node, dict) and node.get("node_id"):
                    proposals.append(node)
    return proposals


# Valid doc_kind values (mirrors DocKind); used to validate the LLM's answer.
_DOC_KINDS = (
    "regulation",
    "compliance_directive",
    "industry_standard",
    "national_standard",
    "reference_package",
    "non_regulatory",
)

# Generic (document-agnostic) phrases that tend to reveal what a document IS —
# archival markings, reference/informational disclaimers, teaching material,
# national adoption of a standard. Pages containing any of these are pulled into
# the profile input so the classifier sees the smoking-gun page wherever it sits
# (e.g. a "reference purposes only" appendix, an "Introduction to ..." cover).
# These are classification SIGNALS, not answers for any specific document.
# Deliberately narrow: broad words like "scope", "enforcement", or "directive"
# match most pages of a regulatory PDF, which would pull in huge swaths of
# unrelated neighbouring material and drown the real signal.
_PROFILE_SIGNAL_PHRASES = (
    "reference purposes",
    "not to be approved",
    "for informational purposes",
    "informational only",
    "lecture notes",
    "introduction to",
    "university",
    "archived",
    "superseded",
    "adoption of",
    "implementation plan",
)

# Keep the profile call's input well under Sonnet's context while giving it far
# more than the old two-page front-matter slice. ~70k chars ≈ 18k tokens; across
# 11 documents that stays comfortably under the ~300k-token/iteration ceiling.
#
# Bigger is NOT better here: widening this to 90k measurably hurt precision,
# pulling in enough incidental late-document material (processing media, test
# limits, neighbouring programs) that the classifier started reporting it as
# in-scope. The cap is a relevance filter, not just a cost control.
_PROFILE_INPUT_CHAR_CAP = 70000
# Enough front pages to clear a cover, a revision/foreword page, and a
# multi-page table of contents before the scope/applicability section — which is
# where the facets we profile are actually defined. Eight pages stopped just
# short of it on standards with a long front matter.
_PROFILE_FRONT_PAGES = 12
_PROFILE_BACK_PAGES = 3
_PROFILE_SAMPLE_COUNT = 5


def _page_budgets(lengths: list[int], cap: int) -> list[int]:
    """Split ``cap`` characters across pages of the given lengths, fairly.

    Each still-unsatisfied page is offered an equal share of what is left; pages
    shorter than their share take only what they need and donate the remainder
    to the pages still over budget, repeating until the budget is exhausted.

    This matters because a flat ``cap // n`` slice truncates long pages even
    when the selected text would have fit under the cap in full, silently
    dropping real text while short pages waste their unused share.
    """
    budgets = [0] * len(lengths)
    pending = list(range(len(lengths)))
    remaining = cap
    while pending:
        share = remaining // len(pending)
        if share == 0:
            break
        satisfied = [i for i in pending if lengths[i] <= share]
        if not satisfied:
            # Every remaining page is longer than its share: split evenly.
            for index in pending:
                budgets[index] = share
            break
        for index in satisfied:
            budgets[index] = lengths[index]
            remaining -= lengths[index]
        pending = [i for i in pending if lengths[i] > share]
    return budgets


def _profile_input_text(pages: list[PageRecord]) -> str:
    """Select the text the profile classifier sees.

    Front matter alone is often insufficient to tell what a document IS (a
    "reference purposes only" note can live in an appendix; a document's true
    nature as lecture notes shows on the cover). We include the front pages, the
    last few pages, evenly-spaced samples, and any page containing a generic
    classification-signal phrase — then cap the total size.
    """
    if not pages:
        return ""
    total = len(pages)
    chosen: set[int] = set(range(min(_PROFILE_FRONT_PAGES, total)))
    chosen.update(range(max(0, total - _PROFILE_BACK_PAGES), total))
    if total > _PROFILE_FRONT_PAGES + _PROFILE_BACK_PAGES:
        step = max(1, total // (_PROFILE_SAMPLE_COUNT + 1))
        chosen.update(range(step, total, step))
    for index, page in enumerate(pages):
        lowered = page.text.lower()
        if any(phrase in lowered for phrase in _PROFILE_SIGNAL_PHRASES):
            chosen.add(index)

    selected = [pages[i] for i in sorted(chosen) if pages[i].text.strip()]
    # Budget the cap ACROSS the selected pages rather than truncating the joined
    # text: a hard prefix cut would spend the whole budget on the front pages and
    # silently drop the sampled later pages, which is exactly where a document's
    # requirement areas and catch-all scope terms tend to live.
    budgets = _page_budgets(
        [len(page.text) for page in selected], _PROFILE_INPUT_CHAR_CAP
    )
    parts = [
        f"[PAGE {page.page_number}]\n{page.text[:budget]}"
        for page, budget in zip(selected, budgets)
    ]
    return "\n".join(parts)


# Facets of the profile that are lists, and so can be unioned across passes.
_PROFILE_LIST_FIELDS = (
    "jurisdiction",
    "activities",
    "substances",
    "equipment",
    "industries",
    "addressee_types",
)


def _merge_profile_passes(passes: list[dict[str, Any]]) -> dict[str, Any]:
    """Union the list facets of several profile passes into one profile dict.

    Scalar facets (title, framework, edition, doc_kind) come from the first
    pass — they are single judgements, and averaging them is meaningless. The
    list facets are UNIONED in first-seen order, deduplicated case-insensitively.

    Rationale mirrors ``STRUCTURE_PASSES``: on any single pass the model
    under-enumerates a facet, dropping one or two real items more or less at
    random, and which items it drops varies run to run. Recall is what matters
    for a profile (a missed concept makes a document look inapplicable), and a
    second independent pass recovers most of what the first missed. Extra items
    are cheap by comparison, so the union is the right trade.
    """
    merged = dict(passes[0])
    for field in _PROFILE_LIST_FIELDS:
        seen: dict[str, str] = {}
        for data in passes:
            for value in data.get(field, []) or []:
                text = str(value).strip()
                key = " ".join(text.lower().split())
                if key and key not in seen:
                    seen[key] = text
        merged[field] = list(seen.values())
    return merged


def _run_profile_passes(
    llm: StructureLLM, system_prompt: str, text: str
) -> dict[str, Any]:
    """Run the profile call ``PROFILE_PASSES`` times and merge the results.

    A pass that raises is skipped so one bad call cannot sink the rest; if every
    pass fails the error propagates to the caller's best-effort fallback.
    """
    passes: list[dict[str, Any]] = []
    for _ in range(PROFILE_PASSES):
        try:
            passes.append(
                llm.propose_json(
                    system_prompt,
                    text,
                    max_tokens=4000,
                    thinking={"type": "disabled"},
                )
            )
        except Exception:  # noqa: BLE001,S112 — a failed pass is simply skipped
            continue
    if not passes:
        raise ValueError("every profile pass failed")
    return _merge_profile_passes(passes)


def _build_profile(
    llm: StructureLLM, pages: list[PageRecord], doc_id: str
) -> tuple[RegProfile, str, str, str]:
    """Derive a RegProfile plus (title, framework, edition) from the document.

    The classifier sees a widened slice of the document (see
    :func:`_profile_input_text`) rather than only the first two pages, and
    thinking is disabled so the whole token budget is spent on the JSON answer.
    The call is repeated ``PROFILE_PASSES`` times and the list facets are
    unioned (see :func:`_merge_profile_passes`); a pass that fails is skipped,
    so the profile still lands as long as one pass succeeds.
    """
    sample = _profile_input_text(pages)
    title, framework, edition = doc_id, "", ""
    try:
        data = _run_profile_passes(llm, _PROFILE_SYSTEM_PROMPT, sample)
        title = str(data.get("title") or doc_id)
        framework = str(data.get("framework") or "")
        edition = str(data.get("edition") or "")
        doc_kind = data.get("doc_kind")
        if doc_kind not in _DOC_KINDS:
            doc_kind = "regulation"
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
            doc_kind="regulation",
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
    proposals = _collect_proposals(llm, windows, warnings)

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


# ── SOP parsing ─────────────────────────────────────────
# The SOP's Word styles form a strict hierarchy ladder. Heading styles nest
# CNXL1 < CNXL3 < CNXL4 < CNXL5; the "*Body" (and "List Paragraph") styles are
# body-level content that attach as leaves one level below their heading tier.
# Each style maps to an integer depth; a paragraph's parent is the most recent
# paragraph at a strictly shallower depth (the synthetic root sits at depth 0).
_SOP_STYLE_LEVELS = {
    "CNXL1": 1,
    "CNXL1Body": 2,
    "CNXL3": 2,
    "CNXL3Body": 3,
    "List Paragraph": 3,
    "CNXL4": 3,
    "CNXL5": 4,
}

# Deterministic internal-reference extraction, scoped to the SOP's "References"
# section. The truth set expands the section's compressed doc-number list into
# full canonical ids (e.g. "MJV-CGP-10-0297, 0298, ..." -> "MJV-CGP-10-0297",
# "MJV-CGP-10-0298", ...) and keeps the fuller named forms ("CNX SPCC Plan",
# "Majorsville Cause & Effect Matrix"). References mentioned OUTSIDE that section
# (e.g. a P&ID note, a mole-sieve procedure in a step) are deliberately not
# collected, so scoping to the section is what keeps the set exact.

# A leading canonical doc id like "MJV-CGP-10-0297" — group 1 is the reusable
# prefix ("MJV-CGP-10-") that the rest of the comma list inherits.
_MJV_PREFIX_RE = re.compile(r"^(MJV-CGP-\d+-)")
# A single doc token in the list: digits with an optional trailing letter
# (0297, 0301B, 314A, ...).
_MJV_TOKEN_RE = re.compile(r"\b(\d+[A-Za-z]?)\b")
_MS_SWP_RE = re.compile(r"\bMS-SWP-\d+\b")
# Named references, captured with their qualifier (the leading word) so the
# canonical fuller form is produced.
_NAMED_REFERENCE_RES = (
    re.compile(r"\b\w+ Cause & Effect Matrix\b"),
    re.compile(r"\bVendor Manuals\b"),
    re.compile(r"\b\w+ SPCC Plan\b"),
)

_SOP_PROFILE_SYSTEM_PROMPT = """\
You read the text of a Standard Operating Procedure (SOP) for an industrial
facility and return a compact structured profile as JSON, of this exact shape:
{
  "jurisdiction": ["where the facility operates, using the vocabulary below"],
  "industry": "the single industry this SOP belongs to, e.g. 'midstream
                natural gas processing' ('' if unknown)",
  "activities": ["operational activities the SOP covers, as verb-phrases"],
  "substances": ["substances / materials handled"],
  "equipment": ["notable equipment / vessels / valves / systems involved"]
}

jurisdiction vocabulary (a list; include every level that applies): "US" for
the country, "US-<STATE>" for a US state (e.g. "US-PA" for Pennsylvania). If the
SOP indicates it operates in a US state, include BOTH the country ("US") and the
state ("US-<STATE>"). Use a country name for a non-US facility.

Field guidance:
- activities: verb-phrases naming the MAJOR operations the SOP covers (e.g.
  "blowdown", "valve line-up", "purging with natural gas or nitrogen"). Include
  control-system / SCADA-based operation, venting and draining operations, any
  permitted work (such as hot work) the SOP governs, any safety assessment the
  SOP requires before work (risk assessment / job safety analysis), and any
  start-up / shutdown procedures it performs or references — naming the referenced
  procedure id when the SOP cites one. Name the major operations, not every
  individual step or variant of one operation.
- equipment: concrete items with their tags where given, grouped into families.
  Cover the whole process — including safety / relief valves, inlet separation
  and collection vessels, local analogue gauges and sight/level glasses, and the
  control (SCADA) system itself — not only the main process vessels.
- substances: the process materials actually handled, including utility and fuel
  streams. Exclude ambient air constituents that are merely monitored for.
Use [] for lists you cannot determine and "" for unknown strings. Output ONLY
the JSON object, no prose and no code fences.
"""


def _references_section(paragraphs: list[ParagraphRecord]) -> list[ParagraphRecord]:
    """Return the paragraphs under the SOP's ``References`` heading.

    The section begins after the top-level (``CNXL1``) heading whose text is
    "References" and ends at the next ``CNXL1`` heading. Returns ``[]`` when no
    such heading exists.
    """
    section: list[ParagraphRecord] = []
    collecting = False
    for record in paragraphs:
        if record.style == "CNXL1":
            if collecting:
                break  # next top-level heading closes the References section
            if record.text.strip().rstrip(".").lower() == "references":
                collecting = True
            continue
        if collecting:
            section.append(record)
    return section


def _extract_internal_references(paragraphs: list[ParagraphRecord]) -> list[str]:
    """Collect distinct internal references from the SOP's References section.

    Deterministic (no LLM). The compressed doc-number list is expanded to full
    canonical ids by inheriting the prefix of its leading id; ``MS-SWP`` ids and
    named references (SPCC plan, cause & effect matrix, vendor manuals) are taken
    verbatim. First-seen order is preserved.
    """
    seen: dict[str, None] = {}
    for record in _references_section(paragraphs):
        text = record.text
        stripped = text.strip()
        prefix_match = _MJV_PREFIX_RE.match(stripped)
        if prefix_match:
            prefix = prefix_match.group(1)  # e.g. "MJV-CGP-10-"
            # Strip every full prefix, then re-attach it to each bare token so
            # "MJV-CGP-10-0297, 0298, 314A" -> the three full ids.
            body = stripped.replace(prefix, " ")
            for token in _MJV_TOKEN_RE.findall(body):
                seen.setdefault(prefix + token, None)
        for match in _MS_SWP_RE.findall(text):
            seen.setdefault(match, None)
        for pattern in _NAMED_REFERENCE_RES:
            for match in pattern.finditer(text):
                seen.setdefault(match.group(0), None)
    return list(seen)


def _build_sop_profile(
    llm: StructureLLM, full_text: str, internal_references: list[str]
) -> SopProfile:
    """Derive a :class:`SopProfile` from the SOP text.

    ``internal_references`` are computed deterministically upstream and passed
    through unchanged; the LLM only fills the descriptive facets. As for
    regulatory profiles, the call is repeated ``PROFILE_PASSES`` times and the
    list facets are unioned so one pass's omission does not lose an operation or
    a piece of equipment. Profiling is best-effort — any failure yields an
    empty-but-valid profile.
    """
    try:
        data = _run_profile_passes(llm, _SOP_PROFILE_SYSTEM_PROMPT, full_text)
        return SopProfile(
            jurisdiction=[str(x) for x in data.get("jurisdiction", [])],
            industry=str(data.get("industry") or ""),
            activities=[str(x) for x in data.get("activities", [])],
            substances=[str(x) for x in data.get("substances", [])],
            equipment=[str(x) for x in data.get("equipment", [])],
            internal_references=internal_references,
        )
    except Exception:  # noqa: BLE001 — profiling is best-effort, never fatal
        return SopProfile(
            jurisdiction=[],
            industry="",
            activities=[],
            substances=[],
            equipment=[],
            internal_references=internal_references,
        )


def parse_operating_procedure(
    path: Path, out_dir: Path | None = None
) -> OperatingProcedure:
    """Parse a Standard Operating Procedure DOCX into an ``OperatingProcedure``.

    The document's Word styles already encode its hierarchy, so unlike the
    regulatory path no LLM is used for structure. We read the ordered non-empty
    paragraphs verbatim, map each style to a depth, and rebuild the tree with a
    style-level stack: a paragraph's parent is the most recent paragraph at a
    strictly shallower depth. A synthetic document root (``S-001``, empty body,
    ``parent_id`` None) anchors the tree; content ids ``S-002``.. are assigned
    in document order, with ``order`` a 1-based running index. ``is_leaf`` and
    ``section_lineage`` are derived from the finished tree. Only the profile
    facets use one best-effort Haiku call.

    The parsed procedure is written as flat JSONL to ``out_dir`` when given,
    otherwise to the default :data:`PARSED_DIR`.
    """
    path = Path(path)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    doc_id = _slug(path.stem)

    extraction = extract_sop(path)

    root_id = "S-001"
    nodes: list[SopNode] = [
        SopNode(
            node_id=root_id,
            parent_id=None,
            section_lineage=[],
            body="",
            is_leaf=False,
            order=1,
        )
    ]

    warnings: list[str] = []
    unknown_styles: set[str] = set()

    # Build the tree with a depth stack seeded by the synthetic root at depth 0.
    # Each stack entry is (node_id, depth). A paragraph pops every entry at its
    # own depth or deeper, then attaches to whatever remains on top.
    stack: list[tuple[str, int]] = [(root_id, 0)]
    # node_id -> (parent_id, lineage) so is_leaf can be derived after the walk.
    parent_of: dict[str, str] = {}
    lineage_of: dict[str, list[str]] = {}

    for index, record in enumerate(extraction.paragraphs):
        node_id = f"S-{index + 2:03d}"
        order = index + 2
        level = _SOP_STYLE_LEVELS.get(record.style)
        if level is None:
            # Unknown style: treat as body content one level below the current
            # node so it lands as a leaf, and warn once per style name.
            unknown_styles.add(record.style)
            level = stack[-1][1] + 1

        while len(stack) > 1 and stack[-1][1] >= level:
            stack.pop()
        parent_id = stack[-1][0]
        parent_of[node_id] = parent_id
        lineage_of[node_id] = lineage_of.get(parent_id, []) + [parent_id]

        nodes.append(
            SopNode(
                node_id=node_id,
                parent_id=parent_id,
                section_lineage=lineage_of[node_id],
                body=record.text,
                is_leaf=True,  # provisional; corrected once the tree is known
                order=order,
            )
        )
        stack.append((node_id, level))

    # A node is internal iff some other node names it as parent.
    has_child = set(parent_of.values())
    nodes = [
        node.model_copy(update={"is_leaf": node.node_id not in has_child})
        for node in nodes
    ]

    for style in sorted(unknown_styles):
        warnings.append(f"unknown style {style!r}: attached as leaf content")
    if extraction.skipped_empty:
        warnings.append(
            f"skipped {extraction.skipped_empty} empty paragraph(s) "
            f"of {extraction.total_paragraphs} total"
        )

    full_text = "\n".join(record.text for record in extraction.paragraphs)
    internal_references = _extract_internal_references(extraction.paragraphs)
    profile = _build_sop_profile(StructureLLM(), full_text, internal_references)

    pages_total = (
        extraction.pages_total
        if extraction.pages_total is not None
        else extraction.total_paragraphs
    )
    parse_accounting = {
        "pages_total": pages_total,
        "pages_parsed": pages_total,
        "warnings": warnings,
    }

    procedure = OperatingProcedure(
        doc_id=doc_id,
        source_path=str(path),
        file_hash=file_hash,
        title=extraction.title or doc_id,
        parse_accounting=parse_accounting,
        profile=profile,
        nodes=nodes,
    )

    write_parsed_document(procedure, out_dir if out_dir is not None else PARSED_DIR)
    return procedure
