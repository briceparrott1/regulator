"""Profile-only extraction for regulatory documents and the SOP.

This module produces just the *identity* and *profile* of a document — the
fields document-level applicability / matching reasons over — WITHOUT running
any of the expensive structure-parsing passes (no LLM structure proposals over
page windows, no deterministic SOP style atomization). One cheap Haiku profile
call per document is all it costs, so applicability work can iterate on
profiles quickly and repeatedly.

The output mirrors :mod:`regulator.parse_store` exactly: each document is
written to ``data/profiles/<doc_id>.jsonl`` as two type-tagged records — a slim
``{"type": "document", ...}`` header (identity fields; ``parse_accounting``
records that structure was NOT parsed) followed by the ``{"type": "profile",
...}`` record. We build a real :class:`RegulatoryDocument` /
:class:`OperatingProcedure` with an empty ``nodes`` list and hand it to the same
``write_parsed_document`` writer, so downstream loading is uniform with the full
parse artifacts in ``data/parsed/``.

Reuse note: the profile-building and identity machinery is imported wholesale
from :mod:`regulator.parsing` (including its private helpers ``_slug``,
``_build_profile``, ``_build_sop_profile``, ``_extract_internal_references``).
Importing them — rather than re-implementing — guarantees a profile extracted
here is byte-for-byte identical to the one the full parse would produce; there
is deliberately no second copy of that logic to drift.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Imports reused from sibling modules. Two things are deliberate here:
#   1. The underscore-prefixed names from regulator.parsing (_build_profile,
#      _build_sop_profile, _extract_internal_references, _slug) are imported on
#      purpose — replicating them would risk the profile-only output drifting
#      from the full parse output, which is exactly what this module must avoid.
#      parsing.py is owned by another workstream and is not edited here.
#   2. resolve_regulation_flag / _available_regulations come from parse_cli
#      (a stable module) so flag→file resolution is not duplicated.
from regulator.llm import StructureLLM
from regulator.models import OperatingProcedure, RegulatoryDocument
from regulator.parse_cli import _available_regulations, resolve_regulation_flag
from regulator.parse_store import write_parsed_document
from regulator.parsing import (
    _build_profile,
    _build_sop_profile,
    _extract_internal_references,
    _slug,
)
from regulator.pdf_extract import extract_pages
from regulator.sop_extract import extract_sop

# Where profile-only artifacts are persisted. This is the directory
# applicability / document-level matching iterates on.
PROFILES_DIR = Path(__file__).resolve().parent.parent / "data" / "profiles"

# The single SOP the pipeline operates on (same source main.py / make parse-sop
# use).
SOP_PATH = Path(__file__).resolve().parent.parent / "data" / "sop" / "original.docx"


def extract_regulatory_profile(path: Path) -> RegulatoryDocument:
    """Extract identity + profile for one regulatory PDF, WITHOUT structure.

    Runs the same front-matter extraction and profile LLM call that
    ``parse_regulatory_document`` uses, but skips every structure pass. The
    returned :class:`RegulatoryDocument` carries the document-identity fields
    (``doc_id``, ``source_path``, ``file_hash``, ``title``, ``framework``,
    ``edition``) and the profile, with an empty ``nodes`` list and a
    ``parse_accounting`` that flags structure as not parsed.
    """
    path = Path(path)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    doc_id = _slug(path.stem)

    pages = extract_pages(path)
    llm = StructureLLM()
    profile, title, framework, edition = _build_profile(llm, pages, doc_id)

    parse_accounting = {
        "pages_total": len(pages),
        "pages_parsed": 0,
        "structure_parsed": False,
        "warnings": ["structure passes skipped: profile-only extraction"],
    }

    return RegulatoryDocument(
        doc_id=doc_id,
        source_path=str(path),
        file_hash=file_hash,
        title=title,
        framework=framework,
        edition=edition,
        parse_accounting=parse_accounting,
        profile=profile,
        nodes=[],
    )


def extract_sop_profile(path: Path = SOP_PATH) -> OperatingProcedure:
    """Extract identity + profile for the SOP DOCX, WITHOUT structure.

    Reads the SOP paragraphs (verbatim text only), computes the deterministic
    ``internal_references``, and runs the same profile LLM call that
    ``parse_operating_procedure`` uses — but skips the style→depth atomization
    that builds the node tree. The returned :class:`OperatingProcedure` carries
    the identity fields and profile with an empty ``nodes`` list.
    """
    path = Path(path)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    doc_id = _slug(path.stem)

    extraction = extract_sop(path)
    full_text = "\n".join(record.text for record in extraction.paragraphs)
    internal_references = _extract_internal_references(full_text)
    profile = _build_sop_profile(StructureLLM(), full_text, internal_references)

    pages_total = (
        extraction.pages_total
        if extraction.pages_total is not None
        else extraction.total_paragraphs
    )
    parse_accounting = {
        "pages_total": pages_total,
        "pages_parsed": 0,
        "structure_parsed": False,
        "warnings": ["structure atomization skipped: profile-only extraction"],
    }

    return OperatingProcedure(
        doc_id=doc_id,
        source_path=str(path),
        file_hash=file_hash,
        title=extraction.title or doc_id,
        parse_accounting=parse_accounting,
        profile=profile,
        nodes=[],
    )


def write_profile(
    doc: RegulatoryDocument | OperatingProcedure, out_dir: Path = PROFILES_DIR
) -> Path:
    """Write a profile-only document to ``out_dir/<doc_id>.jsonl``.

    Delegates to :func:`regulator.parse_store.write_parsed_document`, which emits
    a ``document`` record then a ``profile`` record then one record per node.
    Because ``doc.nodes`` is empty for profile-only extraction, exactly the two
    header/profile records are written — the shape applicability loading expects.
    """
    return write_parsed_document(doc, out_dir)


# ── One-line summaries for the CLI ──────────────────────
def _reg_summary(doc: RegulatoryDocument) -> str:
    """Compact one-line description of a regulatory profile."""
    profile = doc.profile
    framework = doc.framework or "?"
    return (
        f"{framework} · {profile.doc_kind} · "
        f"{len(profile.activities)} activities, "
        f"{len(profile.substances)} substances, "
        f"{len(profile.industries)} industries"
    )


def _sop_summary(procedure: OperatingProcedure) -> str:
    """Compact one-line description of an SOP profile."""
    profile = procedure.profile
    industry = profile.industry or "?"
    return (
        f"industry={industry} · "
        f"{len(profile.activities)} activities, "
        f"{len(profile.substances)} substances, "
        f"{len(profile.internal_references)} internal refs"
    )


def _profile_one_regulation(path: Path, out_dir: Path) -> None:
    """Extract, write, and report one regulatory PDF's profile."""
    document = extract_regulatory_profile(path)
    out_path = write_profile(document, out_dir)
    print(f"{document.doc_id}")
    print(f"  {_reg_summary(document)}")
    print(f"  -> {out_path}")


def _profile_sop(path: Path, out_dir: Path) -> None:
    """Extract, write, and report the SOP's profile."""
    procedure = extract_sop_profile(path)
    out_path = write_profile(procedure, out_dir)
    print(f"{procedure.doc_id}")
    print(f"  {_sop_summary(procedure)}")
    print(f"  -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    """CLI: extract profile-only artifacts into ``data/profiles/``.

    One of three modes: a substring ``flag`` selects a single regulatory PDF,
    ``--sop`` profiles the SOP, and ``--all`` profiles all regulatory PDFs then
    the SOP sequentially. All target resolution happens BEFORE any environment /
    LLM work so error paths (bad flag, ambiguous flag, missing selection) never
    make a live LLM call.
    """
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m regulator.profiles",
        description="Extract profile-only artifacts (identity + profile, no "
        "structure) into data/profiles/. Applicability / document-level "
        "matching iterates on these.",
    )
    parser.add_argument(
        "flag",
        nargs="?",
        help="case-insensitive substring selecting one regulatory PDF",
    )
    parser.add_argument(
        "--sop", action="store_true", help="profile data/sop/original.docx"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="profile all regulatory PDFs then the SOP, sequentially",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for the JSONL output (default: data/profiles/)",
    )
    args = parser.parse_args(argv)

    out_dir = args.out if args.out is not None else PROFILES_DIR

    # Resolve every target up front (raises SystemExit on a bad/ambiguous flag or
    # no selection) so no LLM call happens on an error path.
    reg_paths: list[Path] = []
    do_sop = False
    if args.all:
        reg_paths = _available_regulations()
        do_sop = True
    elif args.sop:
        do_sop = True
    elif args.flag:
        reg_paths = [resolve_regulation_flag(args.flag)]
    else:
        parser.error("provide a DOC substring flag, --sop, or --all")

    load_dotenv()
    for path in reg_paths:
        _profile_one_regulation(path, out_dir)
    if do_sop:
        _profile_sop(SOP_PATH, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
