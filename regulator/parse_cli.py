"""Command-line entry point for parsing a single regulatory PDF.

This thin module holds the CLI (argument parsing, flag resolution, and the
``python -m regulator.parse_cli`` guard) so it is NOT imported by
:mod:`regulator.__init__`. Keeping it out of the package import graph means
``python -m regulator.parse_cli`` executes cleanly without runpy's
"found in sys.modules after import of package" RuntimeWarning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from regulator.parsing import PARSED_DIR, parse_regulatory_document

# Where the source regulatory PDFs live; the CLI resolves a flag against these.
REGULATIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "regulations"


def _available_regulations() -> list[Path]:
    """All regulatory PDFs available to the CLI, sorted by filename."""
    return sorted(REGULATIONS_DIR.glob("*.pdf"))


def resolve_regulation_flag(flag: str) -> Path:
    """Resolve a DOC flag to exactly one regulatory PDF.

    Matches ``flag`` case-insensitively as a substring of the PDF filenames in
    :data:`REGULATIONS_DIR`. Raises :class:`SystemExit` with a helpful message
    on zero matches (lists what is available) or more than one match (lists the
    ambiguous candidates).
    """
    candidates = _available_regulations()
    needle = flag.strip().lower()
    matches = [p for p in candidates if needle in p.name.lower()]

    if len(matches) == 1:
        return matches[0]

    available = "\n".join(f"  - {p.name}" for p in candidates) or "  (none found)"
    if not matches:
        raise SystemExit(
            f"No regulatory PDF matches DOC='{flag}'.\nAvailable documents:\n"
            f"{available}"
        )
    ambiguous = "\n".join(f"  - {p.name}" for p in matches)
    raise SystemExit(
        f"DOC='{flag}' is ambiguous; it matches {len(matches)} documents:\n"
        f"{ambiguous}\nPlease narrow the flag."
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: parse one regulatory PDF selected by a substring flag.

    Resolves the flag to a single PDF, parses it, writes JSONL to ``--out``
    (default :data:`PARSED_DIR`), and prints where the output went plus the
    node count.
    """
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m regulator.parse_cli",
        description="Parse one regulatory PDF selected by a case-insensitive "
        "substring flag against the filenames in data/regulations/.",
    )
    parser.add_argument("flag", help="substring selecting one regulatory PDF")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for the JSONL output (default: data/parsed/)",
    )
    args = parser.parse_args(argv)

    # Resolve the flag BEFORE any environment / LLM work so error paths never
    # make live LLM calls.
    path = resolve_regulation_flag(args.flag)

    load_dotenv()
    document = parse_regulatory_document(path, out_dir=args.out)

    out_dir = args.out if args.out is not None else PARSED_DIR
    out_path = out_dir / f"{document.doc_id}.jsonl"
    print(f"Parsed '{path.name}' -> {out_path} ({len(document.nodes)} nodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
