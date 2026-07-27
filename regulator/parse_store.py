"""Persistence for parsed documents (flat JSONL).

A parsed :class:`RegulatoryDocument` or :class:`OperatingProcedure` is written
to ``<dir>/<doc_id>.jsonl`` as one type-tagged record per line, in a fixed
order: the document header (without its nested profile/nodes), then the
profile, then each node in document order. This keeps the on-disk form
grep-friendly and streamable. Both document kinds share the same record shape,
so a single writer handles either.

:func:`read_parsed_document` is the exact inverse of the writer: it rebuilds the
model from those records. Which of the two document kinds a file holds is
inferred from the records themselves (see :func:`_is_regulatory`), so callers
need not know in advance. No caching lives here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from regulator.models import (
    OperatingProcedure,
    RegProfile,
    RegulatoryDocument,
    RegulatoryNode,
    SopNode,
    SopProfile,
)


def write_parsed_document(
    doc: RegulatoryDocument | OperatingProcedure, path: Path
) -> Path:
    """Write ``doc`` as flat JSONL to ``path/<doc_id>.jsonl``.

    Accepts either a regulatory document or an operating procedure; both expose
    ``doc_id``, ``profile`` and ``nodes`` with the same record layout. Returns
    the path written. Parent directories are created as needed.
    """
    path.mkdir(parents=True, exist_ok=True)
    out_path = path / f"{doc.doc_id}.jsonl"

    document_record = doc.model_dump(exclude={"profile", "nodes"})
    document_record["type"] = "document"

    profile_record = doc.profile.model_dump()
    profile_record["type"] = "profile"

    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(document_record, ensure_ascii=False) + "\n")
        handle.write(json.dumps(profile_record, ensure_ascii=False) + "\n")
        for node in doc.nodes:
            node_record = node.model_dump()
            node_record["type"] = "node"
            handle.write(json.dumps(node_record, ensure_ascii=False) + "\n")

    return out_path


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file into records, ignoring blank lines."""
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _without_type(record: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``record`` without the on-disk ``type`` tag."""
    return {key: value for key, value in record.items() if key != "type"}


def _is_regulatory(document_record: dict[str, Any]) -> bool:
    """True when a document header describes a regulatory document.

    ``framework`` and ``edition`` exist only on :class:`RegulatoryDocument`; an
    :class:`OperatingProcedure` header carries neither.
    """
    return "framework" in document_record


def read_parsed_document(path: Path) -> RegulatoryDocument | OperatingProcedure:
    """Read a JSONL artifact written by :func:`write_parsed_document`.

    Returns a :class:`RegulatoryDocument` or an :class:`OperatingProcedure`
    depending on what the file holds. Profile-only artifacts (``data/profiles/``)
    carry no node records and come back with an empty ``nodes`` list, which is
    exactly what document-level applicability needs.

    Raises :class:`ValueError` when the file is missing its header/profile
    records or when its records no longer satisfy the current schema — the
    latter happens when an artifact predates a schema change and simply needs
    re-generating.
    """
    records = _read_records(path)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_type.setdefault(str(record.get("type")), []).append(record)

    if not by_type.get("document") or not by_type.get("profile"):
        raise ValueError(f"{path} is missing its document and/or profile record")

    document_record = _without_type(by_type["document"][0])
    profile_record = _without_type(by_type["profile"][0])
    node_records = [_without_type(r) for r in by_type.get("node", [])]

    try:
        if _is_regulatory(document_record):
            return RegulatoryDocument(
                **document_record,
                profile=RegProfile(**profile_record),
                nodes=[RegulatoryNode(**r) for r in node_records],
            )
        return OperatingProcedure(
            **document_record,
            profile=SopProfile(**profile_record),
            nodes=[SopNode(**r) for r in node_records],
        )
    except ValidationError as exc:
        raise ValueError(
            f"{path} does not match the current schema — re-generate it "
            f"(e.g. `make profiles`): {exc}"
        ) from exc
