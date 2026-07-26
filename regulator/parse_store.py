"""Persistence for parsed documents (flat JSONL).

A parsed :class:`RegulatoryDocument` or :class:`OperatingProcedure` is written
to ``<dir>/<doc_id>.jsonl`` as one type-tagged record per line, in a fixed
order: the document header (without its nested profile/nodes), then the
profile, then each node in document order. This keeps the on-disk form
grep-friendly and streamable. Both document kinds share the same record shape,
so a single writer handles either. No caching or load logic lives here.
"""

from __future__ import annotations

import json
from pathlib import Path

from regulator.models import OperatingProcedure, RegulatoryDocument


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
