"""Applicability filtering: narrow regulations and nodes to the SOP.

Stub implementations. These functions will decide which regulatory documents
and which individual requirements are actually relevant to a given SOP.
"""

from __future__ import annotations

from regulator.models import OperatingProcedure, RegulatoryDocument, RegulatoryNode


def get_applicable_regulatory_procedures(
    sop: OperatingProcedure,
    docs: list[RegulatoryDocument],
) -> list[RegulatoryDocument]:
    """Return the regulatory documents applicable to ``sop``.

    Will (once implemented) score each document against the SOP and drop those
    that are clearly irrelevant. For now it returns ``docs`` unchanged.
    """
    return docs


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
