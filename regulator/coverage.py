"""Coverage analysis: check the SOP against applicable regulations.

Stub implementation. Will (once implemented) evaluate whether the SOP satisfies
each applicable regulatory requirement.
"""

from __future__ import annotations

from regulator.models import CoverageVerdict, OperatingProcedure, RegulatoryDocument


def get_batch_coverage(
    sop: OperatingProcedure,
    docs: list[RegulatoryDocument],
) -> list[CoverageVerdict]:
    """Check ``sop`` against the applicable regulatory documents.

    Will (once implemented) gather the applicable atoms from ``docs`` and, for
    each, determine whether the SOP covers it, producing one
    :class:`CoverageVerdict` per atom. For now it returns an empty list.
    """
    return []
