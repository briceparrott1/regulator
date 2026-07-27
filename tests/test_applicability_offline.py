"""Offline unit tests for the document-level applicability stage.

Every test here runs WITHOUT an API key and without touching the network: the
LLM is stubbed, and the only real files read are the frozen truth case and
temporary JSONL written by the test itself.

Run from the repo root::

    .venv/bin/python -m unittest tests.test_applicability_offline
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from regulator.applicability import (
    _merge_ranges,
    judge_document_applicability,
    normalize_quote,
    quote_is_verbatim,
    scope_slice,
    should_audit,
)
from regulator.evaluation import (
    _SOP_ID_ALIASES,
    _is_applicability_record,
    _read_jsonl,
)
from regulator.models import (
    DocApplicabilityVerdict,
    OperatingProcedure,
    RegProfile,
    RegulatoryDocument,
    SopProfile,
)
from regulator.parse_store import read_parsed_document, write_parsed_document
from regulator.pdf_extract import PageRecord

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRUTH_PATH = PROJECT_ROOT / "tests" / "cases" / "applicability_truth.jsonl"


def _page(number: int, text: str) -> PageRecord:
    """A PageRecord with its lines derived from ``text``, as the extractor does."""
    return PageRecord(page_number=number, text=text, lines=text.splitlines())


def _verdict(verdict: str, **overrides: Any) -> DocApplicabilityVerdict:
    fields: dict[str, Any] = {
        "doc_id": "reg-x",
        "sop_id": "original",
        "verdict": verdict,
        "confidence": "high",
    }
    fields.update(overrides)
    return DocApplicabilityVerdict(**fields)


class _StubLLM:
    """Stand-in for StructureLLM: returns canned JSON or raises."""

    def __init__(self, payload: dict[str, Any] | None = None, error: bool = False):
        self._payload = payload or {}
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def propose_json(self, system_prompt: str, user_prompt: str, **_: Any) -> dict:
        self.calls.append((system_prompt, user_prompt))
        if self._error:
            raise RuntimeError("boom")
        return self._payload


def _reg_document(doc_id: str = "reg-x") -> RegulatoryDocument:
    """A profile-only regulatory document whose source PDF does not exist."""
    return RegulatoryDocument(
        doc_id=doc_id,
        source_path="/nonexistent/reg-x.pdf",
        file_hash="0" * 8,
        title="A Standard For Things",
        framework="ACME",
        edition="2020",
        parse_accounting={"pages_total": 0, "pages_parsed": 0, "warnings": []},
        profile=RegProfile(
            doc_id=doc_id,
            jurisdiction=["US"],
            doc_kind="industry_standard",
            activities=["doing things"],
            substances=[],
            equipment=["things"],
            industries=["thing-making"],
            addressee_types=["manufacturer"],
        ),
        nodes=[],
    )


def _sop_document() -> OperatingProcedure:
    return OperatingProcedure(
        doc_id="original",
        source_path="/nonexistent/original.docx",
        file_hash="1" * 8,
        title="Initial Purge of Skid",
        parse_accounting={"pages_total": 0, "pages_parsed": 0, "warnings": []},
        profile=SopProfile(
            jurisdiction=["US", "US-PA"],
            industry="midstream natural gas processing",
            activities=["purging"],
            substances=["natural gas"],
            equipment=["V-750"],
            internal_references=[],
        ),
        nodes=[],
    )


class ShouldAuditTest(unittest.TestCase):
    def test_applicable_and_conditional_continue(self) -> None:
        self.assertTrue(should_audit(_verdict("applicable")))
        self.assertTrue(should_audit(_verdict("conditional")))

    def test_not_applicable_is_dropped(self) -> None:
        self.assertFalse(should_audit(_verdict("not_applicable")))


class VerdictModelTest(unittest.TestCase):
    def test_defaults_and_frozen(self) -> None:
        verdict = _verdict("applicable")
        self.assertIsNone(verdict.trigger_quote)
        self.assertEqual(verdict.reasons, [])
        self.assertEqual(verdict.missing_facts, [])
        with self.assertRaises(ValidationError):
            verdict.verdict = "conditional"  # type: ignore[misc]

    def test_vocabularies_are_enforced(self) -> None:
        with self.assertRaises(ValidationError):
            _verdict("maybe")
        with self.assertRaises(ValidationError):
            _verdict("applicable", confidence="very high")


class QuoteCheckTest(unittest.TestCase):
    def test_normalize_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_quote("  a\n b\t c "), "a b c")

    def test_line_wrapped_quote_is_verbatim(self) -> None:
        source = "This standard applies to\nvehicle-mounted aerial devices."
        self.assertTrue(
            quote_is_verbatim("This standard applies to vehicle-mounted", source)
        )

    def test_absent_and_empty_quotes(self) -> None:
        self.assertFalse(quote_is_verbatim("not in here", "some other text"))
        self.assertFalse(quote_is_verbatim(None, "some other text"))


class ScopeSliceTest(unittest.TestCase):
    def test_empty_pages(self) -> None:
        self.assertEqual(scope_slice([]), "")

    def test_front_matter_and_scope_passage_selected(self) -> None:
        pages = [
            _page(1, "ACME STANDARD 12\nSecond Edition"),
            _page(2, "Foreword\nPublished by ACME."),
            _page(3, "Filler line one.\nFiller line two."),
            _page(
                4,
                "1.1 Scope\nThis standard applies to widget presses.\n"
                "It does not apply to hand tools.",
            ),
        ]
        slice_text = scope_slice(pages)
        self.assertIn("ACME STANDARD 12", slice_text)  # front matter
        self.assertIn("This standard applies to widget presses.", slice_text)
        self.assertNotIn("Filler line one.", slice_text)  # unrelated page dropped
        self.assertIn("[PAGE 4", slice_text)  # page attribution kept

    def test_slice_is_deterministic(self) -> None:
        pages = [_page(1, "Cover"), _page(2, "x"), _page(3, "2 Applicability\nA rule.")]
        self.assertEqual(scope_slice(pages), scope_slice(pages))


class MergeRangesTest(unittest.TestCase):
    def test_overlapping_windows_merge(self) -> None:
        self.assertEqual(_merge_ranges([(0, 5), (3, 8), (20, 22)]), [(0, 8), (20, 22)])


class JudgeDocumentApplicabilityTest(unittest.TestCase):
    def test_parses_a_well_formed_answer(self) -> None:
        llm = _StubLLM(
            {
                "verdict": "conditional",
                "trigger_quote": "This standard applies to widget presses.",
                "reasons": ["subject matter overlaps", " "],
                "missing_facts": ["press tonnage"],
                "confidence": "medium",
            }
        )
        verdict = judge_document_applicability(
            _sop_document(), _reg_document(), llm=llm
        )
        self.assertEqual(verdict.verdict, "conditional")
        self.assertEqual(verdict.sop_id, "original")
        self.assertEqual(verdict.doc_id, "reg-x")
        self.assertEqual(verdict.reasons, ["subject matter overlaps"])  # blanks dropped
        self.assertEqual(verdict.missing_facts, ["press tonnage"])
        self.assertEqual(verdict.confidence, "medium")

    def test_unknown_vocabulary_falls_back(self) -> None:
        llm = _StubLLM({"verdict": "probably", "confidence": "certain"})
        verdict = judge_document_applicability(
            _sop_document(), _reg_document(), llm=llm
        )
        self.assertEqual(verdict.verdict, "conditional")
        self.assertEqual(verdict.confidence, "low")
        self.assertIsNone(verdict.trigger_quote)

    def test_call_failure_fails_open(self) -> None:
        verdict = judge_document_applicability(
            _sop_document(), _reg_document(), llm=_StubLLM(error=True)
        )
        self.assertEqual(verdict.verdict, "conditional")
        self.assertTrue(should_audit(verdict))
        self.assertIn("RuntimeError", verdict.reasons[0])

    def test_prompt_carries_both_profiles(self) -> None:
        llm = _StubLLM({"verdict": "applicable", "confidence": "high"})
        judge_document_applicability(_sop_document(), _reg_document(), llm=llm)
        _, user_prompt = llm.calls[0]
        self.assertIn("midstream natural gas processing", user_prompt)
        self.assertIn("A Standard For Things", user_prompt)


class ProfileArtifactRoundTripTest(unittest.TestCase):
    def test_regulatory_and_sop_artifacts_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            reg_path = write_parsed_document(_reg_document(), out)
            sop_path = write_parsed_document(_sop_document(), out)

            reg = read_parsed_document(reg_path)
            self.assertIsInstance(reg, RegulatoryDocument)
            self.assertEqual(reg.profile.doc_kind, "industry_standard")
            self.assertEqual(reg.nodes, [])

            sop = read_parsed_document(sop_path)
            self.assertIsInstance(sop, OperatingProcedure)
            self.assertEqual(sop.profile.industry, "midstream natural gas processing")

    def test_stale_artifact_raises_a_helpful_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg-stale.jsonl"
            document = {"type": "document", "doc_id": "reg-stale", "framework": "ACME"}
            profile = {"type": "profile", "doc_id": "reg-stale", "doc_kind": "standard"}
            path.write_text(
                json.dumps(document) + "\n" + json.dumps(profile) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as caught:
                read_parsed_document(path)
            self.assertIn("re-generate", str(caught.exception))


class TruthCaseLoadingTest(unittest.TestCase):
    def test_truth_file_is_recognized(self) -> None:
        records = _read_jsonl(TRUTH_PATH)
        self.assertTrue(records)
        self.assertTrue(all(_is_applicability_record(r) for r in records))

    def test_sop_alias_resolves_to_our_slug(self) -> None:
        records = _read_jsonl(TRUTH_PATH)
        for record in records:
            self.assertEqual(_SOP_ID_ALIASES.get(record["sop_id"]), "original")

    def test_non_applicability_jsonl_is_not_claimed(self) -> None:
        self.assertFalse(_is_applicability_record({"node_id": "S-002"}))


if __name__ == "__main__":
    unittest.main()
