"""Regression tests for company-paper discovery boundaries."""

import sys
import unittest
from datetime import date
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from paper_pipeline.core.date_window import DateWindow

pyalex_stub = ModuleType("pyalex")
pyalex_stub.Institutions = Mock()
pyalex_stub.Works = Mock()
pyalex_stub.config = SimpleNamespace(email=None)
with patch.dict(sys.modules, {"pyalex": pyalex_stub}):
    from paper_pipeline.discovery import openalex_2026_audit

fetch_org_works = openalex_2026_audit.fetch_org_works


class TestDiscoveryGuards(unittest.TestCase):
    def test_meta_affiliation_rejects_bare_fair_acronym(self) -> None:
        with patch.dict(sys.modules, {"arxiv": Mock()}):
            from paper_pipeline.discovery.arxiv_org_search import ORG_SEARCHES

        meta_pattern = next(pattern for label, _, pattern in ORG_SEARCHES if label == "Meta / FAIR")

        self.assertIsNone(meta_pattern.search("GSI/FAIR, Darmstadt, Germany"))
        self.assertIsNotNone(meta_pattern.search("Meta AI, Menlo Park, CA"))
        self.assertIsNotNone(meta_pattern.search("Facebook AI Research"))

    @patch.object(openalex_2026_audit, "Works")
    def test_openalex_query_failure_is_explicit(self, works: Mock) -> None:
        works.return_value.filter.side_effect = RuntimeError("429 Too Many Requests")
        publication_window = DateWindow(since=date(2026, 7, 23), through=date(2026, 7, 27))

        result = fetch_org_works("I123", publication_window)

        self.assertEqual(result.works, [])
        self.assertFalse(result.truncated)
        self.assertEqual(result.error, "429 Too Many Requests")


if __name__ == "__main__":
    unittest.main()
