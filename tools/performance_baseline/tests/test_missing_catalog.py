"""Reject incomplete or mismatched collectors before GPU measurements."""

from pathlib import Path
import sys
import tempfile
import unittest

BASELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE))
from run_benchmark_nvidia_missing import (  # noqa: E402
    DEFAULT_REPORT,
    requested_cases,
    select_catalogs,
)


class MissingCatalogTests(unittest.TestCase):
    def test_document_inventory(self):
        requested = requested_cases(DEFAULT_REPORT)
        self.assertEqual(len(requested), 371)
        self.assertEqual(len({op for op, _ in requested}), 22)
        self.assertEqual(sum(op == "batchnorm" for op, _ in requested), 24)
        self.assertEqual(sum(op == "reduction" for op, _ in requested), 20)

    def test_case_name_alone_does_not_match(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            select_catalogs(
                [("identity", "same_name")],
                {"core": [{"op": "relu", "case_name": "same_name"}]},
            )

    def test_duplicate_collectors_are_rejected(self):
        row = {"op": "identity", "case_name": "copy"}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_catalogs(
                [("identity", "copy")], {"core": [row], "aux": [row]}
            )

    def test_duplicate_requested_cases_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            report.write_text(
                "## 附录一\n### identity 2 个\n```text\ncopy\ncopy\n```\n"
                "## 附录二\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                requested_cases(report)


if __name__ == "__main__":
    unittest.main()
