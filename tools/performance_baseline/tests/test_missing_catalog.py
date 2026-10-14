"""Reject incomplete or mismatched collectors before GPU measurements."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

BASELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE))
from run_benchmark_nvidia_missing import (  # noqa: E402
    DEFAULT_REPORT,
    MEASUREMENT_ID,
    requested_cases,
    select_catalogs,
)


class MissingCatalogTests(unittest.TestCase):
    def test_baseline_inventory(self):
        self.assertEqual(DEFAULT_REPORT.name, "performance-baseline.json")
        requested = requested_cases(DEFAULT_REPORT)
        self.assertEqual(len(requested), 371)
        self.assertEqual(len({op for op, _ in requested}), 22)
        self.assertEqual(sum(op == "batchnorm" for op, _ in requested), 24)
        self.assertEqual(sum(op == "reduction" for op, _ in requested), 20)

    def test_json_inventory_selects_only_the_measurement_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "baseline.json"
            report.write_text(json.dumps({"reference": {
                "cases": {
                    "copy": {"op": "identity", "measurement_id": MEASUREMENT_ID},
                    "sum": {"op": "reduction", "measurement_id": MEASUREMENT_ID},
                    "old": {"op": "identity", "measurement_id": "another-run"},
                    "unlabeled": {"op": "relu"},
                },
                "unavailable_cases": {
                    "excluded": {"op": "identity", "measurement_id": MEASUREMENT_ID},
                },
            }}))
            self.assertEqual(requested_cases(report),
                             [("identity", "copy"), ("reduction", "sum")])

    def test_malformed_json_inventory_is_rejected(self):
        documents = [None, {}, {"reference": {"cases": []}},
                     {"reference": {"cases": {"copy": None}}}]
        for op in (None, "", " ", ["identity"]):
            documents.append({"reference": {"cases": {
                "copy": {"op": op, "measurement_id": MEASUREMENT_ID},
            }}})
        documents.append({"reference": {"cases": {
            "": {"op": "identity", "measurement_id": MEASUREMENT_ID},
        }}})
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "baseline.json"
            for document in documents:
                with self.subTest(document=document):
                    report.write_text(json.dumps(document))
                    with self.assertRaises(ValueError):
                        requested_cases(report)

    def test_json_inventory_requires_cases_from_the_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "baseline.json"
            report.write_text(json.dumps({"reference": {"cases": {
                "copy": {"op": "identity", "measurement_id": "another-run"},
            }}}))
            with self.assertRaisesRegex(ValueError, "empty"):
                requested_cases(report)

    def test_explicit_markdown_inventory_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            report.write_text(
                "## 附录一\n### identity 1 个\n```text\ncopy\n```\n"
                "### reduction 1 个\n```text\nsum\n```\n"
                "## 附录二\n### relu 1 个\n```text\nexcluded\n```\n"
            )
            self.assertEqual(requested_cases(report),
                             [("identity", "copy"), ("reduction", "sum")])

    def test_malformed_markdown_inventory_is_rejected(self):
        reports = (
            "# Missing appendix\n",
            "## 附录一\n## 附录二\n",
            "## 附录一\n```text\ncopy\n```\n",
            "## 附录一\n### identity\n```text\ncopy\n```\n"
            "### relu\n```text\ncopy\n```\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            for contents in reports:
                with self.subTest(contents=contents):
                    report.write_text(contents)
                    with self.assertRaises(ValueError):
                        requested_cases(report)

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
