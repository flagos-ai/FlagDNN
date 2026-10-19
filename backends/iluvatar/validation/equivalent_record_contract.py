#!/usr/bin/env python3
"""Reject forged equivalent records and preserve independent coverage gates."""
import copy
import importlib.util
import pathlib
import unittest

path = pathlib.Path(__file__).with_name("run_tests_adapter.py")
spec = importlib.util.spec_from_file_location("iluvatar_adapter", path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def fixture():
    return dict(
        schema_version=4, kind="steady_state", provider="flagdnn",
        case="div_perf_fp32_1x1x1024_by_1x1x1024", unit="us",
        median=2.0, p90=2.0, samples=[2.0] * 20,
        equivalent_baseline=dict(
            provider="nvidia_equivalent", platform="nvidia", target_platform="iluvatar",
            timing_profile="native", scale_metric="vector_fp32",
            reference_time_us=1.0, reference_peak=4.0, target_peak=2.0,
            min_relative_efficiency=0.8, equivalent_time_us=2.0, limit_us=2.5,
            relative_efficiency=1.0, traffic_ratio=1.0, reference_validation="cpu_full",
            passed=True))


class RecordContract(unittest.TestCase):
    def test_formula_identity_and_no_fictitious_provider(self):
        record = fixture()
        adapter.validate_benchmark_record(record)
        self.assertEqual([], adapter.validate_iluvatar_benchmark_pairs(
            {record["case"]: {"flagdnn": record}}, "div", ["div"]))
        for field, value in {
            "target_peak": 0, "reference_peak": float("nan"),
            "target_platform": "hygon", "relative_efficiency": 4,
            "traffic_ratio": 2, "passed": False,
        }.items():
            altered = copy.deepcopy(record)
            altered["equivalent_baseline"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapter.validate_benchmark_record(altered)

    def test_explicit_performance_gate(self):
        record = fixture()
        record.update(median=4.0, p90=4.0, samples=[4.0] * 20)
        record["equivalent_baseline"].update(relative_efficiency=0.5, passed=False)
        adapter.validate_benchmark_record(record)
        results = {"div": {"benchmark": {"status": "passed", "case_accounting": {},
                   "records": {record["case"]: {"flagdnn": record}}}}}
        self.assertTrue(adapter.iluvatar_speedup_summary(results, None)["gate_passed"])
        self.assertFalse(adapter.iluvatar_speedup_summary(results, 0.1)["gate_passed"])


if __name__ == "__main__":
    unittest.main()
