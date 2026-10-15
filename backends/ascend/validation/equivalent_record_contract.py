"""Host-only checks for truthful Ascend equivalent timing reports."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load(
    "ascend_adapter", Path(__file__).with_name("run_tests_adapter.py")
)
runner = load("run_tests", ROOT / "tools/run_tests.py")


def fixture():
    metric = dict(median=2.0, p90=2.0, samples=[2.0] * 20)
    return dict(
        schema_version=6,
        kind="steady_state",
        provider="flagdnn",
        case="identity_fp8_e4m3_2x3x4",
        environment=dict(
            soc_fingerprint="ascend_test",
            cann_package_version="9.0.0",
            ascendcl_build_id="test",
            runtime_build_id="test",
        ),
        provider_identity=dict(
            libtriton_jit_sha256="a" * 64,
            compiler_identity_sha256="b" * 64,
            artifact_request_sha256="c" * 64,
            launch_abi="ltj_npu_raw_v1",
            selected_candidate="test",
        ),
        benchmark_config=dict(
            warmup_iterations=10,
            sample_count=20,
            iterations_per_sample=50,
            graph_warmup_replays=1,
            timing_method="device_events_over_ascend_graph",
            statistic="nearest_rank_p50",
        ),
        stream_us=copy.deepcopy(metric),
        submit_us=copy.deepcopy(metric),
        end_to_end_us=copy.deepcopy(metric),
        equivalent_baseline=dict(
            provider="nvidia_equivalent",
            reference_platform="nvidia",
            target_platform="ascend",
            reference_case="identity_fp8_e4m3_2x3x4",
            scale_metric="hbm_copy",
            timing_profile="native",
            reference_validation="bit_exact",
            reference_time_us=1.0,
            reference_peak=3.0,
            target_peak=2.0,
            traffic_ratio=1.0,
            min_relative_efficiency=0.8,
            equivalent_time_us=1.5,
            limit_us=1.875,
            relative_efficiency=0.75,
            passed=False,
        ),
    )


class EquivalentRecordContract(unittest.TestCase):
    def test_real_measurement_with_below_baseline_verdict_is_preserved(self):
        record = fixture()
        records, errors = runner.benchmark_records(json.dumps(record), adapter)
        self.assertEqual(errors, [])
        result = dict(status="passed")
        adapter.postprocess_result(
            result=result,
            ctest_reported_status="passed",
            output="",
            operator="identity",
            suite="benchmark",
            records=records,
            manifest_operators=[],
        )
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["equivalent_performance"]["gate_enabled"])
        data = runner.performance_data(dict(records=records))
        detail = next(iter(data.values()))["details"][record["case"]]
        self.assertFalse(detail["equivalent_baseline"]["passed"])
        self.assertEqual(detail["speedup"], 0.75)
        self.assertEqual(set(records[record["case"]]), {"flagdnn"})

    def test_wrong_platform_formula_profile_and_samples_are_rejected(self):
        mutations = [
            ("equivalent_baseline", "target_platform", "hygon"),
            ("equivalent_baseline", "equivalent_time_us", 9.0),
            ("equivalent_baseline", "passed", True),
            ("equivalent_baseline", "target_peak", float("nan")),
            ("equivalent_baseline", "traffic_ratio", True),
            (
                "equivalent_baseline",
                "reference_case",
                "identity_fp8_e5m2_2x3x4",
            ),
            ("benchmark_config", "graph_warmup_replays", 0),
            ("benchmark_config", "graph_warmup_replays", True),
            ("benchmark_config", "timing_method", "host_launches"),
            ("stream_us", "samples", [2.0]),
            ("stream_us", "samples", None),
        ]
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                record = fixture()
                record[section][key] = value
                self.assertRaises(
                    ValueError, adapter.validate_benchmark_record, record
                )

    def test_malformed_metric_becomes_a_record_error(self):
        record = fixture()
        record["stream_us"] = None
        records, errors = runner.benchmark_records(json.dumps(record), adapter)
        self.assertEqual(records, {})
        self.assertEqual(len(errors), 1)

    def test_oversized_numbers_are_record_errors_not_batch_exceptions(self):
        for section, key, value in (
            ("equivalent_baseline", "reference_peak", 10**1000),
            ("stream_us", "median", 10**1000),
            ("submit_us", "samples", [10**1000] * 20),
            ("end_to_end_us", "samples", [float("inf")] * 20),
        ):
            with self.subTest(section=section, key=key):
                record = fixture()
                record[section][key] = value
                records, errors = runner.benchmark_records(
                    json.dumps(record), adapter
                )
                self.assertEqual(records, {})
                self.assertEqual(len(errors), 1)

    def test_all_metrics_check_sample_count_and_nearest_rank_statistics(self):
        for version in (2, 6):
            for metric in ("stream_us", "submit_us", "end_to_end_us"):
                for key, value in (
                    ("samples", [2.0] * 19),
                    ("samples", [True] * 20),
                    ("median", 2.5),
                    ("p90", 3.0),
                ):
                    with self.subTest(version=version, metric=metric, key=key):
                        record = fixture()
                        if version == 2:
                            record["schema_version"] = 2
                            del record["equivalent_baseline"]
                            for extra in (
                                "timing_method",
                                "graph_warmup_replays",
                                "statistic",
                            ):
                                del record["benchmark_config"][extra]
                        record[metric][key] = value
                        self.assertRaises(
                            ValueError,
                            adapter.validate_benchmark_record,
                            record,
                        )

    def test_even_sample_median_is_nearest_rank_not_midpoint(self):
        record = fixture()
        record["submit_us"] = dict(
            median=10.0, p90=18.0, samples=list(range(1, 21))
        )
        adapter.validate_benchmark_record(record)
        record["submit_us"]["median"] = 10.5
        self.assertRaises(
            ValueError, adapter.validate_benchmark_record, record
        )

    def test_formula_tolerance_cannot_flip_the_performance_verdict(self):
        record = fixture()
        measured = 1.875 + 5e-9
        record["stream_us"] = dict(
            median=measured, p90=measured, samples=[measured] * 20
        )
        record["equivalent_baseline"].update(
            limit_us=1.875 + 1e-8,
            relative_efficiency=1.5 / measured,
            passed=True,
        )
        self.assertRaises(
            ValueError, adapter.validate_benchmark_record, record
        )

    def test_native_v2_cannot_smuggle_equivalent_metadata(self):
        record = fixture()
        record["schema_version"] = 2
        self.assertRaises(
            ValueError, adapter.validate_benchmark_record, record
        )

    def test_fake_reference_samples_are_rejected(self):
        record = fixture()
        result = dict(status="passed")
        adapter.postprocess_result(
            result=result,
            ctest_reported_status="passed",
            output="",
            operator="identity",
            suite="benchmark",
            records={
                record["case"]: {
                    "flagdnn": record,
                    "aclnn": copy.deepcopy(record),
                }
            },
            manifest_operators=[],
        )
        self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
