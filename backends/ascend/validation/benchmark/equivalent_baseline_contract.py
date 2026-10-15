#!/usr/bin/env python3
"""
CPU-only contract fixtures; synthetic peaks never enter measured baselines.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    executable, baseline = sys.argv[1:]
    document = json.loads(Path(baseline).read_text())
    document["platforms"]["ascend"] = {
        "device": "contract-device",
        "architecture": "contract-architecture",
        "measured_on": "contract-fixture-only",
        "peaks": {"hbm_copy": 1.25},
        "scaling": {
            "case_policies": {
                "identity": {
                    "scale_metric": "hbm_copy",
                    "supported_dtypes": ["fp8_e4m3", "fp8_e5m2"],
                    "reason": (
                        "Synthetic CPU contract fixture, never device "
                        "evidence."
                    ),
                }
            }
        },
    }
    case = "identity_fp8_e4m3_2x3x4"
    mutations = [
        (("platforms", "ascend", "peaks", "hbm_copy"), None),
        (("platforms", "ascend", "peaks", "hbm_copy"), 0),
        (("platforms", "ascend", "peaks", "hbm_copy"), -1),
        (("platforms", "ascend", "measured_on"), None),
        (("platforms", "ascend", "scaling", "case_policies"), {}),
        (
            (
                "platforms",
                "ascend",
                "scaling",
                "case_policies",
                "identity",
                "scale_metric",
            ),
            "vector_fp32",
        ),
        (
            (
                "platforms",
                "ascend",
                "scaling",
                "case_policies",
                "identity",
                "supported_dtypes",
            ),
            [],
        ),
        (("min_relative_efficiency",), 1.1),
        (("reference", "cases", case, "time_us"), 0),
        (("reference", "cases", case, "validation"), "sanity"),
        (("reference", "cases", case, "attributes", "alpha"), 2),
        (("reference", "cases", case, "input_domain"), 1),
        (("reference", "cases", case, "input_domains"), [1]),
        (("reference", "cases", case, "output_count"), 2),
        (("reference", "cases", case, "tensors", 0, "dtype"), "fp16"),
        (("reference", "cases", case, "tensors", 0, "uid"), -123),
        (
            (
                "reference",
                "cases",
                case,
                "tensors",
                0,
                "reference_storage_bits",
            ),
            1,
        ),
        (
            ("reference", "cases", case, "tensors", 0, "reference_strides"),
            [1, 2, 3],
        ),
        (
            (
                "reference",
                "cases",
                case,
                "tensors",
                0,
                "reference_layout_conversion_timed",
            ),
            True,
        ),
        (
            ("reference", "measurement", "timing_method"),
            "device_events_per_kernel",
        ),
        (
            (
                "reference",
                "measurement",
                "timing_profiles",
                "native",
                "iterations_per_sample",
            ),
            20,
        ),
        (
            (
                "reference",
                "measurement",
                "timing_profiles",
                "native",
                "graph_warmup_replays",
            ),
            0,
        ),
        (("metric_units", "hbm_copy"), "GB/s"),
    ]
    with tempfile.TemporaryDirectory(
        prefix="flagdnn-ascend-baseline-contract-"
    ) as directory:
        path = Path(directory) / "fixture.json"
        path.write_text(json.dumps(document))
        subprocess.run([executable, str(path)], check=True)
        # Exercise the actual C++ serializer against the batch-runner contract.
        emitted = subprocess.run(
            [executable, str(path), "--metadata"],
            check=True,
            text=True,
            capture_output=True,
        )
        contract_spec = importlib.util.spec_from_file_location(
            "ascend_record_contract",
            Path(__file__).parent.parent / "equivalent_record_contract.py",
        )
        contract = importlib.util.module_from_spec(contract_spec)
        contract_spec.loader.exec_module(contract)
        record = contract.fixture()
        record["equivalent_baseline"] = json.loads(emitted.stdout)
        measured = record["equivalent_baseline"]["limit_us"] * 2
        for metric in ("stream_us", "submit_us", "end_to_end_us"):
            record[metric] = dict(
                median=measured, p90=measured, samples=[measured] * 20
            )
        contract.adapter.validate_benchmark_record(record)
        for keys, value in mutations:
            invalid = copy.deepcopy(document)
            target = invalid
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            path.write_text(json.dumps(invalid))
            subprocess.run([executable, str(path), "--reject"], check=True)
    print(
        (
            f"Ascend baseline evidence rejection: {len(mutations)} invalid "
            f"fixtures PASS"
        )
    )


if __name__ == "__main__":
    main()
