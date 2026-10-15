#!/usr/bin/env python3
"""
Audit supplemental raw evidence and emit canonical, strictly validated cases.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import tarfile
from pathlib import Path

from run_benchmark_nvidia_supplement import ROOT, required_names  # noqa: F401

MEASUREMENT_ID = "nvidia-supplement-20261015"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(row, catalog):
    name = row["case_name"]
    require(
        row["status"] == "PASS" and row["process_exit"] == 0,
        f"{name}: not successful",
    )
    for field in (
        "op",
        "case_name",
        "provider",
        "source_revision",
        "tensors",
        "attributes",
        "input_domains",
        "input_generator",
    ):
        require(
            row[field] == catalog[field],
            f"{name}: {field} differs from catalog",
        )
    require(
        row["provider"] == "cudnn" and row["cudnn_version"] == 92400,
        "unexpected provider",
    )
    require(
        row["device"] == "NVIDIA H100 80GB HBM3"
        and row["compute_capability"] == [9, 0],
        "unexpected device; cannot bind this run to the H100 peaks",
    )
    require(
        row["stage"] == "complete"
        and row["graph_capture"] == row["graph_replay"] == "PASS",
        "graph execution incomplete",
    )
    correctness = row["correctness"]
    for field in ("absolute_tolerance", "relative_tolerance"):
        tolerance = correctness[field]
        require(
            math.isfinite(tolerance)
            and tolerance >= 0
            and tolerance == catalog["correctness"][field],
            f"{name}: {field} differs from catalog",
        )
    require(
        correctness["status"] == "PASS" and correctness["scope"] == "cpu_full",
        "CPU validation missing",
    )
    output = row["tensors"][-1]
    total = math.prod(output["shape"])
    require(
        correctness["total_elements"]
        == correctness["checked_elements"]
        == total,
        "output coverage differs",
    )
    phases = correctness["phases"]
    names = [p["phase"] for p in phases]
    require(len(names) == len(set(names)), "duplicate phase")
    require(
        names[-3:] == ["before_timing", "after_timing", "after_reset_replay"],
        "three validation phases missing",
    )
    require(
        len(phases) == row["special_encoding_fixtures"] + 3,
        "fixture count differs",
    )
    for p in phases:
        require(
            p["status"] == "PASS"
            and p["checked_elements"] == total
            and p["inputs_unchanged"]
            and p["padding_and_boundaries_intact"],
            "phase validation failed",
        )
        require(
            all(
                math.isfinite(p[field]) and p[field] >= 0
                for field in (
                    "maximum_absolute_error",
                    "maximum_relative_error",
                )
            ),
            "invalid validation error",
        )
        require(
            p["maximum_absolute_error"] <= correctness["absolute_tolerance"]
            or p["maximum_relative_error"]
            <= correctness["relative_tolerance"],
            "numeric error exceeds tolerance",
        )
    t = row["timing"]
    expected = {
        "method": "cuda_event_cuda_graph_batch",
        "unit": "us",
        "warmup_iterations": 10,
        "sample_count": 20,
        "iterations_per_sample": 50,
        "graph_warmup_launches": 1,
        "percentile_method": "nearest_rank",
    }
    require(
        all(t[k] == v for k, v in expected.items()), "timing profile differs"
    )
    samples, batches = t["samples_us"], t["sample_batches_us"]
    require(len(samples) == len(batches) == 20, "incomplete samples")
    require(
        all(math.isfinite(x) and x > 0 for x in samples + batches),
        "invalid sample",
    )
    require(
        all(
            math.isclose(a * 50, b, rel_tol=1e-12)
            for a, b in zip(samples, batches)
        ),
        "batch division differs",
    )
    require(
        t["median_us"] == sorted(samples)[9]
        and t["p90_us"] == sorted(samples)[17],
        "percentile differs",
    )
    return t["median_us"]


def canonical(row, measurement_id):
    attrs = deepcopy(row["attributes"])
    if row["op"] != "concatenate":
        require(attrs.pop("alpha") == 1, "unexpected alpha")
        attrs["pointwise_mode"] = attrs.pop("mode")
        attrs["pointwise_flags"] = attrs.pop("flags")
    tensors = [
        {
            k: t[k]
            for k in (
                "uid",
                "role",
                "dtype",
                "shape",
                "strides",
                "binding_byte_offset",
                "reference_storage_bits",
            )
        }
        for t in row["tensors"]
    ]
    return {
        "op": row["op"],
        "tensors": tensors,
        "time_us": row["timing"]["median_us"],
        "scale_metric": None,
        "timing_profile": "native",
        "validation": "cpu_full",
        "attributes": attrs,
        "input_domains": row["input_domains"],
        "input_generation": row["input_generator"],
        "measurement_id": measurement_id,
    }


def negative_controls(row, catalog):
    changes = [
        lambda r: r["timing"].update(samples_us=[]),
        lambda r: r["correctness"]["phases"].pop(),
        lambda r: r["correctness"]["phases"][-1].update(
            padding_and_boundaries_intact=False
        ),
        lambda r: r["correctness"]["phases"][-1].update(
            inputs_unchanged=False
        ),
        lambda r: r["correctness"]["phases"][-1].update(checked_elements=0),
        lambda r: r.update(status="ERROR"),
        lambda r: r.update(source_revision="wrong"),
        lambda r: r["tensors"][0]["strides"].__setitem__(-1, 999),
        lambda r: r["attributes"].update(softplus_beta=123),
        lambda r: r["timing"].update(warmup_iterations=0),
        lambda r: r["timing"].update(median_us=0),
        lambda r: r["timing"]["sample_batches_us"].__setitem__(0, 1),
        lambda r: r["timing"]["samples_us"].__setitem__(0, float("nan")),
        lambda r: r.update(process_exit=1),
        lambda r: r["correctness"].update(scope="sanity"),
        lambda r: r.update(graph_replay="FAIL"),
        lambda r: r.update(input_domains=[999]),
        lambda r: r.update(device="different GPU"),
        lambda r: r["correctness"].update(absolute_tolerance=float("inf")),
        lambda r: r["correctness"].update(
            relative_tolerance=catalog["correctness"]["relative_tolerance"] + 1
        ),
        lambda r: r["correctness"]["phases"][-1].update(
            maximum_absolute_error=-1
        ),
    ]
    for mutate in changes:
        bad = deepcopy(row)
        mutate(bad)
        try:
            validate(bad, catalog)
        except ValueError:
            continue
        raise ValueError("corrupted evidence escaped audit")
    return len(changes)


def audit(output, *, historical=False, measurement_id=None):
    if historical:
        require(
            measurement_id in (None, MEASUREMENT_ID),
            "historical measurement identity differs",
        )
        measurement_id = MEASUREMENT_ID
    else:
        require(
            isinstance(measurement_id, str)
            and measurement_id.strip()
            and measurement_id != MEASUREMENT_ID,
            "a new collection requires a distinct --measurement-id",
        )
    expected_names = required_names(include_retired=historical)
    catalogs = [
        json.loads(s)
        for s in (output / "catalog.jsonl").read_text().splitlines()
    ]
    catalog = {r["case_name"]: r for r in catalogs}
    require(
        len(catalogs) == len(catalog) == len(expected_names)
        and set(catalog) == expected_names,
        "catalog coverage differs",
    )
    rows = [
        json.loads(s)
        for s in (output / "results.jsonl").read_text().splitlines()
    ]
    require(
        len(rows) == len(expected_names) + (15 if historical else 0),
        "unexpected measurement variants or attempts",
    )
    require(
        len({(r["variant"], r["case_name"]) for r in rows}) == len(rows),
        "duplicate measurement attempt",
    )
    defaults = [r for r in rows if r["variant"] == "default"]
    require(
        len(defaults) == len(expected_names)
        and {r["case_name"] for r in defaults} == expected_names,
        "measurement coverage differs",
    )
    probes = [r for r in rows if r["variant"] == "int32_compute"]
    require(
        len(probes) == (15 if historical else 0)
        and {r["case_name"] for r in probes}
        == {r["case_name"] for r in defaults if r["op"] == "pow"},
        "INT32 capability probe missing",
    )
    accepted, sources, failures = {}, {}, {}
    for row in rows:
        name = row["case_name"]
        require(name in catalog, "unexpected case")
        suffix = Path("raw") / row["variant"] / (name + ".jsonl")
        require(
            tuple(Path(row["raw_stdout"]).parts[-len(suffix.parts) :])
            == suffix.parts,
            "stdout artifact identity differs",
        )
        require(
            tuple(Path(row["raw_stderr"]).parts[-len(suffix.parts) :])
            == suffix.with_suffix(".stderr").parts,
            "stderr artifact identity differs",
        )
        stdout = output / suffix
        stderr = stdout.with_suffix(".stderr")
        require(stdout.is_file() and stderr.is_file(), "raw logs missing")
        require(
            int(stdout.with_suffix(".exit").read_text())
            == row["process_exit"],
            "exit evidence differs",
        )
        raw = [
            json.loads(s)
            for s in stdout.read_text().splitlines()
            if s.startswith("{")
        ]
        require(
            len(raw) == 1 and all(row[k] == v for k, v in raw[0].items()),
            "raw result differs",
        )
        if row["status"] != "PASS":
            require(
                "median_us" not in row["timing"],
                "failed case exposes an eligible time",
            )
            failures.setdefault(name, []).append(
                {
                    "variant": row["variant"],
                    "stage": row["stage"],
                    "reason": row["reason"],
                    "graph_build_error": row.get("graph_build_error", ""),
                }
            )
            continue
        validate(row, catalog[name])
        if name not in accepted:
            accepted[name], sources[name] = canonical(row, measurement_id), row
    require(bool(accepted), "no validated measurements")
    first = next(iter(sources.values()))
    negatives = negative_controls(first, catalog[first["case_name"]])
    hashes = json.loads((output / "source-sha256.json").read_text())
    with tarfile.open(output / "source-snapshot.tar.gz", "r:gz") as archive:
        require(
            set(archive.getnames()) == set(hashes),
            "source snapshot membership differs",
        )
        for name, digest in hashes.items():
            stream = archive.extractfile(name)
            require(
                stream is not None,
                f"source snapshot member is not a file: {name}",
            )
            require(
                hashlib.sha256(stream.read()).hexdigest() == digest,
                f"source snapshot corrupted: {name}",
            )
    binary_hashes = json.loads((output / "binary-sha256.json").read_text())
    expected_binary = binary_hashes.get(
        "collector.bin", binary_hashes.get("build/nvidia/nv_supplement")
    )
    require(
        expected_binary is not None, "collector binary fingerprint missing"
    )
    require(
        hashlib.sha256((output / "collector.bin").read_bytes()).hexdigest()
        == expected_binary,
        "archived collector binary differs from measurement",
    )
    missing = sorted(expected_names - accepted.keys())
    summary = {
        "requested": len(expected_names),
        "passed": len(accepted),
        "unavailable": len(missing),
        "raw_attempts": len(rows),
        "negative_controls": negatives,
        "added_case_counts": dict(Counter(r["op"] for r in sources.values())),
        "elements_per_validation_phase": sum(
            r["correctness"]["total_elements"] for r in sources.values()
        ),
        "unavailable_names": missing,
    }
    (output / "accepted-cases.json").write_text(
        json.dumps(accepted, indent=2) + "\n"
    )
    (output / "failures.json").write_text(
        json.dumps(failures, indent=2) + "\n"
    )
    (output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--historical-20261015",
        action="store_true",
        help=(
            "audit the immutable original 175-case evidence, including "
            "retired cases"
        ),
    )
    parser.add_argument(
        "--measurement-id", help="unique identity for a new active collection"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.output.resolve(),
                historical=args.historical_20261015,
                measurement_id=args.measurement_id,
            ),
            indent=2,
        )
    )
