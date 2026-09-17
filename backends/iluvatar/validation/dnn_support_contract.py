#!/usr/bin/env python3
"""Probe the selected CoreX DNN ABI before accepting capability skips."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def main() -> None:
    executable, destination = map(Path, sys.argv[1:])
    result = subprocess.run(
        [str(executable)], text=True, capture_output=True, timeout=60, check=True
    )
    lines = result.stdout.splitlines()
    records = [json.loads(line) for line in lines if line.startswith("{")]
    vendor_stdout = [line for line in lines if line and not line.startswith("{")]
    symbols = {r["name"]: r["available"] for r in records if r["kind"] == "symbol"}
    expected_symbols = {
        "cudnnBackendCreateDescriptor": False,
        "cudnnFlashAttnForward": False,
        "cudnnFlashAttnBackward": False,
        "cudnnBatchNormalizationBackward": True,
        "cudnnPoolingForward": True,
        "cudnnSpatialTfSamplerForward": True,
        "cudnnTransformTensor": True,
        "cudnnReduceTensor": True,
        "cudnnActivationBackward": True,
        "cudnnRmsNormalizationForward": True,
        "cudnnFusedOpsExecute": True,
        "cudnnCausalConv1dForward": False,
        "cudnnRmsNormalizationBackward": False,
    }
    if symbols != expected_symbols:
        raise RuntimeError(
            "DNN exported capabilities changed; requalify adapters and skip policy"
        )
    versions = [r for r in records if r["kind"] == "version"]
    if versions != [{"kind": "version", "header": 7605, "runtime": 7605}]:
        raise RuntimeError("DNN header/runtime does not match the audited CoreX ABI")
    activations = [r for r in records if r["kind"] == "activation"]
    if {(r["dtype"], r["mode"]) for r in activations} != {
        (t, m) for t in (0, 2, 9) for m in range(9)
    } or len(activations) != 27:
        raise RuntimeError("incomplete activation capability probe")
    for r in activations:
        # Header enums 5-8 are accepted by the descriptor but rejected by both
        # execution APIs. BF16 classic activations return NOT_SUPPORTED.
        expected = 3 if r["mode"] >= 5 else 9 if r["dtype"] == 9 else 0
        if (r["set"], r["forward"], r["backward"]) != (0, expected, expected):
            raise RuntimeError(f"activation capability changed; qualify {r}")
    transforms = [r for r in records if r["kind"] == "transform"]
    expected_pairs = {(t, d) for t in (0, 2, 9, 4, 3) for d in (t, 0)}
    if {(r["source"], r["destination"]) for r in transforms} != expected_pairs:
        raise RuntimeError("incomplete transform dtype probe")
    for r in transforms:
        supported = r["source"] == r["destination"] and r["source"] in (0, 2, 3)
        if r["execute"] != 0 or r["correct"] != supported:
            raise RuntimeError(f"transform capability changed; qualify {r}")
    logical = {r["compute"]: r for r in records if r["kind"] == "logical_not"}
    if (
        set(logical) != {0, 3}
        or (
            logical[0]["set"],
            logical[0]["execute"],
            logical[0]["output0"],
            logical[0]["output1"],
        )
        != (0, 0, 0, 255)
        or (logical[3]["set"], logical[3]["execute"]) != (3, 3)
    ):
        raise RuntimeError("logical NOT semantics changed; qualify the boolean adapter")
    integer = [r for r in records if r["kind"] == "int32_pointwise"]
    if (
        len(integer) != 4
        or {r["mode"] for r in integer} != {0, 1, 2, 3}
        or any(
            (r["set"], r["execute"], r["first"], r["integer_compute_descriptor_status"])
            != (0, 9, -1515870811, 3)
            for r in integer
        )
    ):
        raise RuntimeError("INT32 pointwise capability changed; requalify adapters")
    reduction = [r for r in records if r["kind"] == "int32_reduction"]
    if len(reduction) != 1 or reduction[0]["execute"] != 0 or reduction[0]["finite"]:
        raise RuntimeError("INT32 reduction semantics changed; requalify adapter")
    rms = {r["dtype"]: r for r in records if r["kind"] == "rmsnorm"}
    if set(rms) != {0, 2, 9} or rms[0]["execute"] != 9:
        raise RuntimeError("RMSNorm dtype capability changed; requalify adapter")
    for dtype, expected_inverse, expected_bits in (
        (2, 1 / math.sqrt(0.251), 15356),
        (9, 1 / math.sqrt(1.75**2 + 0.001), 16256),
    ):
        r = rms[dtype]
        if (
            r["execute"] != 0
            or any(
                not math.isclose(r[key], expected_inverse, rel_tol=1e-5)
                for key in ("inverse_first", "inverse_last")
            )
            or any(
                r[key] != expected_bits
                for key in ("output_first_bits", "output_last_bits")
            )
        ):
            raise RuntimeError(f"RMSNorm semantics changed; requalify {r}")
    finalize = [r for r in records if r["kind"] == "bn_finalize"]
    if {r["stats_dtype"] for r in finalize} != {0} or len(finalize) != 1:
        raise RuntimeError("incomplete BN finalize capability probe")
    for r in finalize:
        if (
            r["execute"] != 0
            or not r["outputs_unchanged"]
            or r["get_pointer_status"] != [3] * 10
        ):
            raise RuntimeError(f"BN finalize capability changed; requalify {r}")
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "corex_dnn_runtime_capability_evidence",
                "status": "passed",
                "probe_executable_sha256": hashlib.sha256(
                    executable.read_bytes()
                ).hexdigest(),
                "records": records,
                "stderr": result.stderr,
                "vendor_stdout": vendor_stdout,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "PASS CoreX DNN capability evidence: 27 activation modes/dtypes, 9 transform dtype pairs, INT32 arithmetic/reduction, boolean NOT, RMSNorm, BN finalize, exported APIs"
    )


if __name__ == "__main__":
    main()
