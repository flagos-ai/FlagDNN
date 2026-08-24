#!/usr/bin/env python3
"""End-to-end contract for the Iluvatar compiler provider."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(
    command: list[str],
    *,
    expect_success: bool,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if expect_success and result.returncode != 0:
        fail(
            "compiler command failed:\n"
            + " ".join(command)
            + "\nstdout:\n"
            + result.stdout
            + "\nstderr:\n"
            + result.stderr
        )
    if not expect_success and result.returncode == 0:
        fail(
            "negative compiler command unexpectedly passed: "
            + " ".join(command)
        )
    return result


def request(identity: str, operation: str = "add") -> dict[str, Any]:
    return {
        "schema_version": 3,
        "flagdnn_version": "0.1.0",
        "backend": "iluvatar",
        "target": "corex_71",
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": False},
        "graph": {
            "name": "iluvatar compiler contract",
            "tensor_count": 3,
            "tensors": [
                {
                    "uid": uid,
                    "data_type": "float32",
                    "dimensions": [16],
                    "strides": [1],
                    "alignment": 16,
                    "virtual": False,
                }
                for uid in (1, 2, 3)
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": operation,
                    "name": operation + "_0",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "left", "uid": 1},
                        {"name": "right", "uid": 2},
                    ],
                    "outputs": [{"name": "output", "uid": 3}],
                    "attributes": {"alpha": 1.0, "n_elements": 16},
                }
            ],
        },
    }


def conv_bias_relu_request(identity: str) -> dict[str, Any]:
    def tensor(
        uid: int,
        dimensions: list[int],
        strides: list[int],
        *,
        virtual: bool,
    ) -> dict[str, Any]:
        return {
            "uid": uid,
            "data_type": "float32",
            "dimensions": dimensions,
            "strides": strides,
            "alignment": 16,
            "virtual": virtual,
        }

    output_dimensions = [1, 5, 8, 8]
    output_strides = [320, 1, 40, 5]
    return {
        "schema_version": 3,
        "flagdnn_version": "0.1.0",
        "backend": "iluvatar",
        "target": "corex_71",
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": False},
        "graph": {
            "name": "iluvatar fused conv bias relu contract",
            "tensor_count": 6,
            "tensors": [
                tensor(1, [1, 3, 8, 8], [192, 1, 24, 3], virtual=False),
                tensor(2, [5, 3, 1, 1], [3, 1, 3, 3], virtual=False),
                tensor(3, [1, 5, 1, 1], [5, 1, 5, 5], virtual=False),
                tensor(4, output_dimensions, output_strides, virtual=True),
                tensor(5, output_dimensions, output_strides, virtual=True),
                tensor(6, output_dimensions, output_strides, virtual=False),
            ],
            "node_count": 3,
            "nodes": [
                {
                    "id": 0,
                    "type": "convolution_fprop",
                    "name": "convolution_fprop",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "input", "uid": 1},
                        {"name": "filter", "uid": 2},
                    ],
                    "outputs": [{"name": "output", "uid": 4}],
                    "attributes": {
                        "spatial_rank": 2,
                        "groups": 1,
                        "n_outputs": 320,
                        "pre_padding": [0, 0],
                        "post_padding": [0, 0],
                        "stride": [1, 1],
                        "dilation": [1, 1],
                    },
                },
                {
                    "id": 1,
                    "type": "add",
                    "name": "bias_add",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "left", "uid": 4},
                        {"name": "right", "uid": 3},
                    ],
                    "outputs": [{"name": "output", "uid": 5}],
                    "attributes": {"alpha": 1.0, "n_elements": 320},
                },
                {
                    "id": 2,
                    "type": "relu",
                    "name": "relu",
                    "compute_data_type": "float32",
                    "inputs": [{"name": "input", "uid": 5}],
                    "outputs": [{"name": "output", "uid": 6}],
                    "attributes": {
                        "n_elements": 320,
                        "negative_slope": 0.0,
                        "lower_clip": 0.0,
                        "upper_clip": 0.0,
                        "has_upper_clip": 0,
                        "swish_beta": 1.0,
                        "elu_alpha": 1.0,
                        "softplus_beta": 1.0,
                    },
                },
            ],
        },
    }


def write_request(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def directory_snapshot(root: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


def main() -> int:
    if len(sys.argv) != 2:
        fail("usage: compiler_contract.py <FlagDNN source root>")
    source_root = Path(sys.argv[1]).resolve()
    compiler = source_root / "compiler/flagdnn_codegen/main.py"
    python = Path(sys.executable).resolve()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["FLAGDNN_BACKEND_ROOT"] = str(source_root / "backends")

    with tempfile.TemporaryDirectory(
        prefix="flagdnn-iluvatar-compiler-"
    ) as tmp:
        root = Path(tmp)
        identity_output = root / "identity.txt"
        identify = [
            str(python),
            str(compiler),
            "--identify",
            "--backend",
            "iluvatar",
            "--target",
            "corex_71",
            "--execution-engine",
            "libtriton_jit",
            "--identity-output",
            str(identity_output),
            "--quiet",
        ]

        run(
            identify[: identify.index("corex_71")]
            + ["sm_71"]
            + identify[identify.index("corex_71") + 1 :],
            expect_success=False,
            environment=environment,
        )
        external = list(identify)
        external[external.index("libtriton_jit")] = "external_artifact"
        run(external, expect_success=False, environment=environment)
        run(identify, expect_success=True, environment=environment)
        identity_lines = identity_output.read_text(
            encoding="utf-8"
        ).splitlines()
        if not identity_lines or len(identity_lines[0]) != 64:
            fail("compiler identity output is malformed")
        identity = identity_lines[0]
        if any(character not in "0123456789abcdef" for character in identity):
            fail("compiler identity is not lowercase SHA-256")
        metadata = json.loads(identity_lines[1])
        if not metadata.get("dependencies_complete") or not metadata.get(
            "snapshots"
        ):
            fail("compiler identity dependency closure is incomplete")

        malformed = root / "malformed.json"
        write_request(
            malformed,
            {
                "backend": "iluvatar",
                "target": "corex_71",
                "compiler_identity": identity,
            },
        )
        run(
            [
                str(python),
                str(compiler),
                "--request",
                str(malformed),
                "--output-dir",
                str(root / "malformed-output"),
                "--execution-engine",
                "libtriton_jit",
                "--quiet",
            ],
            expect_success=False,
            environment=environment,
        )

        bad_identity = request("0" * 64)
        bad_identity_path = root / "bad-identity.json"
        write_request(bad_identity_path, bad_identity)
        compile_prefix = [str(python), str(compiler), "--request"]
        compile_suffix = ["--execution-engine", "libtriton_jit", "--quiet"]
        run(
            compile_prefix
            + [
                str(bad_identity_path),
                "--output-dir",
                str(root / "bad-identity"),
            ]
            + compile_suffix,
            expect_success=False,
            environment=environment,
        )

        for name, operation in (
            ("unknown-operation", "definitely_unknown"),
            ("missing-kernel", "conv_bias_relu"),
        ):
            path = root / f"{name}.json"
            write_request(path, request(identity, operation))
            run(
                compile_prefix
                + [str(path), "--output-dir", str(root / name)]
                + compile_suffix,
                expect_success=False,
                environment=environment,
            )

        fused_document = conv_bias_relu_request(identity)
        fused_path = root / "fused-conv-bias-relu.json"
        fused_output = root / "fused-conv-bias-relu"
        write_request(fused_path, fused_document)
        run(
            compile_prefix
            + [str(fused_path), "--output-dir", str(fused_output)]
            + compile_suffix,
            expect_success=True,
            environment=environment,
        )
        fused_manifest = json.loads(
            (fused_output / "manifest.json").read_text(encoding="utf-8")
        )
        fused_program = fused_manifest["program"]
        fused_stage = fused_program["stages"][0]
        fused_tensor_uids = [
            argument["uid"]
            for argument in fused_stage["variants"][0]["arguments"]
            if argument["kind"] == "tensor"
        ]
        if (
            fused_program["stage_count"] != 1
            or fused_stage["source_node_ids"] != [0, 1, 2]
            or fused_stage["kernel"]["function"]
            != "conv2d_spatial_nchw_kernel"
            or fused_tensor_uids != [1, 2, 3, 6]
        ):
            fail("Conv-Bias-ReLU did not lower to one fused stage")

        leaky_document = json.loads(json.dumps(fused_document))
        leaky_document["graph"]["nodes"][2]["attributes"][
            "negative_slope"
        ] = 0.25
        leaky_path = root / "unfused-conv-bias-leaky-relu.json"
        leaky_output = root / "unfused-conv-bias-leaky-relu"
        write_request(leaky_path, leaky_document)
        run(
            compile_prefix
            + [str(leaky_path), "--output-dir", str(leaky_output)]
            + compile_suffix,
            expect_success=True,
            environment=environment,
        )
        leaky_manifest = json.loads(
            (leaky_output / "manifest.json").read_text(encoding="utf-8")
        )
        if leaky_manifest["program"]["stage_count"] != 3:
            fail("non-default ReLU was incorrectly fused into convolution")

        valid_request = request(identity)
        request_path = root / "valid.json"
        write_request(request_path, valid_request)
        outputs = [root / "output-a", root / "output-b"]
        for output in outputs:
            run(
                compile_prefix
                + [str(request_path), "--output-dir", str(output)]
                + compile_suffix,
                expect_success=True,
                environment=environment,
            )
        first = directory_snapshot(outputs[0])
        second = directory_snapshot(outputs[1])
        if first != second:
            fail("clean Add compilations are not byte-identical")
        if sorted(first) != ["generated_stage_0.py", "manifest.json"]:
            fail(f"unexpected artifact file list: {sorted(first)}")

        manifest = json.loads(first["manifest.json"])
        if (
            manifest.get("schema_version") != 1
            or manifest.get("backend") != "iluvatar"
            or manifest.get("target") != "corex_71"
            or manifest.get("engine") != "libtriton_jit"
            or manifest.get("program", {}).get("schema_version") != 1
            or manifest.get("program", {}).get("stage_count") != 1
        ):
            fail("Add manifest identity/schema is invalid")
        descriptor = manifest["program"]["stages"][0]["kernel"][
            "materialized_source"
        ]
        if descriptor.get("path") != "generated_stage_0.py":
            fail("Add kernel source was not materialized relatively")
        source = first["generated_stage_0.py"]
        if (
            descriptor.get("size") != len(source)
            or descriptor.get("sha256") != hashlib.sha256(source).hexdigest()
        ):
            fail("materialized source descriptor is invalid")

        forbidden_roots = {
            str(source_root),
            str(root),
            "/usr/local",
            "/home/wbj/libtriton_jit",
        }
        manifest_text = first["manifest.json"].decode("utf-8")
        for forbidden in forbidden_roots:
            if forbidden in manifest_text:
                fail(f"manifest leaked an absolute path: {forbidden}")

    print("PASS Iluvatar deterministic compiler provider contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
