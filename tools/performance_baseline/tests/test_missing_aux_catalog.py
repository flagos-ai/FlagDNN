#!/usr/bin/env python3
"""GPU-free contract for the 80-case auxiliary collector.

Build its binary first, then run this script with --binary PATH. CUDA devices
are hidden in every subprocess so catalog and host-oracle tests cannot rely on
an available GPU.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from typing import Any, ClassVar

ROOT = Path(__file__).resolve().parents[3]
_BINARY_ENV = os.environ.get("FLAGDNN_MISSING_AUX_BINARY")
BINARY: Path | None = Path(_BINARY_ENV).resolve() if _BINARY_ENV else None


def invoke(
    *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BINARY), *arguments],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=check,
        text=True,
        capture_output=True,
        timeout=30,
    )


class AuxiliaryCatalogContract(unittest.TestCase):
    rows: ClassVar[list[dict[str, Any]]]

    @classmethod
    def setUpClass(cls) -> None:
        if BINARY is None or not BINARY.is_file():
            raise unittest.SkipTest(
                "build the auxiliary collector and set "
                "FLAGDNN_MISSING_AUX_BINARY "
                "or pass --binary when running this file"
            )
        cls.rows = [
            json.loads(line)
            for line in invoke("--catalog").stdout.splitlines()
        ]

    def test_matches_appendix_one_exactly(self) -> None:
        document = (
            ROOT / "docs/nvidia-performance-missing-cases-20261012.md"
        ).read_text()
        appendix = document.split("## 附录一 ", 1)[1].split("## 附录二 ", 1)[0]
        documented = {
            line
            for block in re.findall(r"```text\n(.*?)```", appendix, re.S)
            for line in block.splitlines()
            if line.startswith(
                (
                    "identity_bfloat16_",
                    "identity_int32_",
                    "identity_bool_",
                    "logical_and_bool_",
                    "logical_or_bool_",
                    "reduction_",
                )
            )
        }
        actual = [row["case_name"] for row in self.rows]
        self.assertEqual(len(documented), 80)
        self.assertEqual(len(actual), 80)
        self.assertEqual(set(actual), documented)
        self.assertEqual(set(invoke("--list").stdout.splitlines()), documented)

    def test_group_counts_and_metadata(self) -> None:
        self.assertEqual(
            Counter(
                (row["op"], row["tensors"][0]["dtype"]) for row in self.rows
            ),
            {
                ("identity", "bfloat16"): 12,
                ("identity", "int32"): 12,
                ("identity", "bool"): 12,
                ("logical_and", "bool"): 12,
                ("logical_or", "bool"): 12,
                ("reduction", "fp16"): 6,
                ("reduction", "bfloat16"): 14,
            },
        )
        for row in self.rows:
            self.assertEqual(row["status"], "CATALOG")
            self.assertEqual(row["provider"], "cudnn")
            self.assertEqual(row["correctness"]["status"], "NOT_RUN")
            self.assertEqual(row["timing"]["samples_us"], [])
            self.assertNotIn("median_us", row["timing"])
            self.assertEqual(
                [
                    row["timing"][key]
                    for key in (
                        "warmup_iterations",
                        "sample_count",
                        "iterations_per_sample",
                        "graph_warmup_launches",
                    )
                ],
                [10, 20, 50, 1],
            )
            self.assertEqual(
                len(row["inputgen"]["inputs"]), len(row["tensors"]) - 1
            )
            for tensor in row["tensors"]:
                self.assertGreater(tensor["uid"], 0)
                self.assertEqual(len(tensor["shape"]), len(tensor["strides"]))
                self.assertEqual(tensor["strides"], tensor["native_strides"])
                if tensor["dtype"] == "bool":
                    self.assertEqual(tensor["reference_storage_bits"], 1)
                    self.assertEqual(tensor["logical_storage_bits"], 8)
                    self.assertEqual(
                        tensor["reference_storage_bytes"],
                        (tensor["storage_elements"] + 7) // 8,
                    )
            if row["op"] == "reduction":
                self.assertEqual(
                    [t["binding_byte_offset"] for t in row["tensors"]],
                    [16, 32],
                )
                self.assertEqual(row["tensors"][1]["dtype"], "fp32")
                if row["tensors"][0]["dtype"] == "bfloat16":
                    self.assertEqual(len(row["tensors"][0]["shape"]), 2)
                    self.assertEqual(row["attributes"]["mode_name"], "sum")
                else:
                    self.assertIn(
                        (
                            row["tensors"][0]["shape"],
                            row["attributes"]["axis"],
                        ),
                        [([3, 5, 7], 1), ([2, 7, 5, 3], 2)],
                    )
            domain = "logical" if row["op"].startswith("logical_") else "real"
            self.assertEqual(
                {x["domain"] for x in row["inputgen"]["inputs"]}, {domain}
            )

    def test_exact_case_filters_and_invalid_selection(self) -> None:
        wanted = [self.rows[0]["case_name"], self.rows[-1]["case_name"]]
        actual = invoke("--list", "--case", wanted[0], "--case", wanted[1])
        self.assertEqual(set(actual.stdout.splitlines()), set(wanted))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.txt"
            path.write_text("# exact names\n\n" + "\n".join(wanted) + "\n")
            actual = invoke("--list", "--case-file", str(path))
            self.assertEqual(set(actual.stdout.splitlines()), set(wanted))
            path.write_text("# no cases\n")
            self.assertEqual(
                invoke(
                    "--list", "--case-file", str(path), check=False
                ).returncode,
                2,
            )
        failed = invoke(
            "--catalog", "--case", "identity_fp8_e8m0_2x3x4", check=False
        )
        self.assertEqual(failed.returncode, 2)
        self.assertIn("unknown auxiliary case", failed.stderr)

    def test_host_oracles_reject_unwritten_output_and_corrupted_guards(
        self,
    ) -> None:
        result = json.loads(invoke("--self-test").stdout)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["cases"], 80)
        self.assertGreater(result["checked_elements"], 0)
        self.assertEqual(result["negative_controls_per_case"], 3)
        self.assertFalse(result["gpu_initialized"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=BINARY)
    arguments, remaining = parser.parse_known_args()
    BINARY = (
        arguments.binary.resolve() if arguments.binary is not None else None
    )
    unittest.main(argv=[__file__, *remaining])
