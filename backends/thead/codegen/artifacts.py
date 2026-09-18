# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Finalize execution programs with compiler-authoritative launch resources."""

from pathlib import Path
from typing import Any
import json

from .io import _atomic_write
from .resources import populate_launch_resources


def write_program_manifest(
    output_directory: Path, manifest: dict[str, Any]
) -> None:
    # Dispatch constructs provisional launch plans. Every emitted candidate
    # must use the actual compiler resources, including autotune candidates.
    for stage in manifest["program"]["stages"]:
        kernel = stage["kernel"]
        source = output_directory / kernel["materialized_source"]["path"]
        for variant in stage["variants"]:
            populate_launch_resources(source, kernel["function"], variant)
    encoded = (
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write(output_directory / "manifest.json", encoded)
