#!/usr/bin/env python3
"""Validation-only compiler endpoint that preserves one public Graph request."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


_CAPTURE_IDENTITY = "1" * 64


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identify", action="store_true")
    parser.add_argument("--backend")
    parser.add_argument("--target")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execution-engine")
    parser.add_argument("--identity-output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()

    if arguments.identify:
        if (
            arguments.backend != "mthreads"
            or arguments.execution_engine != "libtriton_jit"
            or arguments.identity_output is None
            or arguments.request is not None
        ):
            return 2
        arguments.identity_output.write_text(
            _CAPTURE_IDENTITY + "\n", encoding="ascii"
        )
        return 0

    capture_value = os.environ.get(
        "FLAGDNN_MTHREADS_CAPTURE_REQUEST", ""
    )
    if (
        not capture_value
        or arguments.request is None
        or arguments.output_dir is None
        or arguments.execution_engine != "libtriton_jit"
    ):
        return 3
    capture = Path(capture_value)
    capture.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(arguments.request, capture)
    return 23


if __name__ == "__main__":
    sys.exit(main())

