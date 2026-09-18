# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Read launch resources from the same compiler used by the PPU JIT engine."""

from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys

from ..compiler_identity import _jit_paths
from ..triton_compat import install_cuda_jit_bridge


@lru_cache(maxsize=1)
def _standalone_compiler():
    path = _jit_paths()["standalone_compile"]
    # The dependency imports gen_ssig from its scripts directory.
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "_thead_standalone_compile", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    install_cuda_jit_bridge(module)
    return module


def populate_launch_resources(
    source: Path, function: str, variant: dict
) -> None:
    options = variant["compile_options"]
    extra = dict(options["ppu_compiler_options"])
    if options["maxnreg"] is not None:
        extra["maxnreg"] = options["maxnreg"]
    cache = Path(
        _standalone_compiler().compile_a_kernel(
            str(source),
            function,
            variant["full_signature"],
            options["num_warps"],
            options["num_stages"],
            0,
            extra,
        )
    )
    metadata = json.loads((cache / f"{function}.json").read_text())
    shared = metadata.get("shared")
    if type(shared) is not int or shared < 0:
        raise ValueError(
            "PPU compiler returned invalid shared-memory metadata"
        )
    variant["launch"]["shared_memory"] = shared
