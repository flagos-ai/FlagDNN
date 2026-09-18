# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead provider entry: validate requests and assemble execution programs."""

from __future__ import annotations

from .compiler_identity import (
    PROVIDER_NAME,
    PROVIDER_VERSION,
    build_compiler_identity,
    compiler_identity_dependency_paths,
)
from pathlib import Path
from typing import Any
from .codegen.emit import (
    _compile_graph_operation,
)
from .codegen.io import (
    _load_request,
)
from .dispatch.common import (
    SCHEMA_VERSION,
    _SHA256,
    _VERSION,
    _parse_build_options,
    _require_exact_fields,
    _string,
)
from .dispatch.graph import (
    _parse_graph,
)


def compiler_identity(
    target: str, execution_engine: str = "libtriton_jit"
) -> dict[str, Any]:
    return build_compiler_identity(target, execution_engine)


def compiler_identity_dependencies(
    target: str, execution_engine: str = "libtriton_jit"
) -> tuple[Path, ...]:
    return compiler_identity_dependency_paths(target, execution_engine)


def compile_request(
    request_path: Path,
    output_directory: Path,
    execution_engine: str = "libtriton_jit",
) -> dict[str, Any]:
    if execution_engine != "libtriton_jit":
        raise ValueError("THead supports only the libtriton_jit engine")
    request, request_bytes = _load_request(Path(request_path))
    _require_exact_fields(
        request,
        {
            "schema_version",
            "flagdnn_version",
            "backend",
            "target",
            "compiler_identity",
            "build_options",
            "graph",
        },
        set(),
        "request",
    )
    if request["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported request schema_version")
    version = _string(request["flagdnn_version"], "request FlagDNN version")
    if _VERSION.fullmatch(version) is None:
        raise ValueError("request FlagDNN version is invalid")
    if request["backend"] != "thead":
        raise ValueError("THead provider received another backend")
    target = _string(request["target"], "request target")
    identity = compiler_identity(target, execution_engine)
    requested_identity = request["compiler_identity"]
    if (
        not isinstance(requested_identity, str)
        or _SHA256.fullmatch(requested_identity) is None
        or requested_identity != identity["identity_sha256"]
    ):
        raise ValueError("request compiler identity does not match provider")
    enable_autotune = _parse_build_options(request["build_options"])
    operation_types = _parse_graph(request["graph"])
    return _compile_graph_operation(
        request=request,
        request_bytes=request_bytes,
        identity=identity,
        target=target,
        output_directory=Path(output_directory),
        enable_autotune=enable_autotune,
        operation_types=operation_types,
        execution_engine=execution_engine,
    )


__all__ = (
    "PROVIDER_NAME",
    "PROVIDER_VERSION",
    "compile_request",
    "compiler_identity",
    "compiler_identity_dependencies",
)
