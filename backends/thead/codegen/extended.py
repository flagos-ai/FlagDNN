# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Materialize single-node extended plans using the common kernel registry."""

from .artifacts import write_program_manifest

import ast
from pathlib import Path

from flagdnn_codegen import kernel_registry
from ..compiler_identity import PROVIDER_NAME
from ..dispatch.extended import kernel_configuration
from ..dispatch.selection import _registry_sha256
from .abi import _manifest_tensor, _tensor_argument, _tensor_pointer_signature
from .io import _atomic_write, _sha256, compiler_entry_path


def _compile_extended(
    *,
    request,
    request_bytes,
    identity,
    target,
    output_directory,
    enable_autotune,
    operation,
):
    tensors, config = kernel_configuration(request["graph"])
    function, runtime_signature, constants, grid, layout = config
    candidate = kernel_registry.select_kernel_candidate("thead", operation)
    if candidate.ownership != "common" or function not in candidate.functions:
        raise ValueError("extended THead kernel registry contract mismatch")
    source = kernel_registry.resolve_kernel_source(
        compiler_entry_path(), candidate
    )
    source_bytes = kernel_registry.materialize_kernel_source(source, candidate)
    definition = next(
        node
        for node in ast.parse(source_bytes).body
        if isinstance(node, ast.FunctionDef) and node.name == function
    )
    tokens, arguments = [], []
    tensor_index, layout_index = 0, 0
    for parameter in definition.args.args:
        name = parameter.arg
        if name in constants:
            tokens.append(str(constants[name]))
        elif name in runtime_signature:
            kind, value = layout[layout_index]
            layout_index += 1
            if kind == "tensor_alias":
                # Optional pointers are compile-time absent when the matching
                # HAS_* constexpr disables the branch using them.
                tokens.append("nullopt")
            elif kind == "tensor":
                tensor = tensors[tensor_index]
                tensor_index += 1
                tokens.append(_tensor_pointer_signature(tensor))
                arguments.append(_tensor_argument(tensor))
            else:
                raise ValueError(
                    f"unsupported extended argument layout: {kind}"
                )
        else:
            raise ValueError(f"extended kernel parameter is not bound: {name}")
    if tensor_index != len(tensors):
        raise ValueError("extended kernel did not bind every graph tensor")
    variant = {
        "variant_id": "default",
        "full_signature": ",".join(tokens),
        "argument_count": len(arguments),
        "arguments": arguments,
        "compile_options": {
            "num_warps": 4,
            "num_stages": 1,
            "maxnreg": None,
            "ppu_compiler_options": {},
        },
        "launch": {
            "grid": list(grid),
            "block": [128, 1, 1],
            "shared_memory": 0,
        },
    }
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink():
        raise ValueError("THead artifact output directory is a symlink")
    materialized = "generated_stage_0.py"
    _atomic_write(output_directory / materialized, source_bytes)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request["flagdnn_version"],
        "graph_ir_schema_version": 3,
        "backend_abi_version": 2,
        "backend": "thead",
        "target": target,
        "warp_size": 32,
        "engine": "libtriton_jit",
        "request_sha256": _sha256(request_bytes),
        "compiler": identity,
        "external_uids": [t["uid"] for t in request["graph"]["tensors"]],
        "tensor_count": len(tensors),
        "tensors": [_manifest_tensor(t) for t in request["graph"]["tensors"]],
        "workspace": {"size": 0, "alignment": 256},
        "program": {
            "schema_version": 1,
            "stage_count": 1,
            "stages": [
                {
                    "stage_id": 0,
                    "source_node_ids": [0],
                    "dependencies": [],
                    "operation": operation,
                    "kernel": {
                        "provider": candidate.provider,
                        "ownership": candidate.ownership,
                        "function": function,
                        "registry_sha256": _registry_sha256(),
                        "materialized_source": {
                            "path": materialized,
                            "size": len(source_bytes),
                            "sha256": _sha256(source_bytes),
                        },
                    },
                    "variants": [variant],
                    "tuning": None,
                }
            ],
        },
    }
    write_program_manifest(output_directory, manifest)
    return {
        "schema_version": 1,
        "status": "success",
        "backend": "thead",
        "provider": PROVIDER_NAME,
        "node_count": 1,
        "stage_count": 1,
        "target": target,
        "warp_size": 32,
        "artifact_directory": str(output_directory),
        "workspace_size": 0,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }
