# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead codegen attention lowering."""

from __future__ import annotations

from .artifacts import write_program_manifest


from ..codegen.io import compiler_entry_path
from ..compiler_identity import PROVIDER_NAME
from flagdnn_codegen import kernel_registry
from pathlib import Path
from typing import Any
import yaml  # type: ignore[import-untyped]
from ..codegen.abi import (
    _manifest_tensor,
    _tensor_argument,
    _tensor_pointer_signature,
)
from ..codegen.io import (
    _atomic_write,
    _canonical_sha256,
    _sha256,
)
from ..dispatch.attention import (
    _validate_attention_graph,
)
from ..dispatch.common import (
    SCHEMA_VERSION,
    _MAX_KERNEL_SOURCE_BYTES,
    _PPU_WARP_SIZE,
    _TUNING_TABLES,
)
from ..dispatch.selection import (
    _registry_sha256,
)


def _compile_attention(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
    operation: str,
) -> dict[str, Any]:
    import ast

    plan = _validate_attention_graph(request["graph"], operation)
    backward, named, attributes = (
        plan["backward"],
        plan["named"],
        plan["attributes"],
    )
    candidate = kernel_registry.select_kernel_candidate("thead", operation)
    fp8 = plan["fp8"]
    function = (
        ("fp8_sdpa_backward_kernel" if backward else "fp8_sdpa_forward_kernel")
        if fp8
        else ("thead_sdpa_bwd_kernel" if backward else "thead_sdpa_fwd_kernel")
    )
    source_name = "fp8_attention.py" if fp8 else "attention.py"
    table = (
        ("sdpa_fp8_backward" if backward else "sdpa_fp8")
        if fp8
        else ("sdpa_backward_dq" if backward else "sdpa")
    )
    tuning = candidate.tuning
    if (
        candidate.backend != "thead"
        or candidate.provider != "thead_triton"
        or candidate.ownership != "platform"
        or candidate.source != source_name
        or candidate.source_layout != "platform"
        or candidate.source_format != "module"
        or candidate.functions != (function,)
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != table
        or tuning.key != "sequence_q"
        or tuning.strategy != "attention"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError("THead attention kernel registry contract is invalid")
    source = kernel_registry.resolve_kernel_source(
        compiler_entry_path(), candidate
    )
    source_bytes = kernel_registry.materialize_kernel_source(source, candidate)
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError("THead attention source size is invalid")
    parameters = next(
        n.args.args
        for n in ast.parse(source_bytes).body
        if isinstance(n, ast.FunctionDef) and n.name == function
    )
    tuning_path = kernel_registry.resolve_tuning_source(
        compiler_entry_path(), candidate
    )
    tuning_bytes = tuning_path.read_bytes()
    document = yaml.safe_load(tuning_bytes)
    if not isinstance(document, dict) or set(document) != _TUNING_TABLES:
        raise ValueError("THead attention tuning tables are invalid")
    configurations = document[tuning.table]
    expected_configs = (
        [
            {
                "META": {},
                "num_warps": warps,
                "num_stages": 1,
                "maxnreg": None,
                "ppu_compiler_options": {},
            }
            for warps in (4, 8)
        ]
        if backward or fp8
        else [
            {
                "META": {"BLOCK_M": tile, "BLOCK_N": 32},
                "num_warps": 4,
                "num_stages": 1,
                "maxnreg": None,
                "ppu_compiler_options": {},
            }
            for tile in (16, 32)
        ]
    )
    if configurations != expected_configs:
        raise ValueError("THead attention tuning candidates are not qualified")
    if not enable_autotune:
        configurations = configurations[:1]
    q, k, v = (named[name] for name in ("q", "k", "v"))
    batch, heads, sq, dimension = q["dimensions"]
    _, key_heads, sk, _ = k["dimensions"]
    value_dimension = v["dimensions"][3]
    pointers = {
        name + "_ptr": tensor
        for name, tensor in named.items()
        if not tensor["virtual"]
    }
    scale = float(attributes["attn_scale"])
    bias_shape = (
        named["bias"]["dimensions"] if "bias" in named else [1, heads, sq, sk]
    )
    constants = dict(
        qk_scale=scale * 1.4426950408889634,
        HQ=heads,
        SQ=sq,
        SKV=sk,
        SK=sk,
        q_per_k=heads // key_heads,
        q_per_v=heads // key_heads,
        min_diag=attributes["min_diag"],
        max_diag=attributes["max_diag"],
        HEAD_DIM=dimension,
        V_DIM=value_dimension,
        ELEM_SIZE=2,
        BLOCK_D=1 << (dimension - 1).bit_length(),
        BLOCK_DV=1 << (value_dimension - 1).bit_length(),
        HAS_BIAS=attributes["has_bias"],
        BANDED=attributes["banded"],
        GENERATE_STATS=attributes["generate_stats"],
        REVERSE_CAUSAL=attributes["reverse_causal"],
        BATCH=batch,
        HKV=key_heads,
        HK=key_heads,
        D=dimension,
        DV=value_dimension,
        SCALE=scale,
        E4=int(q["data_type"] == "fp8_e4m3"),
        STATS=attributes["generate_stats"],
        BIAS_BATCHES=bias_shape[0],
        BIAS_HEADS=bias_shape[1],
        HAS_DBIAS=attributes["has_dbias"],
        CAUSAL=attributes["banded"],
    )
    for name, axes in (
        ("q", "bhmd"),
        ("k", "bhnd"),
        ("v", "bhnd"),
        ("o", "bhmd"),
    ):
        constants.update(
            {
                f"stride_{name}{axis}": stride
                for axis, stride in zip(axes, named[name]["strides"])
            }
        )
    constants.update(
        {
            f"stride_s{axis}": stride
            for axis, stride in zip("bhm", named["stats"]["strides"][:3])
        }
    )
    bias_strides = (
        list(named["bias"]["strides"]) if "bias" in named else [0, 0, 0, 0]
    )
    if bias_shape[0] == 1:
        bias_strides[0] = 0
    constants.update(
        {
            f"stride_bias_{axis}": stride
            for axis, stride in zip("bhmn", bias_strides)
        }
    )
    forward_shared = {
        (32, 32, 16): 4096,
        (32, 32, 32): 6144,
        (32, 64, 16): 6144,
        (32, 64, 32): 8192,
        (64, 64, 16): 7168,
        (64, 64, 32): 10240,
    }
    variants = []
    for config in configurations:
        values = dict(constants)
        if fp8:
            values.update(
                BLOCK_M=16,
                BLOCK_N=32,
            )
            shared = 24576 if backward else 8192
            grid = [1, 1, 1]
        elif backward:
            values.update(
                BLOCK_M=16,
                BLOCK_N=32,
            )
            shared = plan["backward_shared"]
            grid = [(sq + 15) // 16 + (sk + 31) // 32, batch * key_heads, 1]
        else:
            values.update(config["META"])
            shared = forward_shared.get(
                (dimension, value_dimension, values["BLOCK_M"]), 0
            )
            grid = [
                (sq + values["BLOCK_M"] - 1) // values["BLOCK_M"],
                batch * heads,
                1,
            ]
        signature, arguments = [], []
        for parameter in parameters:
            name = parameter.arg
            if name.endswith("_ptr"):
                tensor = pointers.get(name)
                if tensor is None:
                    signature.append("nullopt")
                else:
                    argument = _tensor_argument(tensor)
                    if fp8 and tensor["data_type"] in {"fp8_e4m3", "fp8_e5m2"}:
                        signature.append("*i8:16")
                        argument["storage_view"] = "fp8_bytes"
                    else:
                        signature.append(_tensor_pointer_signature(tensor))
                    arguments.append(argument)
            else:
                if parameter.annotation is None:
                    raise ValueError(
                        "THead attention specialization unexpectedly has a"
                        " runtime scalar"
                    )
                signature.append(str(values[name]))
        variants.append(
            {
                "variant_id": (
                    f"tile{values['BLOCK_M']}_w{config['num_warps']}_s1"
                    if enable_autotune
                    else "default"
                ),
                "full_signature": ",".join(signature),
                "argument_count": len(arguments),
                "arguments": arguments,
                "compile_options": {
                    key: config[key]
                    for key in (
                        "num_warps",
                        "num_stages",
                        "maxnreg",
                        "ppu_compiler_options",
                    )
                },
                "launch": {
                    "grid": grid,
                    "block": [config["num_warps"] * _PPU_WARP_SIZE, 1, 1],
                    "shared_memory": shared,
                },
            }
        )
    registry_digest = _registry_sha256()
    tuning_manifest = None
    if enable_autotune:
        tuning_manifest = {
            "warmup": tuning.warmup,
            "repetitions": tuning.repetitions,
            "candidate_identity": _canonical_sha256(
                {
                    "backend": "thead",
                    "target": target,
                    "compiler_identity": identity["identity_sha256"],
                    "source_sha256": _sha256(source_bytes),
                    "registry_sha256": registry_digest,
                    "tuning_sha256": _sha256(tuning_bytes),
                    "table": tuning.table,
                    "variants": variants,
                    "request_sha256": _sha256(request_bytes),
                }
            ),
            "selection_cache": ".flagdnn-autotune-v1-stage-0.json",
        }
    materialized_name = "generated_stage_0.py"
    manifest = {
        "schema_version": 1,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request["flagdnn_version"],
        "graph_ir_schema_version": SCHEMA_VERSION,
        "backend_abi_version": 2,
        "backend": "thead",
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "engine": "libtriton_jit",
        "request_sha256": _sha256(request_bytes),
        "compiler": {
            "provider": identity["provider"],
            "provider_version": identity["provider_version"],
            "identity_sha256": identity["identity_sha256"],
        },
        "external_uids": [
            t["uid"] for t in plan["tensors"] if not t["virtual"]
        ],
        "tensor_count": len(plan["tensors"]),
        "tensors": [_manifest_tensor(t) for t in plan["tensors"]],
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
                        "registry_sha256": registry_digest,
                        "materialized_source": {
                            "path": materialized_name,
                            "size": len(source_bytes),
                            "sha256": _sha256(source_bytes),
                        },
                    },
                    "variants": variants,
                    "tuning": tuning_manifest,
                }
            ],
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError(
            "THead attention artifact output directory is invalid"
        )
    _atomic_write(output_directory / materialized_name, source_bytes)
    write_program_manifest(output_directory, manifest)
    return {
        "schema_version": 1,
        "status": "success",
        "backend": "thead",
        "provider": PROVIDER_NAME,
        "node_count": 1,
        "stage_count": 1,
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": 0,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }
