"""Lower additional public Graph operators to the MUSA execution-program ABI."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flagdnn_codegen.kernel_registry import (
    resolve_kernel_source,
    select_kernel_candidate,
)

from ..codegen.io import _compiler_entry_path
from .activation_backward import _pointwise_kernel_configuration
from .causal_convolution import _causal_conv1d_kernel_configuration
from .common import _scalar_f32_bits, _scalar_i32_bits
from .fp8_matmul import _fp8_matmul_configuration
from .graph import TARGET_PATTERN, _parse_envelope
from .index import _expand_concatenate_group, _index_kernel_configuration
from .metadata import _tensor_metadata
from .moe_matmul import _moe_matmul_configuration
from .normalization_extended import _extended_normalization_configuration
from .position_embedding import _rope_kernel_configuration
from .program import (
    AutotuneSpec,
    ExecutionPlan,
    KernelStage,
    KernelVariant,
    RuntimeArgument,
)
from .resample import _resample_kernel_configuration
from .statistics import (
    _bn_finalize_kernel_configuration,
    _genstats_kernel_configuration,
)

EXTENDED_OPERATIONS = frozenset(
    {
        "relu_backward",
        "tanh_backward",
        "elu_backward",
        "gelu_backward",
        "softplus_backward",
        "swish_backward",
        "gelu_approx_tanh_backward",
        "concatenate",
        "gen_index",
        "genstats",
        "bn_finalize",
        "resample",
        "rope",
        "rope_backward",
        "instancenorm",
        "adalayernorm",
        "instancenorm_backward",
        "adalayernorm_backward",
        "layernorm_backward",
        "rmsnorm_backward",
        "batchnorm_backward",
        "causal_conv1d",
        "matmul_fp8",
        "moe_grouped_matmul",
        "moe_grouped_matmul_bwd",
    }
)


@dataclass(frozen=True)
class ParsedExtendedRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    parameters: dict[str, Any]
    tensors: list[dict[str, Any]]
    external_binding_uids: tuple[int, ...]
    request_sha256: str


def parse_extended_request(
    request_bytes, *, expected_target, expected_identity
):
    envelope = _parse_envelope(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    if len(envelope.nodes) != 1:
        raise ValueError("extended operator expects one Graph node")
    node = envelope.nodes[0]
    if (
        node.operation not in EXTENDED_OPERATIONS
        and node.operation != "matmul"
    ):
        raise ValueError("unknown extended MThreads operation")
    if any(t.virtual for t in envelope.ordered_tensors):
        raise ValueError("extended operator tensors must be externally bound")
    registry = {
        t.uid: {
            "uid": t.uid,
            "data_type": t.data_type,
            "dimensions": list(t.dimensions),
            "strides": list(t.strides),
            "alignment": t.alignment,
            "virtual": t.virtual,
        }
        for t in envelope.ordered_tensors
    }
    uids, tensors, _ = _tensor_metadata(
        envelope.graph["nodes"][0], node.operation, registry
    )
    if set(uids) != set(registry):
        raise ValueError("extended Graph has unreferenced tensors")
    return ParsedExtendedRequest(
        envelope.version,
        envelope.target,
        envelope.identity,
        envelope.autotune,
        node.id,
        node.operation,
        node.attributes,
        tensors,
        tuple(t.uid for t in envelope.ordered_tensors),
        hashlib.sha256(request_bytes).hexdigest(),
    )


def _configuration(operation, parameters, tensors, architecture):
    if operation.endswith("_backward") and operation in {
        "relu_backward",
        "tanh_backward",
        "elu_backward",
        "gelu_backward",
        "softplus_backward",
        "swish_backward",
        "gelu_approx_tanh_backward",
    }:
        return _pointwise_kernel_configuration(operation, parameters, tensors)
    if operation in {"concatenate", "gen_index"}:
        return _index_kernel_configuration(operation, parameters, tensors)
    if operation == "genstats":
        return _genstats_kernel_configuration(parameters, tensors)
    if operation == "bn_finalize":
        return _bn_finalize_kernel_configuration(parameters, tensors)
    if operation == "resample":
        return _resample_kernel_configuration(parameters, tensors)
    if operation in {"rope", "rope_backward"}:
        return _rope_kernel_configuration(operation, parameters, tensors)
    if operation == "causal_conv1d":
        return _causal_conv1d_kernel_configuration(parameters, tensors)
    if operation in {"matmul_fp8", "matmul"}:
        return _fp8_matmul_configuration(
            {"scale_mode": 0, **parameters}, tensors, architecture
        )
    if operation in {"moe_grouped_matmul", "moe_grouped_matmul_bwd"}:
        return _moe_matmul_configuration(
            operation, parameters, tensors, architecture
        )
    return _extended_normalization_configuration(
        operation, parameters, tensors
    )


def plan_extended(
    request: ParsedExtendedRequest, source_sha256: str
) -> ExecutionPlan:
    candidate = select_kernel_candidate(
        "mthreads",
        "matmul_fp8" if request.operation == "matmul" else request.operation,
    )
    path = resolve_kernel_source(_compiler_entry_path(), candidate)
    definitions = {
        n.name: n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
    }
    source = "kernels/" + candidate.source
    groups = [{"parameters": request.parameters, "tensors": request.tensors}]
    if request.operation == "concatenate":
        groups = _expand_concatenate_group(groups[0])
    target = TARGET_PATTERN.fullmatch(request.target)
    if target is None:
        raise ValueError("extended operator target is invalid")
    architecture = int(target.group(1))
    stages = []
    for stage_id, group in enumerate(groups):
        function, signature, constants, grid, layout = _configuration(
            request.operation,
            group["parameters"],
            group["tensors"],
            architecture,
        )
        if function not in candidate.functions:
            raise ValueError("extended kernel is not registered")
        definition = definitions[function]
        defaults = (
            dict(
                zip(
                    [a.arg for a in definition.args.args][
                        -len(definition.args.defaults) :
                    ],
                    definition.args.defaults,
                )
            )
            if definition.args.defaults
            else {}
        )
        arguments, tokens, tensor_index, runtime_index = [], [], 0, 0
        tensors = group["tensors"]
        for argument in definition.args.args:
            name = argument.arg
            if name not in signature:
                value = (
                    constants[name]
                    if name in constants
                    else ast.literal_eval(defaults[name])
                )
                tokens.append(
                    str(int(value)) if isinstance(value, bool) else str(value)
                )
                continue
            kind, semantic = layout[runtime_index]
            runtime_index += 1
            token = signature[name]
            if kind in {"tensor", "tensor_alias"}:
                index = tensor_index if kind == "tensor" else int(semantic)
                if kind == "tensor":
                    tensor_index += 1
                tensor = tensors[index]
                if kind == "tensor_alias":
                    from .metadata import TRITON_POINTER_TYPES

                    token = TRITON_POINTER_TYPES[tensor["data_type"]]
                if tensor["alignment"] >= 16:
                    token += ":16"
                arguments.append(
                    RuntimeArgument("tensor", name, tensor["uid"], None)
                )
            elif kind in {"scalar_i32", "scalar_f32"}:
                value = group["parameters"][semantic]
                bits = (
                    _scalar_i32_bits(value)
                    if kind == "scalar_i32"
                    else _scalar_f32_bits(value)
                )
                arguments.append(RuntimeArgument(kind, name, None, bits))
            else:
                raise ValueError(
                    f"unsupported MThreads extended ABI kind {kind}"
                )
            tokens.append(token)
        if runtime_index != len(layout):
            raise ValueError("extended ABI count differs")
        variant = KernelVariant(
            "default",
            source,
            source_sha256,
            function,
            ",".join(tokens),
            tuple(grid),
            4,
            1,
            tuple(arguments),
        )
        stages.append(
            KernelStage(
                stage_id,
                request.node_id,
                request.operation,
                () if stage_id == 0 else (stage_id - 1,),
                source,
                function,
                (variant,),
                AutotuneSpec(False, 1, 1, f"tuning/stage-{stage_id}.json"),
            )
        )
    return ExecutionPlan(
        tuple(stages), request.external_binding_uids, 4096, 256
    )
