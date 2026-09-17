"""Portable operator plans lowered to the Iluvatar execution-program ABI."""

from .common import _require_integer, _require_list, _require_object
from .graph_tensor import _parse_port
from .nn_common import GridSpec, KernelStagePlan, NodePlan, UNTUNED_TUNING
from .index import _expand_concatenate_group, _index_kernel_configuration
from .normalization_extended import _extended_normalization_configuration
from .causal_convolution import _causal_conv1d_kernel_configuration
from .position_embedding import _rope_kernel_configuration
from .random import _rng_kernel_configuration
from .resample import _resample_kernel_configuration
from .statistics import (
    _bn_finalize_kernel_configuration,
    _genstats_kernel_configuration,
)

_NORMALIZATION = {
    "instancenorm",
    "adalayernorm",
    "instancenorm_backward",
    "adalayernorm_backward",
    "layernorm_backward",
    "rmsnorm_backward",
    "batchnorm_backward",
}
SUPPORTED_OPERATIONS = frozenset(
    _NORMALIZATION
    | {
        "concatenate",
        "gen_index",
        "genstats",
        "bn_finalize",
        "rng",
        "rope",
        "rope_backward",
        "resample",
        "causal_conv1d",
    }
)


def _ports(operation, parameters):
    if operation in _NORMALIZATION:
        if operation.endswith("_backward"):
            stats = (
                ("inv_variance",)
                if operation == "rmsnorm_backward"
                else ("mean", "inv_variance")
            )
            return ("dy", "x", "scale") + stats, ("dx", "dscale", "dbias")
        return ("x", "scale", "bias"), ("y", "mean", "inv_variance")
    if operation in {"gen_index", "rng"}:
        return (), ("output",)
    if operation == "concatenate":
        count = _require_integer(parameters, "input_count", maximum=65536)
        return tuple(f"input_{i}" for i in range(count)), ("output",)
    if operation == "genstats":
        return ("x",), ("sum", "sq_sum")
    if operation == "bn_finalize":
        running = _require_integer(parameters, "has_running", minimum=0, maximum=1)
        return (
            ("sum", "sq_sum", "scale", "bias")
            + (
                ("previous_running_mean", "previous_running_variance")
                if running
                else ()
            ),
            ("eq_scale", "eq_bias", "mean", "inv_variance")
            + (("next_running_mean", "next_running_variance") if running else ()),
        )
    if operation == "resample":
        index = _require_integer(parameters, "generate_index", minimum=0, maximum=1)
        return ("input",), ("output", "index") if index else ("output",)
    if operation == "causal_conv1d":
        bias = _require_integer(parameters, "has_bias", minimum=0, maximum=1)
        return ("input", "weight", "bias") if bias else ("input", "weight"), ("output",)
    return (
        (("dy", "freqs"), ("dx",))
        if operation == "rope_backward"
        else (("input", "freqs"), ("output",))
    )


def parse_node(node_value, position, node_count, tensor_registry):
    node = _require_object(node_value, f"graph.nodes[{position}]")
    operation = node.get("type")
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"unsupported Iluvatar extended operation: {operation!r}")
    node_id = _require_integer(node, "id", minimum=0, maximum=node_count - 1)
    compute = node.get("compute_data_type")
    if compute not in ("float32", "float16", "bfloat16", "int32", "boolean"):
        raise ValueError(f"unsupported extended compute type: {compute!r}")
    parameters = _require_object(node.get("attributes"), "node.attributes")
    input_roles, output_roles = _ports(operation, parameters)
    tensors, roles, uids = [], [], []
    for direction, expected in [("inputs", input_roles), ("outputs", output_roles)]:
        ports = _require_list(node.get(direction), f"node.{direction}")
        if len(ports) != len(expected):
            raise ValueError(f"{operation} {direction} count is invalid")
        for port, role in zip(ports, expected):
            uid, tensor = _parse_port(port, role, direction, tensor_registry)
            tensors.append(tensor)
            roles.append(role)
            uids.append(uid)
    outputs = uids[len(input_roles) :]
    if len(set(outputs)) != len(outputs) or set(outputs).intersection(
        uids[: len(input_roles)]
    ):
        raise ValueError(f"{operation} requires distinct, non-aliased outputs")
    return dict(
        id=node_id,
        operation=operation,
        compute_data_type=compute,
        parameters=parameters,
        tensors=tensors,
        tensor_roles=roles,
        input_uids=uids[: len(input_roles)],
        output_uids=uids[len(input_roles) :],
    )


def _configuration(operation, parameters, tensors):
    if operation in _NORMALIZATION:
        return _extended_normalization_configuration(operation, parameters, tensors)
    if operation in {"concatenate", "gen_index"}:
        return _index_kernel_configuration(operation, parameters, tensors)
    if operation in {"rope", "rope_backward"}:
        return _rope_kernel_configuration(operation, parameters, tensors)
    return {
        "rng": _rng_kernel_configuration,
        "genstats": _genstats_kernel_configuration,
        "bn_finalize": _bn_finalize_kernel_configuration,
        "resample": _resample_kernel_configuration,
        "causal_conv1d": _causal_conv1d_kernel_configuration,
    }[operation](parameters, tensors)


def plan_kernel_stages(node):
    operation, parameters, tensors = (
        node["operation"],
        node["parameters"],
        node["tensors"],
    )
    groups = [dict(parameters=parameters, tensors=tensors)]
    if operation == "concatenate":
        groups = _expand_concatenate_group(groups[0])
    stages = []
    for index, group in enumerate(groups):
        function, signature, constants, grid, layout = _configuration(
            operation, group["parameters"], group["tensors"]
        )
        tensor_index = 0
        arguments = []
        for kind, payload in layout:
            if kind == "tensor":
                local = tensor_index
                tensor_index += 1
            elif kind == "tensor_alias":
                local = payload
            else:
                raise ValueError("extended kernels require a pointer-only ABI")
            uid = group["tensors"][local]["uid"]
            arguments.append(
                ("tensor", next(i for i, t in enumerate(tensors) if t["uid"] == uid))
            )
        # Common kernels use numeric constexprs and work with CoreX warp size 64.
        # Keep fixed launch metadata; NVIDIA autotune tables are not applicable.
        stages.append(
            KernelStagePlan(
                operation=operation,
                stage_name=f"{operation}_{index}",
                function_name=function,
                runtime_signature=signature,
                runtime_values={},
                constants=constants,
                default_grid=grid,
                argument_layout=tuple(arguments),
                tuning=UNTUNED_TUNING,
                tuning_key_value=1,
                default_num_warps=4,
                default_num_stages=1,
                grid_spec=GridSpec("fixed", grid),
                dependencies=(stages[-1].stage_name,) if stages else (),
            )
        )
    result = NodePlan(operation, tuple(stages))
    result.validate_dependencies()
    return result
