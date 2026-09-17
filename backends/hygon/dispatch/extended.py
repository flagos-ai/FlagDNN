# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Hygon plans for index, statistics, routing and extended neural operators."""
from .common import _require_object, _require_list, POINTER_TYPES
from .extended_schema import tensor_roles
from .tensor import KernelConfiguration, TuningMetadata, GridSpec
from .index import _index_kernel_configuration, _expand_concatenate_group
from .statistics import (
    _genstats_kernel_configuration,
    _bn_finalize_kernel_configuration,
)
from .position_embedding import _rope_kernel_configuration
from .random import _rng_kernel_configuration
from .resample import _resample_kernel_configuration
from .causal_convolution import _causal_conv1d_kernel_configuration
from .normalization_extended import _extended_normalization_configuration
from .moe_matmul import _moe_matmul_configuration
from .fp8_matmul import _fp8_matmul_configuration

NORMALIZATION_OPERATIONS = frozenset(
    (
        "instancenorm",
        "adalayernorm",
        "instancenorm_backward",
        "adalayernorm_backward",
        "layernorm_backward",
        "rmsnorm_backward",
        "batchnorm_backward",
    )
)
SUPPORTED_OPERATIONS = NORMALIZATION_OPERATIONS | frozenset(
    (
        "gen_index",
        "concatenate",
        "genstats",
        "bn_finalize",
        "rope",
        "rope_backward",
        "rng",
        "resample",
        "causal_conv1d",
        "moe_grouped_matmul",
        "moe_grouped_matmul_bwd",
        "matmul_fp8",
    )
)


def parse_node(value, position, count, registry):
    node = _require_object(value, "extended graph node")
    if type(node.get("id")) is not int or not 0 <= node["id"] < count:
        raise ValueError("extended node ID is invalid")
    operation = node.get("type")
    if operation not in SUPPORTED_OPERATIONS and operation != "matmul":
        raise ValueError("unknown extended Hygon operation")
    if node.get("compute_data_type") not in POINTER_TYPES:
        raise ValueError("invalid extended operation compute type")
    parameters = dict(_require_object(node.get("attributes"), "attributes"))
    expected = tensor_roles(operation, parameters)
    tensors, roles, uids = [], [], []
    for direction, expected_names in zip(("inputs", "outputs"), expected):
        ports = _require_list(node.get(direction), direction)
        if len(ports) != len(expected_names):
            raise ValueError(
                f"{operation} {direction} count does not match its schema"
            )
        names, direction_uids = set(), []
        for port, expected_name in zip(ports, expected_names):
            port = _require_object(port, "tensor port")
            name, uid = port.get("name"), port.get("uid")
            if name != expected_name or name in names:
                raise ValueError("invalid or duplicate extended tensor role")
            if (
                type(uid) is not int
                or uid not in registry
                or not isinstance(port.get("optional", False), bool)
                or port.get("optional", False)
            ):
                raise ValueError("invalid extended tensor UID")
            names.add(name)
            roles.append(name)
            tensors.append(registry[uid])
            direction_uids.append(uid)
        uids.append(direction_uids)
    if (
        not uids[1]
        or len(set(uids[1])) != len(uids[1])
        or set(uids[0]) & set(uids[1])
    ):
        raise ValueError("extended operation outputs must be distinct")
    return dict(
        id=node["id"],
        operation=operation,
        tensors=tensors,
        tensor_roles=roles,
        input_uids=uids[0],
        output_uids=uids[1],
        compute_data_type=node["compute_data_type"],
        parameters=parameters,
    )


def expand_node(node):
    if node["operation"] != "concatenate":
        return [node]
    groups = _expand_concatenate_group(node)
    for i, group in enumerate(groups):
        group["tensor_roles"] = [
            node["tensor_roles"][i],
            node["tensor_roles"][-1],
        ]
    return groups


def handles(node):
    return node["operation"] in SUPPORTED_OPERATIONS or (
        node["operation"] == "matmul"
        and node["tensors"][0]["data_type"] in {"fp8_e4m3", "fp8_e5m2"}
    )


def kernel_operation(node):
    return "matmul_fp8" if node["operation"] == "matmul" else node["operation"]


def kernel_configuration(node):
    op, parameters, tensors = (
        kernel_operation(node),
        node["parameters"],
        node["tensors"],
    )
    if node["operation"] == "matmul":
        parameters = dict(parameters, scale_mode=0)
    if op in ("concatenate", "gen_index"):
        config = _index_kernel_configuration(op, parameters, tensors)
    elif op in NORMALIZATION_OPERATIONS:
        config = _extended_normalization_configuration(op, parameters, tensors)
    elif op.startswith("moe_grouped_matmul"):
        config = _moe_matmul_configuration(op, parameters, tensors)
    elif op in ("rope", "rope_backward"):
        config = _rope_kernel_configuration(op, parameters, tensors)
    else:
        config = {
            "genstats": _genstats_kernel_configuration,
            "bn_finalize": _bn_finalize_kernel_configuration,
            "rng": _rng_kernel_configuration,
            "resample": _resample_kernel_configuration,
            "causal_conv1d": _causal_conv1d_kernel_configuration,
            "matmul_fp8": _fp8_matmul_configuration,
        }[op](parameters, tensors)
    function, signature, constants, grid, arguments = config
    return KernelConfiguration(
        operation=op,
        function_name=function,
        runtime_signature=signature,
        constants=constants,
        default_grid=grid,
        argument_layout=tuple(arguments),
        tuning=TuningMetadata("", "", "", "fixed"),
        tuning_key_value=1,
        default_num_warps=4,
        default_num_stages=1,
        workspace_alignment=256,
        grid_spec=GridSpec("fixed", tuple(grid)),
    )
