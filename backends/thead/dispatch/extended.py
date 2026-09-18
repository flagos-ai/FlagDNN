"""Dispatch additional operations supported by the acDNN validation stack."""

from .extended_schema import tensor_roles
from .normalization_extended import _extended_normalization_configuration
from .resample import _resample_kernel_configuration
from .statistics import (
    _genstats_kernel_configuration,
    _bn_finalize_kernel_configuration,
)

SUPPORTED_OPERATIONS = {
    "batchnorm_backward",
    "resample",
    "genstats",
    "bn_finalize",
}


def kernel_configuration(graph):
    nodes = graph["nodes"]
    if len(nodes) != 1 or nodes[0]["id"] != 0:
        raise ValueError(
            "extended THead operation requires one canonical node"
        )
    node = nodes[0]
    operation, parameters = node["type"], node["attributes"]
    allowed_compute = (
        {"float32", "float16", "bfloat16"}
        if operation == "resample"
        else {"float32"}
    )
    if (
        operation not in SUPPORTED_OPERATIONS
        or node["compute_data_type"] not in allowed_compute
    ):
        raise ValueError(
            "unsupported extended THead operation or compute type"
        )
    registry = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    tensors = []
    for direction, names in zip(
        ("inputs", "outputs"), tensor_roles(operation, parameters)
    ):
        ports = node[direction]
        if [port["name"] for port in ports] != list(names):
            raise ValueError(
                "extended THead operation port roles are inconsistent"
            )
        tensors.extend(registry[port["uid"]] for port in ports)
    if len({t["uid"] for t in tensors}) != len(tensors) or any(
        t["virtual"] for t in tensors
    ):
        raise ValueError(
            "extended THead operation requires distinct external tensors"
        )
    if operation == "batchnorm_backward":
        config = _extended_normalization_configuration(
            operation, parameters, tensors
        )
    else:
        config = {
            "resample": _resample_kernel_configuration,
            "genstats": _genstats_kernel_configuration,
            "bn_finalize": _bn_finalize_kernel_configuration,
        }[operation](parameters, tensors)
    return tensors, config
