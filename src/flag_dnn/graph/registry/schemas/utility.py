# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

import torch

from flag_dnn.graph.registry.core import OpDef, register_op_def
from flag_dnn.graph.registry.schemas._run_common import (
    _require_runtime_backend,
)
from flag_dnn.graph.registry.schemas.common import (
    _normalize_axis,
    _numel,
    _rank_of,
    _shape_like_first,
)
from flag_dnn.graph.registry.schemas.pointwise import _normalize_unary
from flag_dnn.graph.tensor import TensorSpec, canonical_dtype


def _normalize_shape_arg(shape: Any) -> tuple[int, ...]:
    if shape is None:
        raise TypeError("reshape missing required argument: shape")
    if isinstance(shape, torch.Size):
        return tuple(int(dim) for dim in shape)
    if isinstance(shape, int):
        return (int(shape),)
    return tuple(int(dim) for dim in shape)


def _normalize_permutation_arg(permutation: Any, rank: int) -> tuple[int, ...]:
    permutation = tuple(int(dim) for dim in permutation)
    if len(permutation) != rank:
        raise RuntimeError(
            f"graph transpose permutation length {len(permutation)} does not "
            f"match input rank {rank}"
        )
    normalized = tuple(
        _normalize_axis(dim, rank, "transpose") for dim in permutation
    )
    if len(set(normalized)) != rank:
        raise RuntimeError(
            f"graph transpose permutation must be unique, got {permutation}"
        )
    return normalized


def _swap_permutation(rank: int, dim0: Any, dim1: Any) -> tuple[int, ...]:
    dim0 = _normalize_axis(dim0, rank, "transpose")
    dim1 = _normalize_axis(dim1, rank, "transpose")
    permutation = list(range(rank))
    permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
    return tuple(permutation)


def _infer_reshape_shape(
    input_shape: tuple[Any, ...], requested_shape: tuple[int, ...]
) -> tuple[int, ...]:
    unknown_dims = [
        idx for idx, dim in enumerate(requested_shape) if dim == -1
    ]
    if len(unknown_dims) > 1:
        raise RuntimeError("graph reshape can only infer one dimension")
    for dim in requested_shape:
        if dim < -1:
            raise RuntimeError(f"graph reshape invalid dimension {dim}")

    input_numel = _numel(input_shape)
    known_product = 1
    for dim in requested_shape:
        if dim != -1:
            known_product *= dim

    if unknown_dims:
        if input_numel is None:
            raise NotImplementedError(
                "graph reshape cannot infer symbolic -1 dimensions"
            )
        if known_product == 0:
            raise RuntimeError(
                "graph reshape cannot infer -1 with zero known product"
            )
        if input_numel % known_product != 0:
            raise RuntimeError(
                f"graph reshape shape {requested_shape} is invalid for input "
                f"shape {input_shape}"
            )
        result = list(requested_shape)
        result[unknown_dims[0]] = input_numel // known_product
        return tuple(result)

    output_numel = _numel(requested_shape)
    if input_numel is not None and output_numel != input_numel:
        raise RuntimeError(
            f"graph reshape shape {requested_shape} is invalid for input "
            f"shape {input_shape}"
        )
    return requested_shape


def _reshape_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    inp = input_specs[0]
    out_shape = _infer_reshape_shape(inp.shape, tuple(attrs["shape"]))
    return [
        TensorSpec(
            name="",
            shape=out_shape,
            dtype=inp.dtype,
            layout="contiguous" if inp.contiguous else inp.layout,
            device=inp.device,
            contiguous=True if inp.contiguous else None,
        )
    ]


def _transpose_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    inp = input_specs[0]
    permutation = tuple(attrs["permutation"])
    identity = permutation == tuple(range(len(inp.shape)))
    return [
        TensorSpec(
            name="",
            shape=tuple(inp.shape[dim] for dim in permutation),
            dtype=inp.dtype,
            layout=inp.layout if identity else "strided",
            device=inp.device,
            contiguous=inp.contiguous if identity else False,
        )
    ]


def _normalize_slice_specs(
    slices: Any, rank: int
) -> tuple[tuple[Any, Any, Any], ...]:
    import builtins

    raw: tuple[Any, ...]
    if slices is None:
        raw = ()
    elif isinstance(slices, builtins.slice):
        raw = (slices,)
    else:
        raw = tuple(slices)
    if len(raw) > rank:
        raise IndexError(
            f"graph slice got {len(raw)} slice specs for rank {rank}"
        )

    specs: list[tuple[Any, Any, Any]] = []
    for item in raw:
        if isinstance(item, builtins.slice):
            specs.append((item.start, item.stop, item.step))
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            specs.append((item[0], item[1], item[2]))
        elif item is None:
            specs.append((None, None, None))
        else:
            raise TypeError(
                f"graph slice expects slice specs, got {type(item)!r}"
            )
    while len(specs) < rank:
        specs.append((None, None, None))
    for _, _, step in specs:
        step_value = 1 if step is None else int(step)
        if step_value <= 0:
            raise ValueError("graph slice step must be greater than zero")
    return tuple(specs)


def _resolve_slice_dim(dim: Any, spec: tuple[Any, Any, Any]) -> Any:
    start, stop, step = spec
    step_value = 1 if step is None else int(step)
    if not isinstance(dim, int):
        if start is None and stop is None and step_value == 1:
            return dim
        raise NotImplementedError(
            "graph slice symbolic dimensions only support full slices"
        )
    import builtins

    normalized = builtins.slice(start, stop, step).indices(dim)
    return len(range(*normalized))


def _slice_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    inp = input_specs[0]
    specs = tuple(attrs["slices"])
    return [
        TensorSpec(
            name="",
            shape=tuple(
                _resolve_slice_dim(dim, spec)
                for dim, spec in zip(inp.shape, specs)
            ),
            dtype=inp.dtype,
            layout="strided",
            device=inp.device,
            contiguous=False,
        )
    ]


def _concatenate_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    if not input_specs:
        raise RuntimeError("graph concatenate expects at least one input")
    first = input_specs[0]
    rank = len(first.shape)
    axis = _normalize_axis(attrs["axis"], rank, "concatenate")
    out_shape = list(first.shape)
    axis_size = first.shape[axis]

    for spec in input_specs[1:]:
        if len(spec.shape) != rank:
            raise RuntimeError("graph concatenate inputs must have same rank")
        if spec.dtype != first.dtype:
            raise RuntimeError("graph concatenate inputs must have same dtype")
        for dim_index, (expected, actual) in enumerate(
            zip(first.shape, spec.shape)
        ):
            if dim_index == axis:
                continue
            if expected != actual:
                raise RuntimeError(
                    "graph concatenate non-axis dimensions must match: "
                    f"{first.shape} vs {spec.shape}"
                )
        axis_value = spec.shape[axis]
        if not isinstance(axis_size, int) or not isinstance(axis_value, int):
            raise NotImplementedError(
                "graph concatenate symbolic axis dimensions are not enabled"
            )
        axis_size = int(axis_size) + int(axis_value)

    out_shape[axis] = axis_size
    return [
        TensorSpec(
            name="",
            shape=tuple(out_shape),
            dtype=first.dtype,
            layout="contiguous",
            device=first.device,
            contiguous=True,
        )
    ]


def _compute_dtype_or_default(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return canonical_dtype(value)
    key = str(value).lower()
    if key in ("none", "not_set", "data_type.not_set"):
        return default
    aliases = {
        "boolean": "bool",
        "data_type.boolean": "bool",
        "data_type.bfloat16": "bfloat16",
        "data_type.double": "float64",
        "data_type.float": "float32",
        "data_type.float16": "float16",
        "data_type.int32": "int32",
        "data_type.int64": "int64",
        "double": "float64",
        "float": "float32",
    }
    key = aliases.get(key, key)
    tail = key.rsplit(".", 1)[-1]
    tail = aliases.get(tail, tail)
    return canonical_dtype(tail)


def _gen_index_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    inp = input_specs[0]
    return [
        TensorSpec(
            name="",
            shape=inp.shape,
            dtype=_compute_dtype_or_default(
                attrs.get("compute_data_type"), "int32"
            ),
            layout=inp.layout,
            device=inp.device,
            contiguous=inp.contiguous,
        )
    ]


def _normalize_reshape(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 2:
        raise TypeError("reshape expects input and shape")
    params = dict(kwargs)
    if params.get("out") is not None:
        raise NotImplementedError("FlagDNN graph does not support reshape out")
    params.pop("out", None)
    x = args[0] if args else params.pop("input", None)
    if x is None:
        raise TypeError("reshape missing input tensor")
    if len(args) > 1:
        shape = args[1]
    else:
        shape = params.pop("shape", params.pop("size", None))
    attrs = {
        "shape": _normalize_shape_arg(shape),
        "name": params.pop("name", ""),
        "reshape_mode": params.pop("reshape_mode", "VIEW_ONLY"),
    }
    if params:
        raise TypeError(f"reshape got unsupported attrs: {sorted(params)}")
    return [ctx.as_value(x, "input")], attrs


def _normalize_transpose(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 3:
        raise TypeError("transpose got too many positional args")
    params = dict(kwargs)
    if params.get("out") is not None:
        raise NotImplementedError(
            "FlagDNN graph does not support transpose out"
        )
    params.pop("out", None)
    x = args[0] if args else params.pop("input", None)
    if x is None:
        raise TypeError("transpose missing input tensor")
    rank = _rank_of(x)

    if len(args) == 3:
        permutation = _swap_permutation(rank, args[1], args[2])
    elif "dim0" in params or "dim1" in params:
        if len(args) > 1:
            dim0 = args[1]
        else:
            dim0 = params.pop("dim0")
        dim1 = params.pop("dim1")
        permutation = _swap_permutation(rank, dim0, dim1)
    else:
        if len(args) > 1:
            raw_permutation = args[1]
        else:
            raw_permutation = params.pop("permutation", None)
        if raw_permutation is None:
            raise TypeError("transpose missing permutation")
        permutation = _normalize_permutation_arg(raw_permutation, rank)

    attrs = {
        "permutation": permutation,
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(f"transpose got unsupported attrs: {sorted(params)}")
    return [ctx.as_value(x, "input")], attrs


def _normalize_slice(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    params = dict(kwargs)
    if params.get("out") is not None:
        raise NotImplementedError("FlagDNN graph does not support slice out")
    params.pop("out", None)
    x = args[0] if args else params.pop("input", None)
    if x is None:
        raise TypeError("slice missing input tensor")
    if len(args) > 2:
        raw_slices = args[1:]
    elif len(args) == 2:
        raw_slices = args[1]
    else:
        raw_slices = params.pop("slices", ())
    attrs = {
        "slices": _normalize_slice_specs(raw_slices, _rank_of(x)),
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(f"slice got unsupported attrs: {sorted(params)}")
    return [ctx.as_value(x, "input")], attrs


def _normalize_concatenate(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 2:
        raise TypeError("concatenate expects inputs and axis")
    params = dict(kwargs)
    if params.get("out") is not None:
        raise NotImplementedError(
            "FlagDNN graph does not support concatenate out"
        )
    params.pop("out", None)
    inputs = args[0] if args else params.pop("inputs", None)
    if inputs is None:
        raise TypeError("concatenate missing inputs")
    inputs = tuple(inputs)
    if not inputs:
        raise RuntimeError("concatenate expects a non-empty input sequence")
    if len(args) > 1:
        axis = args[1]
    else:
        axis = params.pop("axis", None)
    if axis is None:
        raise TypeError("concatenate missing axis")
    attrs = {
        "axis": int(axis),
        "in_place_index": params.pop("in_place_index", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(f"concatenate got unsupported attrs: {sorted(params)}")
    return [
        ctx.as_value(value, f"input{index}")
        for index, value in enumerate(inputs)
    ], attrs


def _normalize_gen_index(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 2:
        raise TypeError("gen_index expects input and axis")
    params = dict(kwargs)
    if params.get("out") is not None:
        raise NotImplementedError(
            "FlagDNN graph does not support gen_index out"
        )
    params.pop("out", None)
    x = args[0] if args else params.pop("input", None)
    if x is None:
        raise TypeError("gen_index missing input tensor")
    axis = args[1] if len(args) > 1 else params.pop("axis", None)
    if axis is None:
        raise TypeError("gen_index missing axis")
    attrs = {
        "axis": _normalize_axis(axis, _rank_of(x), "gen_index"),
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(f"gen_index got unsupported attrs: {sorted(params)}")
    return [ctx.as_value(x, "input")], attrs


# --- eager-fallback run functions ---


def _run_identity(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "identity")
        return flag_ops.identity(
            inputs[0],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_reshape(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "reshape")
        return flag_ops.reshape(
            inputs[0],
            attrs["shape"],
            name=attrs.get("name", ""),
            reshape_mode=attrs.get("reshape_mode", "VIEW_ONLY"),
        )

    return run


def _run_transpose(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "transpose")
        return flag_ops.transpose(
            inputs[0],
            attrs["permutation"],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_slice(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "slice")
        return flag_ops.slice(
            inputs[0],
            attrs.get("slices", ()),
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_concatenate(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "concatenate")
        return flag_ops.concatenate(
            inputs,
            attrs["axis"],
            in_place_index=attrs.get("in_place_index"),
            name=attrs.get("name", ""),
        )

    return run


def _run_gen_index(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "gen_index")
        return flag_ops.gen_index(
            inputs[0],
            attrs["axis"],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def register(flag_ops: Any) -> None:
    """Register the utility op family (identity / reshape / transpose /
    slice / concatenate / gen_index)."""
    register_op_def(
        OpDef(
            name="identity",
            normalize=_normalize_unary(
                "identity", ("compute_data_type", "name")
            ),
            shape=_shape_like_first,
            run=_run_identity(flag_ops),
        )
    )
    register_op_def(
        OpDef(
            name="reshape",
            normalize=_normalize_reshape,
            shape=_reshape_shape,
            run=_run_reshape(flag_ops),
        )
    )
    register_op_def(
        OpDef(
            name="transpose",
            normalize=_normalize_transpose,
            shape=_transpose_shape,
            run=_run_transpose(flag_ops),
        )
    )
    register_op_def(
        OpDef(
            name="slice",
            normalize=_normalize_slice,
            shape=_slice_shape,
            run=_run_slice(flag_ops),
        )
    )
    register_op_def(
        OpDef(
            name="concatenate",
            normalize=_normalize_concatenate,
            shape=_concatenate_shape,
            run=_run_concatenate(flag_ops),
        )
    )
    register_op_def(
        OpDef(
            name="gen_index",
            normalize=_normalize_gen_index,
            shape=_gen_index_shape,
            run=_run_gen_index(flag_ops),
        )
    )


__all__ = (
    "register",
    "_normalize_shape_arg",
    "_normalize_permutation_arg",
    "_swap_permutation",
    "_infer_reshape_shape",
    "_reshape_shape",
    "_transpose_shape",
    "_normalize_slice_specs",
    "_resolve_slice_dim",
    "_slice_shape",
    "_concatenate_shape",
    "_compute_dtype_or_default",
    "_gen_index_shape",
    "_normalize_reshape",
    "_normalize_transpose",
    "_normalize_slice",
    "_normalize_concatenate",
    "_normalize_gen_index",
)
