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

import builtins
from typing import Any, Iterable

import torch

from .common import (
    CUDNN_COMPARE_DTYPES,
    NvidiaContext,
    PreparedCudnnOperation,
    cudnn,
    cudnn_data_type,
    cudnn_graph,
)


UTILITY_OPERATION_NAMES = (
    "identity",
    "reshape",
    "transpose",
    "slice",
    "concatenate",
    "gen_index",
)


def _normalize_axis(axis: int, rank: int) -> int:
    value = int(axis)
    if value < 0:
        value += rank
    if value < 0 or value >= rank:
        raise IndexError(f"axis out of range for rank {rank}: {axis}")
    return value


def _normalize_permutation(
    permutation: Iterable[int], rank: int
) -> tuple[int, ...]:
    result = tuple(_normalize_axis(dim, rank) for dim in permutation)
    if len(result) != rank or len(set(result)) != rank:
        raise ValueError(f"invalid permutation {result} for rank {rank}")
    return result


def _slice_output_shape(x: torch.Tensor, slices: Any) -> tuple[int, ...]:
    if slices is None:
        specs: tuple[Any, ...] = ()
    elif isinstance(slices, builtins.slice):
        specs = (slices,)
    else:
        specs = tuple(slices)
    specs += (builtins.slice(None),) * (x.dim() - len(specs))
    result = []
    for dim, spec in zip(x.shape, specs):
        if isinstance(spec, (tuple, list)):
            spec = builtins.slice(*spec)
        start, end, step = spec.indices(int(dim))
        result.append(len(range(start, end, step)))
    return tuple(result)


class NvidiaUtilityOperation:
    def __init__(self, name: str, context: NvidiaContext) -> None:
        if name not in UTILITY_OPERATION_NAMES:
            raise ValueError(f"unsupported cuDNN utility op: {name}")
        self.name = name
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in CUDNN_COMPARE_DTYPES

    def run(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        prepared = self.prepare(*args, **kwargs)
        try:
            output = prepared.run()
            self._context.synchronize()
            return output
        finally:
            prepared.close()

    def prepare(self, *args: Any, **kwargs: Any) -> PreparedCudnnOperation:
        context = self._context
        call_kwargs = dict(kwargs)
        if self.name == "concatenate":
            inputs = tuple(args[0])
            if not inputs:
                raise ValueError("cuDNN concatenate requires inputs")
            axis = call_kwargs.pop("axis", args[1] if len(args) > 1 else None)
            if axis is None:
                raise TypeError("concatenate requires axis")
        else:
            inputs = (args[0],)
        for tensor in inputs:
            context.validate_tensor(self.name, tensor)
        first = inputs[0]
        if not self.supports_dtype(first.dtype):
            raise TypeError(
                f"cuDNN {self.name} does not support {first.dtype}"
            )
        if any(
            item.device != first.device or item.dtype != first.dtype
            for item in inputs
        ):
            raise ValueError(
                f"cuDNN {self.name} inputs must share device and dtype"
            )

        with torch.cuda.device(first.device):
            context.activate_stream(first.device)
            graph = cudnn_graph(first.dtype, context.handle)
            input_tensors = [graph.tensor_like(item) for item in inputs]
            io_dtype = cudnn_data_type(first.dtype)

            if self.name == "identity":
                output_shape = tuple(first.shape)
                output = torch.empty_strided(
                    output_shape,
                    tuple(first.stride()),
                    device=first.device,
                    dtype=first.dtype,
                )
                output_tensor = graph.identity(
                    input=input_tensors[0],
                    compute_data_type=cudnn.data_type.FLOAT,
                    name=self.name,
                )
            elif self.name == "reshape":
                shape = call_kwargs.pop(
                    "shape", args[1] if len(args) > 1 else None
                )
                if shape is None:
                    raise TypeError("reshape requires shape")
                output_shape = tuple(int(dim) for dim in shape)
                output = torch.empty(
                    output_shape, device=first.device, dtype=first.dtype
                )
                output_tensor = graph.reshape(
                    input=input_tensors[0],
                    name=self.name,
                    reshape_mode=cudnn.reshape_mode.LOGICAL,
                )
                output_tensor.set_dim(list(output.shape)).set_stride(
                    list(output.stride())
                )
            elif self.name == "transpose":
                permutation = call_kwargs.pop(
                    "permutation", args[1] if len(args) > 1 else None
                )
                if permutation is None:
                    raise TypeError("transpose requires permutation")
                permutation = _normalize_permutation(permutation, first.dim())
                output_shape = tuple(
                    int(first.shape[index]) for index in permutation
                )
                output = torch.empty(
                    output_shape, device=first.device, dtype=first.dtype
                )
                output_tensor = graph.transpose(
                    input=input_tensors[0],
                    permutation=list(permutation),
                    compute_data_type=cudnn.data_type.FLOAT,
                    name=self.name,
                )
            elif self.name == "slice":
                slices = call_kwargs.pop(
                    "slices", args[1] if len(args) > 1 else ()
                )
                output_shape = _slice_output_shape(first, slices)
                output = torch.empty(
                    output_shape, device=first.device, dtype=first.dtype
                )
                output_tensor = graph.slice(
                    input=input_tensors[0],
                    slices=list(slices),
                    compute_data_type=cudnn.data_type.FLOAT,
                    name=self.name,
                )
            elif self.name == "concatenate":
                axis = _normalize_axis(int(axis), first.dim())
                concatenate_output_shape = list(first.shape)
                concatenate_output_shape[axis] = sum(
                    int(item.shape[axis]) for item in inputs
                )
                output = torch.empty(
                    tuple(concatenate_output_shape),
                    device=first.device,
                    dtype=first.dtype,
                )
                output_tensor = graph.concatenate(
                    inputs=input_tensors,
                    axis=axis,
                    name=self.name,
                )
            else:
                axis = call_kwargs.pop(
                    "axis", args[1] if len(args) > 1 else None
                )
                if axis is None:
                    raise TypeError("gen_index requires axis")
                requested_dtype = call_kwargs.pop("compute_data_type", None)
                if requested_dtype not in (None, first.dtype):
                    raise TypeError(
                        "cuDNN gen_index output dtype must match input"
                    )
                output = torch.empty_like(first)
                output_tensor = graph.gen_index(
                    input=input_tensors[0],
                    axis=_normalize_axis(int(axis), first.dim()),
                    compute_data_type=io_dtype,
                    name=self.name,
                )
                output_tensor.set_dim(list(output.shape)).set_stride(
                    list(output.stride())
                )

            if call_kwargs:
                unknown = ", ".join(sorted(call_kwargs))
                raise TypeError(f"unexpected {self.name} arguments: {unknown}")
            output_tensor.set_output(True).set_data_type(io_dtype)
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            workspace = torch.empty(
                graph.get_workspace_size(),
                device=first.device,
                dtype=torch.uint8,
            )
            exec_tensors = dict(zip(input_tensors, inputs))
            exec_tensors[output_tensor] = output
        context.last_device = first.device
        return PreparedCudnnOperation(
            graph,
            exec_tensors,
            workspace,
            output,
            context.handle,
        )


def create_utility_operations(
    context: NvidiaContext,
) -> tuple[NvidiaUtilityOperation, ...]:
    return tuple(
        NvidiaUtilityOperation(name, context)
        for name in UTILITY_OPERATION_NAMES
    )
