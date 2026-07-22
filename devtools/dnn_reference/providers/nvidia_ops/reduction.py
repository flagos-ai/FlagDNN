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

from typing import Any, Iterable

import torch

from devtools.dnn_reference.interfaces import (
    DnnReferenceNotSupportedError,
    PreparedOperation,
)

from .common import (
    CUDNN_COMPARE_DTYPES,
    NvidiaContext,
    PreparedCudnnOperation,
    build_cudnn_graph,
    cudnn,
    cudnn_data_type,
    cudnn_graph,
)
from .reduction_composite import prepare_unaligned_reduction


_MODE_ALIASES = {"SUM": "ADD", "MEAN": "AVG", "PROD": "MUL"}
_POINTWISE_FALLBACK_MODES = frozenset(("ADD", "AVG", "MUL"))


def _mode_name(mode: Any) -> str:
    name = getattr(mode, "name", None)
    if name is None:
        name = str(mode).rsplit(".", 1)[-1]
    normalized = str(name).upper()
    return _MODE_ALIASES.get(normalized, normalized)


def _dimensions(dim: int | Iterable[int] | None, rank: int) -> tuple[int, ...]:
    if dim is None:
        values = tuple(range(rank))
    elif isinstance(dim, int):
        values = (dim,)
    else:
        values = tuple(int(item) for item in dim)
    result = []
    for value in values:
        if value < 0:
            value += rank
        if value < 0 or value >= rank:
            raise IndexError(f"reduction dimension out of range: {value}")
        if value not in result:
            result.append(value)
    return tuple(sorted(result))


def _output_shape(
    x: torch.Tensor, dimensions: tuple[int, ...], keepdim: bool
) -> tuple[int, ...]:
    shape = []
    for axis, size in enumerate(x.shape):
        if axis in dimensions:
            if keepdim:
                shape.append(1)
        else:
            shape.append(int(size))
    return tuple(shape)


class NvidiaReductionOperation:
    name = "reduction"

    def __init__(self, context: NvidiaContext) -> None:
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

    def prepare(
        self,
        x: torch.Tensor,
        mode: Any,
        *,
        dim: int | Iterable[int] | None = None,
        keepdim: bool = True,
        dtype: torch.dtype | None = None,
        **_: Any,
    ) -> PreparedOperation:
        context = self._context
        context.validate_tensor(self.name, x)
        if x.data_ptr() % 16:
            raise DnnReferenceNotSupportedError(
                "cuDNN reduction requires a 16-byte-aligned input " "entrance"
            )
        if dtype not in (None, x.dtype):
            raise TypeError(
                "cuDNN reduction reference requires output dtype to match "
                "input"
            )
        mode_name = _mode_name(mode)
        if not hasattr(cudnn.reduction_mode, mode_name):
            raise ValueError(f"unsupported cuDNN reduction mode: {mode_name}")
        dimensions = _dimensions(dim, x.dim())

        try:
            return self._prepare_native(
                x, mode_name, dimensions=dimensions, keepdim=keepdim
            )
        except DnnReferenceNotSupportedError:
            if (
                mode_name not in _POINTWISE_FALLBACK_MODES
                or len(dimensions) != 1
            ):
                raise
        return self._prepare_pointwise(
            x, mode_name, axis=dimensions[0], keepdim=keepdim
        )

    def _prepare_native(
        self,
        x: torch.Tensor,
        mode_name: str,
        *,
        dimensions: tuple[int, ...],
        keepdim: bool,
    ) -> PreparedCudnnOperation:
        context = self._context
        graph_input = x
        graph_dimensions = dimensions
        if (
            x.dim() < 4
            and x.is_contiguous()
            and dimensions == tuple(range(x.dim()))
        ):
            extent = x.numel()
            graph_input = x.reshape(-1).as_strided(
                (1, extent, 1, 1),
                (extent, 1, extent, extent),
            )
            graph_dimensions = (1,)
        output_shape = _output_shape(
            graph_input, graph_dimensions, keepdim=True
        )
        logical_output_shape = _output_shape(x, dimensions, keepdim)
        with torch.cuda.device(x.device):
            context.activate_stream(x.device)
            graph = cudnn_graph(x.dtype, context.handle)
            input_tensor = graph.tensor_like(graph_input)
            output_tensor = graph.reduction(
                input=input_tensor,
                mode=getattr(cudnn.reduction_mode, mode_name),
                compute_data_type=cudnn.data_type.FLOAT,
                name=self.name,
            )
            output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
            output_tensor.set_dim(list(output.shape)).set_stride(
                list(output.stride())
            )
            output_tensor.set_output(True).set_data_type(
                cudnn_data_type(x.dtype)
            )
            build_cudnn_graph(graph, self.name)
            workspace = torch.empty(
                graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            exec_tensors = {
                input_tensor: graph_input,
                output_tensor: output,
            }
        context.last_device = x.device
        transform = (
            None
            if tuple(output.shape) == logical_output_shape
            else lambda result: result.reshape(logical_output_shape)
        )
        prepared = PreparedCudnnOperation(
            graph,
            exec_tensors,
            workspace,
            output,
            context.handle,
            result_transform=transform,
        )
        if transform is not None:
            prepared.output = transform(output)
        return prepared

    def _prepare_pointwise(
        self,
        x: torch.Tensor,
        mode_name: str,
        *,
        axis: int,
        keepdim: bool,
    ) -> PreparedOperation:
        extent = int(x.shape[axis])
        if extent <= 0:
            raise DnnReferenceNotSupportedError(
                "cuDNN pointwise reduction fallback requires a nonempty axis"
            )

        context = self._context
        output_shape = tuple(
            int(size) for index, size in enumerate(x.shape) if index != axis
        )
        if not output_shape:
            raise DnnReferenceNotSupportedError(
                "cuDNN pointwise reduction fallback requires rank >= 2"
            )

        slice_stride_bytes = int(x.stride(axis)) * x.element_size()
        if extent > 1 and slice_stride_bytes % 16:
            if mode_name == "MUL":
                raise DnnReferenceNotSupportedError(
                    "cuDNN has no exact unaligned MUL reduction fallback"
                )
            return prepare_unaligned_reduction(
                context,
                x,
                mode_name,
                axis=axis,
                keepdim=keepdim,
            )

        views = tuple(x.select(axis, index) for index in range(extent))
        with torch.cuda.device(x.device):
            context.activate_stream(x.device)
            graph = cudnn_graph(x.dtype, context.handle)
            leaf_tensors = tuple(graph.tensor_like(view) for view in views)
            values = list(leaf_tensors)
            method = graph.mul if mode_name == "MUL" else graph.add
            level = 0
            while len(values) > 1:
                next_values = []
                for index in range(0, len(values), 2):
                    if index + 1 == len(values):
                        next_values.append(values[index])
                        continue
                    next_values.append(
                        method(
                            a=values[index],
                            b=values[index + 1],
                            compute_data_type=cudnn.data_type.FLOAT,
                            name=(
                                f"{self.name}_{mode_name.lower()}_{level}_"
                                f"{index // 2}"
                            ),
                        )
                    )
                values = next_values
                level += 1

            output_tensor = values[0]
            exec_tensors = dict(zip(leaf_tensors, views))
            if extent == 1:
                output_tensor = graph.identity(
                    input=output_tensor,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name=f"{self.name}_identity",
                )
            if mode_name == "AVG":
                scalar_shape = (1,) * len(output_shape)
                scale_tensor = graph.tensor(
                    dim=scalar_shape,
                    stride=scalar_shape,
                    data_type=cudnn.data_type.FLOAT,
                    is_pass_by_value=True,
                    name=f"{self.name}_average_scale",
                )
                scale_value = torch.full(
                    scalar_shape,
                    1.0 / extent,
                    dtype=torch.float32,
                    device="cpu",
                )
                exec_tensors[scale_tensor] = scale_value
                output_tensor = graph.mul(
                    a=output_tensor,
                    b=scale_tensor,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name=f"{self.name}_average",
                )

            raw_output = torch.empty(
                output_shape, device=x.device, dtype=x.dtype
            )
            output_tensor.set_output(True).set_data_type(
                cudnn_data_type(x.dtype)
            ).set_dim(list(raw_output.shape)).set_stride(
                list(raw_output.stride())
            )
            build_cudnn_graph(graph, self.name)
            workspace = torch.empty(
                graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            exec_tensors[output_tensor] = raw_output

        context.last_device = x.device
        transform = (
            (lambda result, output_axis=axis: result.unsqueeze(output_axis))
            if keepdim
            else None
        )
        prepared = PreparedCudnnOperation(
            graph,
            exec_tensors,
            workspace,
            raw_output,
            context.handle,
            result_transform=transform,
        )
        if transform is not None:
            prepared.output = transform(raw_output)
        prepared.reference_name = "cuDNN standard composite"
        return prepared
