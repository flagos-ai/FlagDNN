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

from .common import (
    CUDNN_COMPARE_DTYPES,
    NvidiaContext,
    PreparedCudnnOperation,
    cudnn,
    cudnn_data_type,
    cudnn_graph,
)


_MODE_ALIASES = {"SUM": "ADD", "MEAN": "AVG", "PROD": "MUL"}


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
    ) -> PreparedCudnnOperation:
        context = self._context
        context.validate_tensor(self.name, x)
        if dtype not in (None, x.dtype):
            raise TypeError(
                "cuDNN reduction reference requires output dtype to match "
                "input"
            )
        mode_name = _mode_name(mode)
        if not hasattr(cudnn.reduction_mode, mode_name):
            raise ValueError(f"unsupported cuDNN reduction mode: {mode_name}")
        dimensions = _dimensions(dim, x.dim())
        output_shape = []
        for axis, size in enumerate(x.shape):
            if axis in dimensions:
                if keepdim:
                    output_shape.append(1)
            else:
                output_shape.append(int(size))
        if not output_shape:
            output_shape = [1]

        with torch.cuda.device(x.device):
            context.activate_stream(x.device)
            graph = cudnn_graph(x.dtype, context.handle)
            input_tensor = graph.tensor_like(x)
            output_tensor = graph.reduction(
                input=input_tensor,
                mode=getattr(cudnn.reduction_mode, mode_name),
                compute_data_type=cudnn.data_type.FLOAT,
                name=self.name,
            )
            output = torch.empty(
                tuple(output_shape), device=x.device, dtype=x.dtype
            )
            output_tensor.set_dim(list(output.shape)).set_stride(
                list(output.stride())
            )
            output_tensor.set_output(True).set_data_type(
                cudnn_data_type(x.dtype)
            )
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            workspace = torch.empty(
                graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            exec_tensors = {input_tensor: x, output_tensor: output}
        context.last_device = x.device
        return PreparedCudnnOperation(
            graph,
            exec_tensors,
            workspace,
            output,
            context.handle,
        )
