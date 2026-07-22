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

from .common import (
    NvidiaContext,
    PreparedCudnnOperation,
    build_cudnn_graph,
    cudnn,
    cudnn_data_type,
    cudnn_graph,
)


class PreparedReductionComposite:
    reference_name = "cuDNN standard composite"

    def __init__(
        self,
        operations: tuple[PreparedCudnnOperation, ...],
        output: torch.Tensor,
    ) -> None:
        self._operations = operations
        self.output = output
        self._closed = False

    def run(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("prepared cuDNN operation is closed")
        for operation in self._operations:
            operation.run()
        return self.output

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for operation in reversed(self._operations):
            operation.close()


def _keepdim_shape(x: torch.Tensor, axis: int) -> tuple[int, ...]:
    shape = list(x.shape)
    shape[axis] = 1
    return tuple(int(value) for value in shape)


def _empty_reduction_output(x: torch.Tensor, axis: int) -> torch.Tensor:
    shape = _keepdim_shape(x, axis)
    output = torch.empty(shape, device=x.device, dtype=torch.float32)
    if x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last):
        output = output.contiguous(memory_format=torch.channels_last)
    return output


def _prepare_add_to_float(
    context: NvidiaContext,
    x: torch.Tensor,
    axis: int,
    name: str,
) -> PreparedCudnnOperation:
    graph = cudnn_graph(x.dtype, context.handle)
    input_value = graph.tensor_like(x)
    output_value = graph.reduction(
        input=input_value,
        mode=cudnn.reduction_mode.ADD,
        compute_data_type=cudnn.data_type.FLOAT,
        name=name,
    )
    output = _empty_reduction_output(x, axis)
    output_value.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(
        list(output.shape)
    ).set_stride(list(output.stride()))
    build_cudnn_graph(graph, name)
    workspace = torch.empty(
        graph.get_workspace_size(), device=x.device, dtype=torch.uint8
    )
    return PreparedCudnnOperation(
        graph,
        {input_value: x, output_value: output},
        workspace,
        output,
        context.handle,
    )


def _prepare_sum_postprocess(
    context: NvidiaContext,
    reduced: torch.Tensor,
    output_dtype: torch.dtype,
    extent: int,
    mode_name: str,
    name: str,
) -> PreparedCudnnOperation:
    graph_input = reduced
    graph = cudnn_graph(torch.float32, context.handle)
    input_value = graph.tensor_like(graph_input)
    exec_tensors: dict[Any, torch.Tensor] = {input_value: graph_input}
    if mode_name == "AVG":
        scalar_shape = (1,) * graph_input.dim()
        scale_value = graph.tensor(
            dim=scalar_shape,
            stride=scalar_shape,
            data_type=cudnn.data_type.FLOAT,
            is_pass_by_value=True,
            name=f"{name}_scale",
        )
        exec_tensors[scale_value] = torch.full(
            scalar_shape,
            1.0 / extent,
            dtype=torch.float32,
            device="cpu",
        )
        output_value = graph.mul(
            a=input_value,
            b=scale_value,
            compute_data_type=cudnn.data_type.FLOAT,
            name=f"{name}_average",
        )
    else:
        output_value = graph.identity(
            input=input_value,
            compute_data_type=cudnn.data_type.FLOAT,
            name=f"{name}_cast",
        )
    output = torch.empty(
        tuple(graph_input.shape),
        device=graph_input.device,
        dtype=output_dtype,
    )
    output_value.set_output(True).set_data_type(
        cudnn_data_type(output_dtype)
    ).set_dim(list(output.shape)).set_stride(list(output.stride()))
    build_cudnn_graph(graph, name)
    workspace = torch.empty(
        graph.get_workspace_size(),
        device=graph_input.device,
        dtype=torch.uint8,
    )
    exec_tensors[output_value] = output
    return PreparedCudnnOperation(
        graph,
        exec_tensors,
        workspace,
        output,
        context.handle,
    )


def prepare_unaligned_reduction(
    context: NvidiaContext,
    x: torch.Tensor,
    mode_name: str,
    *,
    axis: int,
    keepdim: bool,
) -> PreparedReductionComposite:
    """Build an ADD/AVG cuDNN composite from one aligned input entrance."""
    if mode_name not in ("ADD", "AVG"):
        raise ValueError(f"unsupported reduction composite: {mode_name}")

    extent = int(x.shape[axis])
    name = f"reduction_{mode_name.lower()}_unaligned"
    with torch.cuda.device(x.device):
        context.activate_stream(x.device)
        reduction = _prepare_add_to_float(context, x, axis, name)
        reduced = reduction.output.squeeze(axis)
        postprocess = _prepare_sum_postprocess(
            context,
            reduced,
            x.dtype,
            extent,
            mode_name,
            name,
        )

    context.last_device = x.device
    raw_output = postprocess.output
    output = raw_output.unsqueeze(axis) if keepdim else raw_output
    return PreparedReductionComposite((reduction, postprocess), output)
