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

import torch

from .common import (
    CUDNN_COMPARE_DTYPES,
    NvidiaContext,
    PreparedCudnnOperation,
    build_cudnn_graph,
    cudnn,
    cudnn_data_type,
    cudnn_graph,
)


class NvidiaSigmoidBackwardOperation:
    name = "sigmoid_backward"

    def __init__(self, context: NvidiaContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in CUDNN_COMPARE_DTYPES

    def run(
        self,
        loss: torch.Tensor,
        input: torch.Tensor,
    ) -> torch.Tensor:
        prepared = self.prepare(loss, input)
        try:
            output = prepared.run()
            self._context.synchronize()
            return output
        finally:
            prepared.close()

    def prepare(
        self,
        loss: torch.Tensor,
        input: torch.Tensor,
    ) -> PreparedCudnnOperation:
        context = self._context
        context.validate_tensor(self.name, loss)
        context.validate_tensor(self.name, input)
        if loss.shape != input.shape:
            raise ValueError(
                "cuDNN sigmoid_backward requires equal input shapes"
            )
        if loss.device != input.device or loss.dtype != input.dtype:
            raise ValueError(
                "cuDNN sigmoid_backward inputs must share device and dtype"
            )
        if not self.supports_dtype(input.dtype):
            raise TypeError(
                f"cuDNN sigmoid_backward does not support {input.dtype}"
            )

        with torch.cuda.device(input.device):
            context.activate_stream(input.device)
            graph = cudnn_graph(input.dtype, context.handle)
            loss_tensor = graph.tensor_like(loss)
            input_tensor = graph.tensor_like(input)
            output_tensor = graph.sigmoid_backward(
                loss=loss_tensor,
                input=input_tensor,
                compute_data_type=cudnn.data_type.FLOAT,
                name=self.name,
            )
            output_tensor.set_output(True).set_data_type(
                cudnn_data_type(input.dtype)
            )
            build_cudnn_graph(graph, self.name)
            output = torch.empty_strided(
                tuple(input.shape),
                tuple(input.stride()),
                device=input.device,
                dtype=input.dtype,
            )
            workspace = torch.empty(
                graph.get_workspace_size(),
                device=input.device,
                dtype=torch.uint8,
            )
            exec_tensors = {
                loss_tensor: loss,
                input_tensor: input,
                output_tensor: output,
            }
        context.last_device = input.device
        return PreparedCudnnOperation(
            graph,
            exec_tensors,
            workspace,
            output,
            context.handle,
        )
