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
    empty_output_like_layout,
    require_non_overlapping_layout,
)


class NvidiaBinarySelectOperation:
    name = "binary_select"

    def __init__(self, context: NvidiaContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in CUDNN_COMPARE_DTYPES

    def run(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        prepared = self.prepare(x, y, mask)
        try:
            output = prepared.run()
            self._context.synchronize()
            return output
        finally:
            prepared.close()

    def prepare(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> PreparedCudnnOperation:
        context = self._context
        context.validate_tensor(self.name, x)
        context.validate_tensor(self.name, y)
        context.validate_tensor(self.name, mask)
        if x.device != y.device or x.device != mask.device:
            raise ValueError(
                "cuDNN binary_select inputs must use the same GPU"
            )
        if x.dtype != y.dtype:
            raise TypeError(
                "cuDNN binary_select x and y must have the same dtype"
            )
        require_non_overlapping_layout(self.name, x, y)
        if not self.supports_dtype(x.dtype):
            raise TypeError(f"cuDNN binary_select does not support {x.dtype}")
        if mask.dtype != torch.bool:
            raise TypeError(
                "cuDNN binary_select mask must have dtype torch.bool"
            )
        output_shape = tuple(
            torch.broadcast_shapes(x.shape, y.shape, mask.shape)
        )

        with torch.cuda.device(x.device):
            context.activate_stream(x.device)
            graph = cudnn_graph(x.dtype, context.handle)
            x_tensor = graph.tensor_like(x)
            y_tensor = graph.tensor_like(y)

            # cuDNN frontend represents binary_select masks using the graph's
            # numeric I/O dtype. Keep the public provider contract boolean and
            # perform this compatibility conversion outside timed execution.
            graph_mask = mask.to(dtype=x.dtype)
            mask_tensor = graph.tensor_like(graph_mask)
            try:
                output_tensor = graph.binary_select(
                    input0=x_tensor,
                    input1=y_tensor,
                    mask=mask_tensor,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name=self.name,
                )
            except TypeError:
                output_tensor = graph.binary_select(
                    a=x_tensor,
                    b=y_tensor,
                    mask=mask_tensor,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name=self.name,
                )
            output = empty_output_like_layout(x, output_shape, x.dtype)
            output_tensor.set_output(True).set_data_type(
                cudnn_data_type(x.dtype)
            ).set_dim(list(output.shape)).set_stride(list(output.stride()))
            build_cudnn_graph(graph, self.name)
            workspace = torch.empty(
                graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            exec_tensors = {
                x_tensor: x,
                y_tensor: y,
                mask_tensor: graph_mask,
                output_tensor: output,
            }
        context.last_device = x.device
        return PreparedCudnnOperation(
            graph,
            exec_tensors,
            workspace,
            output,
            context.handle,
        )
