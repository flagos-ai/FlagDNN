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
    PreparedCudnnOperation,
    NvidiaContext,
    build_cudnn_graph,
    cudnn,
    cudnn_data_type,
    cudnn_graph,
)


class NvidiaAbsOperation:
    name = "abs"

    def __init__(self, context: NvidiaContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in CUDNN_COMPARE_DTYPES

    def run(self, x: torch.Tensor) -> torch.Tensor:
        prepared = self.prepare(x)
        try:
            output = prepared.run()
            self._context.synchronize()
        finally:
            prepared.close()
        return output

    def prepare(self, x: torch.Tensor) -> PreparedCudnnOperation:
        context = self._context
        context.validate_tensor("Abs", x)
        with torch.cuda.device(x.device):
            context.activate_stream(x.device)
            graph = cudnn_graph(x.dtype, context.handle)
            x_tensor = graph.tensor_like(x)
            output_tensor = graph.abs(
                input=x_tensor,
                compute_data_type=cudnn.data_type.FLOAT,
                name="abs",
            )
            output_tensor.set_output(True).set_data_type(
                cudnn_data_type(x.dtype)
            )
            build_cudnn_graph(graph, self.name)
            output = torch.empty_strided(
                tuple(x.shape),
                tuple(x.stride()),
                device=x.device,
                dtype=x.dtype,
            )
            workspace = torch.empty(
                graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            exec_tensors = {x_tensor: x, output_tensor: output}
        context.last_device = x.device
        return PreparedCudnnOperation(
            graph, exec_tensors, workspace, output, context.handle
        )
