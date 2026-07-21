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

from typing import Union

import torch

from .common import (
    CUDNN_COMPARE_DTYPES,
    PreparedCudnnOperation,
    NvidiaContext,
    cudnn,
    cudnn_data_type,
    cudnn_graph,
)


Number = Union[int, float]


def _normalize_alpha(alpha: Number) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(
            "cuDNN Add reference alpha must be an int or float, "
            f"got {type(alpha).__name__}"
        )
    try:
        return float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "cuDNN Add reference alpha cannot be represented as a float"
        ) from exc


class NvidiaAddOperation:
    name = "add"

    def __init__(self, context: NvidiaContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in CUDNN_COMPARE_DTYPES

    def _validate(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._context.validate_tensor("Add", x)
        self._context.validate_tensor("Add", y)
        if x.device != y.device:
            raise ValueError(
                "cuDNN Add reference inputs must be on the same GPU, "
                f"got {x.device} and {y.device}"
            )
        if x.dtype != y.dtype:
            raise TypeError(
                "cuDNN Add reference inputs must have the same dtype, "
                f"got {x.dtype} and {y.dtype}"
            )

    def run(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: Number = 1,
    ) -> torch.Tensor:
        prepared = self.prepare(x, y, alpha=alpha)
        try:
            output = prepared.run()
            self._context.synchronize()
        finally:
            prepared.close()
        return output

    def prepare(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: Number = 1,
    ) -> PreparedCudnnOperation:
        alpha_value = _normalize_alpha(alpha)
        self._validate(x, y)
        try:
            output_shape = tuple(torch.broadcast_shapes(x.shape, y.shape))
        except RuntimeError as exc:
            raise ValueError(
                "cuDNN Add inputs are not broadcastable: "
                f"x_shape={tuple(x.shape)}, y_shape={tuple(y.shape)}"
            ) from exc

        context = self._context
        with torch.cuda.device(x.device):
            context.activate_stream(x.device)
            graph = cudnn_graph(x.dtype, context.handle)
            x_tensor = graph.tensor_like(x)
            y_tensor = graph.tensor_like(y)
            exec_tensors = {x_tensor: x, y_tensor: y}

            add_rhs = y_tensor
            if alpha_value != 1:
                alpha_value_tensor = torch.full_like(y, alpha_value)
                alpha_tensor = graph.tensor_like(alpha_value_tensor)
                exec_tensors[alpha_tensor] = alpha_value_tensor
                add_rhs = graph.mul(
                    a=y_tensor,
                    b=alpha_tensor,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="add_alpha_scale",
                )

            output_tensor = graph.add(
                a=x_tensor,
                b=add_rhs,
                compute_data_type=cudnn.data_type.FLOAT,
                name="add",
            )
            output_tensor.set_output(True).set_data_type(
                cudnn_data_type(x.dtype)
            )
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
            workspace = torch.empty(
                graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            exec_tensors[output_tensor] = output
        context.last_device = x.device
        return PreparedCudnnOperation(
            graph, exec_tensors, workspace, output, context.handle
        )
