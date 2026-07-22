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
    NvidiaContext,
    PreparedCudnnOperation,
    build_cudnn_graph,
    cudnn,
    cudnn_data_type,
    cudnn_graph,
    empty_output_like_layout,
    require_non_overlapping_layout,
)


Number = Union[int, float]
BINARY_OPERATION_NAMES = (
    "sub",
    "mul",
    "div",
    "pow",
    "max",
    "min",
    "mod",
    "add_square",
    "scale",
    "cmp_eq",
    "cmp_neq",
    "cmp_gt",
    "cmp_ge",
    "cmp_lt",
    "cmp_le",
    "logical_and",
    "logical_or",
)

_LOGICAL_METHODS = {
    "logical_and": "mul",
    "logical_or": "add",
}


class NvidiaBinaryOperation:
    def __init__(self, name: str, context: NvidiaContext) -> None:
        if name not in BINARY_OPERATION_NAMES:
            raise ValueError(f"unsupported generic cuDNN binary op: {name}")
        self.name = name
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        if self.name in _LOGICAL_METHODS:
            return dtype == torch.bool
        return dtype in CUDNN_COMPARE_DTYPES

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
            return output
        finally:
            prepared.close()

    def prepare(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: Number = 1,
    ) -> PreparedCudnnOperation:
        context = self._context
        context.validate_tensor(self.name, x)
        context.validate_tensor(self.name, y)
        if x.device != y.device or x.dtype != y.dtype:
            raise ValueError(
                f"cuDNN {self.name} inputs must share device and dtype"
            )
        require_non_overlapping_layout(self.name, x, y)
        output_shape = tuple(torch.broadcast_shapes(x.shape, y.shape))
        is_logical = self.name in _LOGICAL_METHODS
        graph_x = x.to(torch.float32) if is_logical else x
        graph_y = y.to(torch.float32) if is_logical else y
        with torch.cuda.device(x.device):
            context.activate_stream(x.device)
            graph = cudnn_graph(graph_x.dtype, context.handle)
            x_tensor = graph.tensor_like(graph_x)
            y_tensor = graph.tensor_like(graph_y)
            exec_tensors = {x_tensor: graph_x, y_tensor: graph_y}

            rhs = y_tensor
            if self.name == "sub" and float(alpha) != 1.0:
                alpha_value = torch.full_like(y, float(alpha))
                alpha_tensor = graph.tensor_like(alpha_value)
                exec_tensors[alpha_tensor] = alpha_value
                rhs = graph.mul(
                    a=y_tensor,
                    b=alpha_tensor,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="sub_alpha_scale",
                )

            if self.name in ("sub", "mul", "div", "scale"):
                method_name = "mul" if self.name == "scale" else self.name
                output_tensor = getattr(graph, method_name)(
                    a=x_tensor,
                    b=rhs,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name=self.name,
                )
            elif self.name in ("pow", "max", "min"):
                output_tensor = getattr(graph, self.name)(
                    input0=x_tensor,
                    input1=y_tensor,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name=self.name,
                )
            elif self.name == "add_square":
                output_tensor = graph.add_square(
                    a=x_tensor,
                    b=y_tensor,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name=self.name,
                )
            elif self.name.startswith("cmp_"):
                output_tensor = getattr(graph, self.name)(
                    input=x_tensor,
                    comparison=y_tensor,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name=self.name,
                )
            elif self.name in _LOGICAL_METHODS:
                output_tensor = getattr(graph, _LOGICAL_METHODS[self.name])(
                    a=x_tensor,
                    b=y_tensor,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name=self.name,
                )
            else:
                try:
                    output_tensor = graph.mod(
                        input0=x_tensor,
                        input1=y_tensor,
                        compute_data_type=cudnn.data_type.FLOAT,
                        name=self.name,
                    )
                except TypeError:
                    output_tensor = graph.mod(
                        a=x_tensor,
                        b=y_tensor,
                        compute_data_type=cudnn.data_type.FLOAT,
                        name=self.name,
                    )

            result_is_bool = self.name.startswith("cmp_") or is_logical
            output_dtype = graph_x.dtype
            output = empty_output_like_layout(
                graph_x, output_shape, output_dtype
            )
            output_tensor.set_output(True).set_data_type(
                cudnn_data_type(output_dtype)
            ).set_dim(list(output.shape)).set_stride(list(output.stride()))
            build_cudnn_graph(graph, self.name)
            workspace = torch.empty(
                graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            exec_tensors[output_tensor] = output
        context.last_device = x.device
        bool_output = None
        result_transform = None
        if result_is_bool:
            bool_output = torch.empty_strided(
                tuple(output.shape),
                tuple(output.stride()),
                device=output.device,
                dtype=torch.bool,
            )

            def transform_result(result):
                return torch.ne(result, 0, out=bool_output)

            result_transform = transform_result
        prepared = PreparedCudnnOperation(
            graph,
            exec_tensors,
            workspace,
            output,
            context.handle,
            result_transform=result_transform,
        )
        if bool_output is not None:
            prepared.output = bool_output
        return prepared


def create_binary_operations(
    context: NvidiaContext,
) -> tuple[NvidiaBinaryOperation, ...]:
    return tuple(
        NvidiaBinaryOperation(name, context) for name in BINARY_OPERATION_NAMES
    )
