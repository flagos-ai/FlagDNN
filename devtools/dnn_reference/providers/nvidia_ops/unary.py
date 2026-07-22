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


UNARY_OPERATION_NAMES = (
    "neg",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "ceil",
    "floor",
    "exp",
    "log",
    "erf",
    "sin",
    "cos",
    "tan",
    "relu",
    "sigmoid",
    "tanh",
    "logical_not",
    "leaky_relu",
    "elu",
    "gelu",
    "gelu_approx_tanh",
    "swish",
    "softplus",
)


class NvidiaUnaryOperation:
    def __init__(self, name: str, context: NvidiaContext) -> None:
        if name not in UNARY_OPERATION_NAMES:
            raise ValueError(f"unsupported generic cuDNN unary op: {name}")
        self.name = name
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        if self.name == "logical_not":
            return dtype == torch.bool
        return dtype in CUDNN_COMPARE_DTYPES

    def run(
        self,
        x: torch.Tensor,
        *,
        negative_slope: float = 0.01,
        alpha: float = 1.0,
        swish_beta: float | None = None,
        beta: float = 1.0,
        threshold: float = 20.0,
        approximate: str = "none",
    ) -> torch.Tensor:
        if self.name == "gelu" and approximate != "none":
            raise ValueError(
                "gelu provider expects approximate='none'; use "
                "gelu_approx_tanh for the tanh approximation"
            )
        prepared = self.prepare(
            x,
            negative_slope=negative_slope,
            alpha=alpha,
            swish_beta=swish_beta,
            beta=beta,
            threshold=threshold,
            approximate=approximate,
        )
        try:
            output = prepared.run()
            self._context.synchronize()
            return output
        finally:
            prepared.close()

    def prepare(
        self,
        x: torch.Tensor,
        *,
        negative_slope: float = 0.01,
        alpha: float = 1.0,
        swish_beta: float | None = None,
        beta: float = 1.0,
        threshold: float = 20.0,
        approximate: str = "none",
    ) -> PreparedCudnnOperation:
        if self.name == "gelu" and approximate != "none":
            raise ValueError(
                "gelu provider expects approximate='none'; use "
                "gelu_approx_tanh for the tanh approximation"
            )
        context = self._context
        context.validate_tensor(self.name, x)
        is_logical_not = self.name == "logical_not"
        graph_x = x.to(torch.float32) if is_logical_not else x
        with torch.cuda.device(x.device):
            context.activate_stream(x.device)
            graph = cudnn_graph(graph_x.dtype, context.handle)
            x_tensor = graph.tensor_like(graph_x)
            method_name = "cmp_eq" if is_logical_not else self.name
            graph_method = getattr(graph, method_name, None)
            if self.name == "leaky_relu" and graph_method is None:
                method_name = "relu"
                graph_method = getattr(graph, method_name, None)
            if graph_method is None:
                raise RuntimeError(
                    f"cuDNN frontend does not expose graph.{method_name}"
                )
            graph_kwargs = {
                "input": x_tensor,
                "compute_data_type": cudnn.data_type.FLOAT,
                "name": self.name,
            }
            exec_tensors = {x_tensor: graph_x}
            if is_logical_not:
                zero = torch.zeros_like(graph_x)
                zero_tensor = graph.tensor_like(zero)
                graph_kwargs["comparison"] = zero_tensor
                exec_tensors[zero_tensor] = zero
            if self.name == "leaky_relu":
                graph_kwargs["negative_slope"] = float(negative_slope)
            elif self.name == "swish":
                graph_kwargs["swish_beta"] = (
                    1.0 if swish_beta is None else float(swish_beta)
                )
            elif self.name == "elu" and float(alpha) != 1.0:
                graph_kwargs["alpha"] = float(alpha)
            elif self.name == "softplus" and (
                float(beta) != 1.0 or float(threshold) != 20.0
            ):
                graph_kwargs["beta"] = float(beta)
                graph_kwargs["threshold"] = float(threshold)
            output_tensor = graph_method(**graph_kwargs)
            output_tensor.set_output(True).set_data_type(
                cudnn_data_type(graph_x.dtype)
            )
            build_cudnn_graph(graph, self.name)
            output = torch.empty_strided(
                tuple(graph_x.shape),
                tuple(graph_x.stride()),
                device=x.device,
                dtype=graph_x.dtype,
            )
            workspace = torch.empty(
                graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            exec_tensors[output_tensor] = output
        context.last_device = x.device
        bool_output = None
        result_transform = None
        if is_logical_not:
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


def create_unary_operations(
    context: NvidiaContext,
) -> tuple[NvidiaUnaryOperation, ...]:
    return tuple(
        NvidiaUnaryOperation(name, context) for name in UNARY_OPERATION_NAMES
    )
