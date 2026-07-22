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
    CUDNN_COMPARE_DTYPES,
    NvidiaContext,
    PreparedCudnnOperation,
    build_cudnn_graph,
    cudnn,
    cudnn_data_type,
    cudnn_graph,
)


_DTYPE_TO_CUDNN_INT = {
    torch.float32: 0,
    torch.float16: 2,
    torch.bfloat16: 9,
}
_ACTIVATION_TO_CUDNN_INT = {"identity": 0, "silu": 1}
_SM100_MAJOR = 10


class _PreparedSequence:
    reference_name = "cuDNN standard composite"

    def __init__(self, operations: tuple[PreparedCudnnOperation, ...]) -> None:
        self._operations = operations
        self.output = operations[-1].output
        self._closed = False

    def run(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("prepared cuDNN operation is closed")
        result = None
        for operation in self._operations:
            result = operation.run()
        self.output = result
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for operation in reversed(self._operations):
            operation.close()


class _PreparedNative:
    reference_name = "cuDNN native"

    def __init__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: str,
    ) -> None:
        self._x = x
        self._weight = weight
        self._bias = bias
        self._activation = activation
        self.output = torch.empty_like(x)
        self._closed = False

    def run(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("prepared cuDNN operation is closed")
        batch, channels, sequence = self._x.shape
        cudnn.causal_conv1d_forward(
            torch.cuda.current_stream(self._x.device).cuda_stream,
            self._x.data_ptr(),
            self._weight.data_ptr(),
            self._bias.data_ptr(),
            self.output.data_ptr(),
            batch,
            channels,
            sequence,
            self._weight.shape[1],
            _DTYPE_TO_CUDNN_INT[self._x.dtype],
            _ACTIVATION_TO_CUDNN_INT[self._activation],
        )
        return self.output

    def close(self) -> None:
        self._closed = True


class NvidiaCausalConv1dOperation:
    name = "causal_conv1d"

    def __init__(self, context: NvidiaContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in CUDNN_COMPARE_DTYPES

    def run(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        prepared = self.prepare(*args, **kwargs)
        try:
            result = prepared.run()
            self._context.synchronize()
            return result
        finally:
            prepared.close()

    def prepare(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str = "identity",
        **_: Any,
    ) -> Any:
        self._validate(x, weight, bias, activation)
        activation = str(activation).lower()
        if self._can_try_native(x, weight, bias):
            native = _PreparedNative(x, weight, bias, activation)
            try:
                native.run()
            except (
                AttributeError,
                ImportError,
                AssertionError,
                ValueError,
                TypeError,
            ):
                native.close()
            else:
                torch.cuda.synchronize(x.device)
                self._context.last_device = x.device
                return native
        return self._prepare_standard(x, weight, bias, activation)

    def _can_try_native(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> bool:
        tensors = (x, weight) if bias is None else (x, weight, bias)
        return (
            bias is not None
            and all(tensor.is_contiguous() for tensor in tensors)
            and torch.cuda.get_device_capability(x.device)[0] >= _SM100_MAJOR
            and hasattr(cudnn, "causal_conv1d_forward")
        )

    def _validate(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str,
    ) -> None:
        tensors = (x, weight) if bias is None else (x, weight, bias)
        for tensor in tensors:
            self._context.validate_tensor(self.name, tensor)
        if x.dim() != 3 or weight.dim() != 2:
            raise ValueError(
                "causal_conv1d expects x shape (batch, dim, seq_len) "
                "and weight shape (dim, kernel_size)"
            )
        if any(tensor.device != x.device for tensor in tensors):
            raise ValueError("x, weight, and bias must be on the same device")
        if any(tensor.dtype != x.dtype for tensor in tensors):
            raise TypeError("x, weight, and bias must have the same dtype")
        channels = int(x.shape[1])
        if tuple(weight.shape[:1]) != (channels,):
            raise ValueError("weight.shape[0] must match x.shape[1]")
        if bias is not None and tuple(bias.shape) != (channels,):
            raise ValueError(f"bias must have shape ({channels},)")
        if int(weight.shape[1]) <= 0:
            raise ValueError("kernel_size must be positive")
        if str(activation).lower() not in _ACTIVATION_TO_CUDNN_INT:
            raise ValueError("activation must be 'identity' or 'silu'")

    def _prepare_standard(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str,
    ) -> Any:
        context = self._context
        batch, channels, sequence = (int(value) for value in x.shape)
        kernel = int(weight.shape[1])
        weight_view = weight.reshape(channels, 1, kernel)
        bias_view = None if bias is None else bias.reshape(1, channels, 1)
        output_stride = (channels * sequence, 1, channels)

        with torch.cuda.device(x.device):
            context.activate_stream(x.device)
            graph = cudnn_graph(x.dtype, context.handle)
            x_tensor = graph.tensor_like(x)
            weight_tensor = graph.tensor_like(weight_view)
            exec_tensors = {x_tensor: x, weight_tensor: weight_view}
            output_tensor = graph.conv_fprop(
                image=x_tensor,
                weight=weight_tensor,
                stride=[1],
                pre_padding=[kernel - 1],
                post_padding=[0],
                dilation=[1],
                name="causal_conv1d_depthwise",
            )
            if bias_view is not None:
                bias_tensor = graph.tensor_like(bias_view)
                exec_tensors[bias_tensor] = bias_view
                output_tensor = graph.add(
                    a=output_tensor,
                    b=bias_tensor,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="causal_conv1d_bias",
                )
            conv_output = torch.empty_strided(
                (batch, channels, sequence),
                output_stride,
                device=x.device,
                dtype=x.dtype,
            )
            output_tensor.set_output(True).set_data_type(
                cudnn_data_type(x.dtype)
            ).set_dim(list(conv_output.shape)).set_stride(
                list(conv_output.stride())
            )
            build_cudnn_graph(graph, self.name)
            workspace = torch.empty(
                graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            exec_tensors[output_tensor] = conv_output
            conv = PreparedCudnnOperation(
                graph,
                exec_tensors,
                workspace,
                conv_output,
                context.handle,
            )
            conv.reference_name = "cuDNN standard composite"
            if activation == "identity":
                context.last_device = x.device
                return conv

            activation_input_value = conv_output.permute(0, 2, 1)
            activation_graph = cudnn_graph(x.dtype, context.handle)
            activation_input = activation_graph.tensor_like(
                activation_input_value
            )
            activation_output_tensor = activation_graph.swish(
                input=activation_input,
                compute_data_type=cudnn.data_type.FLOAT,
                name="causal_conv1d_silu",
            )
            activation_output = torch.empty_like(activation_input_value)
            activation_output_tensor.set_output(True).set_data_type(
                cudnn_data_type(x.dtype)
            ).set_dim(list(activation_output.shape)).set_stride(
                list(activation_output.stride())
            )
            build_cudnn_graph(activation_graph, f"{self.name}_silu")
            activation_workspace = torch.empty(
                activation_graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            activated = PreparedCudnnOperation(
                activation_graph,
                {
                    activation_input: activation_input_value,
                    activation_output_tensor: activation_output,
                },
                activation_workspace,
                activation_output,
                context.handle,
                result_transform=lambda result: result.permute(0, 2, 1),
            )
            activated.output = activation_output.permute(0, 2, 1)
        context.last_device = x.device
        return _PreparedSequence((conv, activated))
