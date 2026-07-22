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

from typing import Any, Sequence

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


def _contiguous_stride(shape: Sequence[int]) -> tuple[int, ...]:
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= int(size)
    return tuple(reversed(stride))


def _normalized_shape(x: torch.Tensor, scale: torch.Tensor) -> tuple[int, ...]:
    aligned = (1,) * (x.dim() - scale.dim()) + tuple(scale.shape)
    axes = tuple(index for index, size in enumerate(aligned) if size != 1)
    if not axes:
        axes = (x.dim() - 1,)
    trailing = tuple(range(x.dim() - len(axes), x.dim()))
    if axes != trailing:
        raise ValueError("norm scale must describe trailing dimensions")
    result = tuple(int(x.shape[index]) for index in axes)
    if scale.numel() != torch.Size(result).numel():
        raise ValueError("norm scale size does not match normalized shape")
    return result


def _broadcast_parameter(value: torch.Tensor, input_rank: int) -> torch.Tensor:
    if value.dim() > input_rank:
        raise ValueError("norm parameter rank cannot exceed input rank")
    return value.reshape((1,) * (input_rank - value.dim()) + value.shape)


def _stat_shape(
    x: torch.Tensor, normalized_shape: Sequence[int]
) -> tuple[int, ...]:
    return tuple(x.shape[: x.dim() - len(normalized_shape)]) + (1,) * len(
        normalized_shape
    )


def _scalar_tensor(graph: Any, rank: int, name: str) -> Any:
    return graph.tensor(
        dim=(1,) * rank,
        stride=(1,) * rank,
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
        name=name,
    )


def _scalar_value(value: float, rank: int) -> torch.Tensor:
    return torch.full((1,) * rank, value, dtype=torch.float32, device="cpu")


def _set_output(tensor: Any, output: torch.Tensor, dtype: torch.dtype) -> None:
    tensor.set_output(True).set_dim(tuple(output.shape)).set_stride(
        tuple(output.stride())
    ).set_data_type(cudnn_data_type(dtype))


def _prepared_graph(
    context: NvidiaContext,
    graph: Any,
    exec_tensors: dict[Any, torch.Tensor],
    graph_outputs: Sequence[Any],
    outputs: Sequence[torch.Tensor],
    device: torch.device,
) -> PreparedCudnnOperation:
    for graph_output, output in zip(graph_outputs, outputs):
        _set_output(graph_output, output, output.dtype)
        exec_tensors[graph_output] = output
    build_cudnn_graph(graph, "normalization")
    workspace = torch.empty(
        graph.get_workspace_size(), device=device, dtype=torch.uint8
    )
    context.last_device = device
    result: Any = outputs[0] if len(outputs) == 1 else tuple(outputs)
    return PreparedCudnnOperation(
        graph,
        exec_tensors,
        workspace,
        result,
        context.handle,
    )


def _validate_same_device(
    context: NvidiaContext,
    op_name: str,
    tensors: Sequence[torch.Tensor],
    *,
    allow_channels_last_input: bool = False,
) -> torch.device:
    for index, tensor in enumerate(tensors):
        context.validate_tensor(op_name, tensor)
        layout_supported = tensor.is_contiguous()
        if allow_channels_last_input and index == 0 and tensor.dim() == 4:
            layout_supported = layout_supported or tensor.is_contiguous(
                memory_format=torch.channels_last
            )
        if not layout_supported:
            raise ValueError(
                f"cuDNN {op_name} reference requires contiguous tensors"
            )
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError(f"cuDNN {op_name} tensors must share a device")
    return device


class NvidiaLayerNormOperation:
    name = "layernorm"

    def __init__(self, context: NvidiaContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in CUDNN_COMPARE_DTYPES

    def prepare(
        self,
        norm_forward_phase: Any,
        x: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        epsilon: float,
        **_: Any,
    ) -> PreparedCudnnOperation:
        del norm_forward_phase
        context = self._context
        device = _validate_same_device(context, self.name, (x, scale, bias))
        normalized = _normalized_shape(x, scale)
        if bias.numel() != scale.numel():
            raise ValueError("layernorm bias size must match scale")
        scale_view = _broadcast_parameter(scale, x.dim())
        bias_view = _broadcast_parameter(bias.reshape(scale.shape), x.dim())
        with torch.cuda.device(device):
            context.activate_stream(device)
            graph = cudnn_graph(x.dtype, context.handle)
            x_tensor = graph.tensor_like(x)
            scale_tensor = graph.tensor_like(scale_view)
            bias_tensor = graph.tensor_like(bias_view)
            epsilon_tensor = _scalar_tensor(graph, x.dim(), "epsilon")
            graph_outputs = graph.layernorm(
                cudnn.norm_forward_phase.TRAINING,
                x_tensor,
                scale_tensor,
                bias_tensor,
                epsilon_tensor,
                compute_data_type=cudnn.data_type.FLOAT,
                name=self.name,
            )
            stat_shape = _stat_shape(x, normalized)
            outputs = (
                torch.empty_like(x),
                torch.empty(stat_shape, device=device, dtype=torch.float32),
                torch.empty(stat_shape, device=device, dtype=torch.float32),
            )
            return _prepared_graph(
                context,
                graph,
                {
                    x_tensor: x,
                    scale_tensor: scale_view,
                    bias_tensor: bias_view,
                    epsilon_tensor: _scalar_value(float(epsilon), x.dim()),
                },
                graph_outputs,
                outputs,
                device,
            )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        prepared = self.prepare(*args, **kwargs)
        try:
            output = prepared.run()
            self._context.synchronize()
            return output
        finally:
            prepared.close()


class NvidiaRmsNormOperation:
    name = "rmsnorm"

    def __init__(self, context: NvidiaContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in CUDNN_COMPARE_DTYPES

    def prepare(
        self,
        norm_forward_phase: Any,
        x: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        epsilon: float = 1e-5,
        **_: Any,
    ) -> PreparedCudnnOperation:
        del norm_forward_phase
        context = self._context
        tensors = (x, scale) if bias is None else (x, scale, bias)
        device = _validate_same_device(context, self.name, tensors)
        normalized = _normalized_shape(x, scale)
        if bias is not None and bias.numel() != scale.numel():
            raise ValueError("rmsnorm bias size must match scale")
        scale_view = _broadcast_parameter(scale, x.dim())
        bias_view = (
            None
            if bias is None
            else _broadcast_parameter(bias.reshape(scale.shape), x.dim())
        )
        with torch.cuda.device(device):
            context.activate_stream(device)
            graph = cudnn_graph(x.dtype, context.handle)
            x_tensor = graph.tensor_like(x)
            scale_tensor = graph.tensor_like(scale_view)
            bias_tensor = (
                None if bias_view is None else graph.tensor_like(bias_view)
            )
            epsilon_tensor = _scalar_tensor(graph, x.dim(), "epsilon")
            graph_outputs = graph.rmsnorm(
                cudnn.norm_forward_phase.TRAINING,
                x_tensor,
                scale_tensor,
                bias=bias_tensor,
                epsilon=epsilon_tensor,
                compute_data_type=cudnn.data_type.FLOAT,
                name=self.name,
            )
            stat_shape = _stat_shape(x, normalized)
            outputs = (
                torch.empty_like(x),
                torch.empty(stat_shape, device=device, dtype=torch.float32),
            )
            exec_tensors = {
                x_tensor: x,
                scale_tensor: scale_view,
                epsilon_tensor: _scalar_value(float(epsilon), x.dim()),
            }
            if bias_tensor is not None and bias_view is not None:
                exec_tensors[bias_tensor] = bias_view
            return _prepared_graph(
                context,
                graph,
                exec_tensors,
                graph_outputs,
                outputs,
                device,
            )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        prepared = self.prepare(*args, **kwargs)
        try:
            output = prepared.run()
            self._context.synchronize()
            return output
        finally:
            prepared.close()


class NvidiaBatchNormOperation:
    name = "batchnorm"

    def __init__(self, context: NvidiaContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in CUDNN_COMPARE_DTYPES

    def prepare(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        epsilon: float,
        momentum: float,
        peer_stats: Sequence[torch.Tensor] | None = None,
        **_: Any,
    ) -> PreparedCudnnOperation:
        context = self._context
        peers = tuple(peer_stats or ())
        if len(peers) > 1:
            raise ValueError(
                "batchnorm supports at most one peer_stats tensor"
            )
        tensors = (x, scale, bias, running_mean, running_var, *peers)
        device = _validate_same_device(
            context,
            self.name,
            tensors,
            allow_channels_last_input=True,
        )
        channels = int(x.shape[1])
        if any(
            tensor.numel() != channels
            for tensor in (scale, bias, running_mean, running_var)
        ):
            raise ValueError("batchnorm channel parameter size mismatch")
        with torch.cuda.device(device):
            context.activate_stream(device)
            graph = cudnn_graph(x.dtype, context.handle)
            x_tensor = graph.tensor_like(x)
            scale_tensor = graph.tensor_like(scale)
            bias_tensor = graph.tensor_like(bias)
            mean_tensor = graph.tensor_like(running_mean)
            var_tensor = graph.tensor_like(running_var)
            epsilon_tensor = _scalar_tensor(graph, x.dim(), "epsilon")
            momentum_tensor = _scalar_tensor(graph, x.dim(), "momentum")
            peer_tensors = [graph.tensor_like(tensor) for tensor in peers]
            graph_outputs = graph.batchnorm(
                x_tensor,
                scale_tensor,
                bias_tensor,
                mean_tensor,
                var_tensor,
                epsilon_tensor,
                momentum_tensor,
                peer_stats=peer_tensors,
                compute_data_type=cudnn.data_type.FLOAT,
                name=self.name,
            )
            stat_shape = tuple(running_mean.shape)
            outputs = (
                torch.empty_like(x),
                torch.empty(stat_shape, device=device, dtype=torch.float32),
                torch.empty(stat_shape, device=device, dtype=torch.float32),
                torch.empty(stat_shape, device=device, dtype=torch.float32),
                torch.empty(stat_shape, device=device, dtype=torch.float32),
            )
            exec_tensors = {
                x_tensor: x,
                scale_tensor: scale,
                bias_tensor: bias,
                mean_tensor: running_mean,
                var_tensor: running_var,
                epsilon_tensor: _scalar_value(float(epsilon), x.dim()),
                momentum_tensor: _scalar_value(float(momentum), x.dim()),
            }
            exec_tensors.update(zip(peer_tensors, peers))
            return _prepared_graph(
                context,
                graph,
                exec_tensors,
                graph_outputs,
                outputs,
                device,
            )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        prepared = self.prepare(*args, **kwargs)
        try:
            output = prepared.run()
            self._context.synchronize()
            return output
        finally:
            prepared.close()


def create_norm_operations(context: NvidiaContext) -> tuple[Any, ...]:
    return (
        NvidiaLayerNormOperation(context),
        NvidiaRmsNormOperation(context),
        NvidiaBatchNormOperation(context),
    )
