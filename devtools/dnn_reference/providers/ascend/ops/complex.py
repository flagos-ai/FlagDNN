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

"""Independent native-NPU references for complex Ascend operators.

These implementations are test and benchmark baselines.  They deliberately
live under ``devtools`` and use PyTorch's native NPU backend; FlagDNN product
kernels never dispatch through them.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

from devtools.dnn_reference.interfaces import (
    DnnReferenceNotSupportedError,
)

from ..common import AscendContext, DTYPE_CODES


_SUPPORTED_DTYPES = tuple(DTYPE_CODES)
_TOP_LEFT = "TOP_LEFT"
_BOTTOM_RIGHT = "BOTTOM_RIGHT"


class _PreparedNativeNpuOperation:
    reference_name = "PyTorch NPU native"

    def __init__(
        self,
        context: AscendContext,
        device: torch.device,
        runner: Callable[[], Any],
        output: Any,
    ) -> None:
        self._context = context
        self._device = device
        self._runner: Callable[[], Any] | None = runner
        self.output = output
        self._closed = False

    def run(self) -> Any:
        if self._closed:
            raise RuntimeError("prepared native NPU operation is closed")
        assert self._runner is not None
        npu = self._context.npu()
        with npu.device(self._device):
            self.output = self._runner()
        self._context.last_device = self._device
        return self.output

    def __call__(self) -> Any:
        return self.run()

    def close(self) -> None:
        self._closed = True
        self._runner = None


class _NativeNpuOperation:
    name: str
    prepare: Callable[..., _PreparedNativeNpuOperation]

    def __init__(self, context: AscendContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in _SUPPORTED_DTYPES

    def run(self, *args: Any, **kwargs: Any) -> Any:
        prepared = self.prepare(*args, **kwargs)
        try:
            result = prepared.run()
            self._context.synchronize()
            return result
        finally:
            prepared.close()


def _validate_tensors(
    op_name: str,
    tensors: Iterable[torch.Tensor | None],
) -> tuple[torch.Tensor, ...]:
    values = tuple(item for item in tensors if item is not None)
    if not values:
        raise TypeError(f"{op_name} requires tensor inputs")
    first = values[0]
    if not all(isinstance(item, torch.Tensor) for item in values):
        raise TypeError(f"{op_name} requires tensor inputs")
    if first.device.type != "npu":
        raise TypeError(f"{op_name} native reference requires NPU tensors")
    if first.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{op_name} does not support {first.dtype}")
    for item in values[1:]:
        if item.device != first.device:
            raise ValueError(f"{op_name} tensors must share a device")
        if item.dtype != first.dtype:
            raise TypeError(f"{op_name} tensors must share a dtype")
    return values


def _tuple_n(value: Any, rank: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),) * rank
    result = tuple(int(item) for item in value)
    if len(result) != rank:
        raise ValueError(f"{name} must have length {rank}, got {value}")
    return result


def _normalize_padding(
    rank: int,
    kernel: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    *,
    padding: Any = 0,
    pre_padding: Any = None,
    post_padding: Any = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if pre_padding is not None or post_padding is not None:
        if pre_padding is None or post_padding is None:
            raise ValueError(
                "pre_padding and post_padding must be specified together"
            )
        return (
            _tuple_n(pre_padding, rank, "pre_padding"),
            _tuple_n(post_padding, rank, "post_padding"),
        )
    if padding is None:
        padding = 0
    if isinstance(padding, str):
        mode = padding.lower()
        if mode == "valid":
            zeros = (0,) * rank
            return zeros, zeros
        if mode != "same":
            raise ValueError(f"unsupported padding mode: {padding}")
        if any(int(value) != 1 for value in stride):
            raise ValueError("padding='same' requires stride=1")
        totals = tuple(
            int(dilation[axis]) * (int(kernel[axis]) - 1)
            for axis in range(rank)
        )
        pre = tuple(value // 2 for value in totals)
        return pre, tuple(totals[axis] - pre[axis] for axis in range(rank))
    if isinstance(padding, int):
        values = (int(padding),) * rank
        return values, values
    values = tuple(int(item) for item in padding)
    if len(values) == rank:
        return values, values
    if len(values) == 2 * rank:
        return (
            tuple(values[2 * axis] for axis in range(rank)),
            tuple(values[2 * axis + 1] for axis in range(rank)),
        )
    raise ValueError(
        f"padding must have length {rank} or {2 * rank}, got {padding}"
    )


def _native_pad(
    image: torch.Tensor,
    pre: Sequence[int],
    post: Sequence[int],
) -> torch.Tensor:
    if not any(pre) and not any(post):
        return image
    pad: list[int] = []
    for before, after in reversed(tuple(zip(pre, post))):
        pad.extend((int(before), int(after)))
    return F.pad(image, tuple(pad))


def _convolution_mode(value: Any) -> str:
    mode = str(value or "CROSS_CORRELATION").rsplit(".", 1)[-1].upper()
    if mode not in ("CROSS_CORRELATION", "CONVOLUTION"):
        raise ValueError(
            "convolution_mode must be CROSS_CORRELATION or CONVOLUTION"
        )
    return mode


def _native_convolution(
    image: torch.Tensor,
    weight: torch.Tensor,
    *,
    stride: Sequence[int],
    pre: Sequence[int],
    post: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    convolution_mode: str,
) -> torch.Tensor:
    rank = weight.dim() - 2
    if rank not in (1, 2, 3):
        raise ValueError("native convolution supports 1D, 2D, and 3D")
    conv = (F.conv1d, F.conv2d, F.conv3d)[rank - 1]
    if convolution_mode == "CONVOLUTION":
        weight = weight.flip(tuple(range(2, weight.dim())))
    return conv(
        _native_pad(image, pre, post),
        weight,
        stride=tuple(stride),
        padding=0,
        dilation=tuple(dilation),
        groups=groups,
    )


def _convolution_output_shape(
    image_shape: Sequence[int],
    weight_shape: Sequence[int],
    *,
    stride: Sequence[int],
    pre: Sequence[int],
    post: Sequence[int],
    dilation: Sequence[int],
) -> tuple[int, ...]:
    rank = len(weight_shape) - 2
    spatial = tuple(int(value) for value in image_shape[-rank:])
    kernel = tuple(int(value) for value in weight_shape[-rank:])
    output_spatial = tuple(
        (
            spatial[axis]
            + int(pre[axis])
            + int(post[axis])
            - int(dilation[axis]) * (kernel[axis] - 1)
            - 1
        )
        // int(stride[axis])
        + 1
        for axis in range(rank)
    )
    if any(value <= 0 for value in output_spatial):
        raise ValueError("computed convolution output size is not positive")
    if rank == 1 and len(image_shape) == 2:
        return int(weight_shape[0]), *output_spatial
    return int(image_shape[0]), int(weight_shape[0]), *output_spatial


def _conv_parameters(
    weight_shape: Sequence[int],
    *,
    stride: Any = 1,
    padding: Any = 0,
    pre_padding: Any = None,
    post_padding: Any = None,
    dilation: Any = 1,
    groups: int = 1,
    convolution_mode: Any = "CROSS_CORRELATION",
    **_: Any,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
    str,
]:
    rank = len(weight_shape) - 2
    if rank not in (1, 2, 3):
        raise ValueError("convolution reference supports ranks 1, 2, and 3")
    stride_tuple = _tuple_n(stride, rank, "stride")
    dilation_tuple = _tuple_n(dilation, rank, "dilation")
    if any(value <= 0 for value in (*stride_tuple, *dilation_tuple)):
        raise ValueError("stride and dilation values must be positive")
    pre, post = _normalize_padding(
        rank,
        tuple(int(value) for value in weight_shape[-rank:]),
        stride_tuple,
        dilation_tuple,
        padding=padding,
        pre_padding=pre_padding,
        post_padding=post_padding,
    )
    groups = int(groups)
    if groups <= 0:
        raise ValueError("groups must be positive")
    return (
        stride_tuple,
        pre,
        post,
        dilation_tuple,
        groups,
        _convolution_mode(convolution_mode),
    )


class AscendMatmulOperation(_NativeNpuOperation):
    name = "matmul"

    def prepare(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        out_dtype: torch.dtype | None = None,
        compute_data_type: Any = "float32",
        **_: Any,
    ) -> _PreparedNativeNpuOperation:
        _validate_tensors(self.name, (a, b))
        if a.dim() < 2 or b.dim() < 2 or a.shape[-1] != b.shape[-2]:
            raise ValueError(
                f"invalid matmul shapes: {tuple(a.shape)} and {tuple(b.shape)}"
            )
        if str(compute_data_type).lower() not in (
            "float",
            "float32",
            "tf32",
            "ieee",
        ):
            raise DnnReferenceNotSupportedError(
                "Ascend matmul reference only supports float32-like compute"
            )
        output_dtype = a.dtype if out_dtype is None else out_dtype
        if output_dtype not in _SUPPORTED_DTYPES:
            raise DnnReferenceNotSupportedError(
                f"Ascend matmul output dtype is unsupported: {output_dtype}"
            )
        output_shape = (
            *torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]),
            int(a.shape[-2]),
            int(b.shape[-1]),
        )

        def runner() -> torch.Tensor:
            result = torch.matmul(a, b)
            return (
                result
                if result.dtype == output_dtype
                else result.to(output_dtype)
            )

        return _PreparedNativeNpuOperation(
            self._context,
            a.device,
            runner,
            torch.empty(output_shape, device=a.device, dtype=output_dtype),
        )


class AscendConvolutionOperation(_NativeNpuOperation):
    def __init__(self, name: str, context: AscendContext) -> None:
        if name not in ("conv_fprop", "conv_dgrad", "conv_wgrad"):
            raise ValueError(f"unsupported convolution operation: {name}")
        super().__init__(context)
        self.name = name

    def prepare(
        self, *args: Any, **kwargs: Any
    ) -> _PreparedNativeNpuOperation:
        if self.name == "conv_fprop":
            return self._prepare_fprop(*args, **kwargs)
        if self.name == "conv_dgrad":
            return self._prepare_dgrad(*args, **kwargs)
        return self._prepare_wgrad(*args, **kwargs)

    def _prepare_fprop(
        self,
        image: torch.Tensor,
        weight: torch.Tensor,
        **kwargs: Any,
    ) -> _PreparedNativeNpuOperation:
        _validate_tensors(self.name, (image, weight))
        parameters = _conv_parameters(weight.shape, **kwargs)

        def runner() -> torch.Tensor:
            return _native_convolution(
                image,
                weight,
                stride=parameters[0],
                pre=parameters[1],
                post=parameters[2],
                dilation=parameters[3],
                groups=parameters[4],
                convolution_mode=parameters[5],
            )

        output_shape = _convolution_output_shape(
            image.shape,
            weight.shape,
            stride=parameters[0],
            pre=parameters[1],
            post=parameters[2],
            dilation=parameters[3],
        )
        return _PreparedNativeNpuOperation(
            self._context,
            image.device,
            runner,
            torch.empty(output_shape, device=image.device, dtype=image.dtype),
        )

    def _prepare_dgrad(
        self,
        loss: torch.Tensor,
        weight: torch.Tensor,
        input_size: Sequence[int] | None = None,
        *,
        input_shape: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> _PreparedNativeNpuOperation:
        _validate_tensors(self.name, (loss, weight))
        requested = input_size if input_size is not None else input_shape
        if requested is None:
            raise TypeError("conv_dgrad requires input_size")
        shape = tuple(int(value) for value in requested)
        parameters = _conv_parameters(weight.shape, **kwargs)

        def runner() -> torch.Tensor:
            with torch.enable_grad():
                image = torch.zeros(
                    shape, device=loss.device, dtype=loss.dtype
                ).requires_grad_(True)
                output = _native_convolution(
                    image,
                    weight,
                    stride=parameters[0],
                    pre=parameters[1],
                    post=parameters[2],
                    dilation=parameters[3],
                    groups=parameters[4],
                    convolution_mode=parameters[5],
                )
                (gradient,) = torch.autograd.grad(
                    output, image, loss, create_graph=False
                )
            return gradient.detach()

        return _PreparedNativeNpuOperation(
            self._context,
            loss.device,
            runner,
            torch.empty(shape, device=loss.device, dtype=loss.dtype),
        )

    def _prepare_wgrad(
        self,
        image: torch.Tensor,
        loss: torch.Tensor,
        filter_size: Sequence[int] | None = None,
        *,
        weight_shape: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> _PreparedNativeNpuOperation:
        _validate_tensors(self.name, (image, loss))
        requested = filter_size if filter_size is not None else weight_shape
        if requested is None:
            raise TypeError("conv_wgrad requires filter_size")
        shape = tuple(int(value) for value in requested)
        parameters = _conv_parameters(shape, **kwargs)

        def runner() -> torch.Tensor:
            with torch.enable_grad():
                weight = torch.zeros(
                    shape, device=image.device, dtype=image.dtype
                ).requires_grad_(True)
                output = _native_convolution(
                    image,
                    weight,
                    stride=parameters[0],
                    pre=parameters[1],
                    post=parameters[2],
                    dilation=parameters[3],
                    groups=parameters[4],
                    convolution_mode=parameters[5],
                )
                (gradient,) = torch.autograd.grad(
                    output, weight, loss, create_graph=False
                )
            return gradient.detach()

        return _PreparedNativeNpuOperation(
            self._context,
            image.device,
            runner,
            torch.empty(shape, device=image.device, dtype=image.dtype),
        )


class AscendCausalConv1dOperation(_NativeNpuOperation):
    name = "causal_conv1d"

    def prepare(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str = "identity",
        **_: Any,
    ) -> _PreparedNativeNpuOperation:
        _validate_tensors(self.name, (x, weight, bias))
        if x.dim() != 3 or weight.dim() != 2:
            raise ValueError(
                "causal_conv1d expects x=(batch, channels, sequence) and "
                "weight=(channels, kernel)"
            )
        channels = int(x.shape[1])
        if tuple(weight.shape[:1]) != (channels,):
            raise ValueError("weight channels must match x")
        if bias is not None and tuple(bias.shape) != (channels,):
            raise ValueError(f"bias must have shape ({channels},)")
        normalized_activation = str(activation).lower()
        if normalized_activation not in ("identity", "silu"):
            raise ValueError("activation must be 'identity' or 'silu'")
        kernel = int(weight.shape[1])

        def runner() -> torch.Tensor:
            output = F.conv1d(
                F.pad(x, (kernel - 1, 0)),
                weight.reshape(channels, 1, kernel),
                groups=channels,
            )
            if bias is not None:
                output = output + bias.reshape(1, channels, 1)
            if normalized_activation == "silu":
                output = F.silu(output)
            return output

        return _PreparedNativeNpuOperation(
            self._context,
            x.device,
            runner,
            torch.empty(tuple(x.shape), device=x.device, dtype=x.dtype),
        )


def _normalize_alignment(value: Any) -> str:
    result = str(value or _TOP_LEFT).rsplit(".", 1)[-1].upper()
    if result not in (_TOP_LEFT, _BOTTOM_RIGHT):
        raise ValueError("diagonal_alignment must be TOP_LEFT or BOTTOM_RIGHT")
    return result


def _attention_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    diagonal_alignment: str,
    left_bound: int | None,
    right_bound: int | None,
) -> torch.Tensor | None:
    if left_bound is None and right_bound is None:
        return None
    seq_q = int(q.shape[-2])
    seq_kv = int(k.shape[-2])
    row = torch.arange(seq_q, device=q.device)[:, None]
    column = torch.arange(seq_kv, device=q.device)[None, :]
    center = row
    if diagonal_alignment == _BOTTOM_RIGHT:
        center = center + (seq_kv - seq_q)
    relative = column - center
    mask = torch.ones((seq_q, seq_kv), device=q.device, dtype=torch.bool)
    if left_bound is not None:
        mask &= relative >= 1 - int(left_bound)
    if right_bound is not None:
        mask &= relative <= int(right_bound)
    return mask


def _native_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_scale: float | None,
    bias: torch.Tensor | None,
    diagonal_alignment: str,
    left_bound: int | None,
    right_bound: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads_q = int(q.shape[1])
    heads_kv = int(k.shape[1])
    if heads_q % heads_kv != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    repeat = heads_q // heads_kv
    if repeat != 1:
        k_for_q = k.repeat_interleave(repeat, dim=1)
        v_for_q = v.repeat_interleave(repeat, dim=1)
    else:
        k_for_q, v_for_q = k, v
    scale = (
        1.0 / math.sqrt(int(q.shape[-1]))
        if attn_scale is None
        else float(attn_scale)
    )
    scores = torch.matmul(q.float(), k_for_q.float().transpose(-2, -1)) * scale
    if bias is not None:
        scores = scores + bias.float()
    mask = _attention_mask(
        q,
        k,
        diagonal_alignment=diagonal_alignment,
        left_bound=left_bound,
        right_bound=right_bound,
    )
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    stats = torch.logsumexp(scores, dim=-1, keepdim=True)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, v_for_q.float()).to(q.dtype)
    return output, stats


def _sdpa_parameters(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    attn_scale: float | None = None,
    diagonal_alignment: Any = _TOP_LEFT,
    diagonal_band_left_bound: int | None = None,
    diagonal_band_right_bound: int | None = None,
    left_bound: int | None = None,
    right_bound: int | None = None,
    use_causal_mask: bool = False,
) -> tuple[float | None, str, int | None, int | None]:
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("sdpa reference expects rank-4 q and k")
    resolved_left = (
        diagonal_band_left_bound
        if diagonal_band_left_bound is not None
        else left_bound
    )
    resolved_right = (
        diagonal_band_right_bound
        if diagonal_band_right_bound is not None
        else right_bound
    )
    if use_causal_mask:
        if resolved_right not in (None, 0):
            raise ValueError(
                "use_causal_mask conflicts with a nonzero right bound"
            )
        resolved_right = 0
    return (
        attn_scale,
        _normalize_alignment(diagonal_alignment),
        None if resolved_left is None else int(resolved_left),
        None if resolved_right is None else int(resolved_right),
    )


class AscendSdpaOperation(_NativeNpuOperation):
    name = "sdpa"

    def prepare(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        generate_stats: bool = False,
        **kwargs: Any,
    ) -> _PreparedNativeNpuOperation:
        _validate_tensors(self.name, (q, k, v, bias))
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError("sdpa expects rank-4 q, k, and v")
        if k.shape[:-1] != v.shape[:-1]:
            raise ValueError("k and v batch/head/sequence shapes must match")
        if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
            raise ValueError("q and k batch/head dimensions are incompatible")
        parameters = _sdpa_parameters(q, k, **kwargs)

        def runner() -> Any:
            output, stats = _native_sdpa(
                q,
                k,
                v,
                attn_scale=parameters[0],
                bias=bias,
                diagonal_alignment=parameters[1],
                left_bound=parameters[2],
                right_bound=parameters[3],
            )
            return (output, stats) if generate_stats else output

        output_shape = (*q.shape[:-1], int(v.shape[-1]))
        output = torch.empty(output_shape, device=q.device, dtype=q.dtype)
        placeholder: Any = output
        if generate_stats:
            placeholder = (
                output,
                torch.empty(
                    (*q.shape[:-1], 1),
                    device=q.device,
                    dtype=torch.float32,
                ),
            )
        return _PreparedNativeNpuOperation(
            self._context, q.device, runner, placeholder
        )


class AscendSdpaBackwardOperation(_NativeNpuOperation):
    name = "sdpa_backward"

    def prepare(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        dO: torch.Tensor,
        stats: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        dBias: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> _PreparedNativeNpuOperation:
        _validate_tensors(self.name, (q, k, v, o, dO, bias, dBias))
        if stats.device != q.device or stats.dtype != torch.float32:
            raise TypeError(
                "sdpa_backward stats must be float32 on the input device"
            )
        parameters = _sdpa_parameters(q, k, **kwargs)

        def runner() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            with torch.enable_grad():
                q_ref = q.detach().requires_grad_(True)
                k_ref = k.detach().requires_grad_(True)
                v_ref = v.detach().requires_grad_(True)
                bias_ref = (
                    None
                    if bias is None
                    else bias.detach().requires_grad_(True)
                )
                output, _ = _native_sdpa(
                    q_ref,
                    k_ref,
                    v_ref,
                    attn_scale=parameters[0],
                    bias=bias_ref,
                    diagonal_alignment=parameters[1],
                    left_bound=parameters[2],
                    right_bound=parameters[3],
                )
                grad_inputs: tuple[torch.Tensor, ...] = (
                    (q_ref, k_ref, v_ref)
                    if bias_ref is None
                    else (q_ref, k_ref, v_ref, bias_ref)
                )
                gradients = torch.autograd.grad(
                    output,
                    grad_inputs,
                    dO,
                    create_graph=False,
                )
            if dBias is not None:
                if bias_ref is None:
                    raise ValueError("dBias requires a bias input")
                dBias.copy_(gradients[3])
            return tuple(item.detach() for item in gradients[:3])

        placeholder = (
            torch.empty(tuple(q.shape), device=q.device, dtype=q.dtype),
            torch.empty(tuple(k.shape), device=k.device, dtype=k.dtype),
            torch.empty(tuple(v.shape), device=v.device, dtype=v.dtype),
        )
        return _PreparedNativeNpuOperation(
            self._context, q.device, runner, placeholder
        )


def create_complex_operations(
    context: AscendContext,
) -> tuple[_NativeNpuOperation, ...]:
    return (
        AscendMatmulOperation(context),
        AscendConvolutionOperation("conv_fprop", context),
        AscendConvolutionOperation("conv_dgrad", context),
        AscendConvolutionOperation("conv_wgrad", context),
        AscendCausalConv1dOperation(context),
        AscendSdpaOperation(context),
        AscendSdpaBackwardOperation(context),
    )


__all__ = (
    "AscendCausalConv1dOperation",
    "AscendConvolutionOperation",
    "AscendMatmulOperation",
    "AscendSdpaBackwardOperation",
    "AscendSdpaOperation",
    "create_complex_operations",
)
