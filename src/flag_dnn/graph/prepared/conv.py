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

from typing import Any, Optional, Sequence, cast

import torch

from flag_dnn import runtime
from flag_dnn.graph.prepared import (
    PreparedSingleKernelRunSpec,
    PreparedSingleKernelSpec,
    PreparedTensorCache,
    RunFn,
    get_cached_empty_tensor,
    get_prepared_output,
    make_single_kernel_run_fn,
    register_prepared_run_fn,
    runtime_tensor_checks_from_specs,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _require_runtime_backend,
    _static_shape,
)
from flag_dnn.graph.tensor import TensorSpec, torch_dtype

# Convolution helpers


def _conv_rank(image: TensorSpec, weight: TensorSpec) -> int:
    image_rank = len(image.shape)
    weight_rank = len(weight.shape)
    if image_rank == 2 and weight_rank == 3:
        return 1
    if image_rank >= 3 and image_rank == weight_rank:
        return image_rank - 2
    return -1


def _tuple_n(value: Any, rank: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),) * rank
    result = tuple(int(item) for item in value)
    if len(result) != rank:
        raise RuntimeError(f"{name} must have length {rank}, got {value}")
    return result


def _direct_padding(
    rank: int,
    padding: Any,
    pre_padding: Any,
    post_padding: Any,
) -> Any:
    if pre_padding is None and post_padding is None:
        if padding is None:
            return 0
        if rank == 1 and not isinstance(padding, str):
            return _tuple_n(padding, 1, "padding")[0]
        return padding

    pre = _tuple_n(pre_padding, rank, "pre_padding")
    post = _tuple_n(post_padding, rank, "post_padding")
    if rank == 1:
        return (pre[0], post[0])
    if rank == 2:
        return (pre[0], post[0], pre[1], post[1])
    return (pre[0], post[0], pre[1], post[1], pre[2], post[2])


def _is_cross_correlation(convolution_mode: Any) -> bool:
    if convolution_mode is None:
        return True
    mode = str(convolution_mode).rsplit(".", 1)[-1].upper()
    return mode == "CROSS_CORRELATION"


# Convolution prepared paths


def _prepare_backend_conv(
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    prepare = runtime.get_backend_hook("prepare_conv")
    if prepare is None:
        return None
    return prepare(op_type, attrs, input_specs, default_run_fn)


@register_prepared_run_fn("causal_conv1d")
def _prepare_causal_conv1d(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    has_bias = bool(attrs.get("has_bias"))
    expected_inputs = 3 if has_bias else 2
    if len(input_specs) != expected_inputs or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None

    activation = str(attrs.get("activation", "identity")).lower()
    if activation not in ("identity", "silu"):
        return None

    x_spec, weight_spec = input_specs[:2]
    x_shape = _static_shape(x_spec)
    weight_shape = _static_shape(weight_spec)
    if (
        x_shape is None
        or weight_shape is None
        or len(x_shape) != 3
        or len(weight_shape) != 2
        or x_spec.stride is None
        or weight_spec.stride is None
        or x_spec.dtype not in ("float16", "bfloat16", "float32")
        or weight_spec.dtype != x_spec.dtype
    ):
        return None

    batch, channels, sequence = x_shape
    weight_channels, kernel_size = weight_shape
    if (
        batch <= 0
        or channels <= 0
        or sequence <= 0
        or kernel_size <= 0
        or weight_channels != channels
    ):
        return None
    if has_bias:
        bias_spec = input_specs[2]
        bias_shape = _static_shape(bias_spec)
        if (
            bias_shape != (channels,)
            or bias_spec.stride is None
            or bias_spec.dtype != x_spec.dtype
        ):
            return None

    tensor_indices = tuple(range(expected_inputs))
    checks = runtime_tensor_checks_from_specs(
        input_specs,
        tensor_indices,
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    output_dtype = torch_dtype(x_spec.dtype)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        source = inputs[0]
        assert isinstance(source, torch.Tensor)
        key = (
            source.device.type,
            source.device.index,
            output_dtype,
            x_shape,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                x_shape, device=source.device, dtype=output_dtype
            ),
        )

    dtype_id = {
        "float16": 0,
        "bfloat16": 1,
        "float32": 2,
    }[x_spec.dtype]

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        x = inputs[0]
        weight = inputs[1]
        bias = inputs[2] if has_bias else output
        assert isinstance(x, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        assert isinstance(bias, torch.Tensor)
        return (
            x,
            weight,
            bias,
            output,
            sequence,
            sequence,
            channels,
            dtype_id,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            weight.stride(0),
            weight.stride(1),
            bias.stride(0) if has_bias else 0,
            output.stride(0),
            output.stride(1),
            output.stride(2),
        )

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        block_l = int(meta["BLOCK_L"])
        block_c = int(meta["BLOCK_C"])
        return (
            (sequence + block_l - 1) // block_l,
            (channels + block_c - 1) // block_c,
            batch,
        )

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        block_c = int(constexprs["BLOCK_C"])
        block_l = int(constexprs["BLOCK_L"])
        return (
            (sequence + block_l - 1) // block_l,
            (channels + block_c - 1) // block_c,
            batch,
        ), (
            1,
            kernel_size - 1,
            1,
            kernel_size,
            has_bias,
            activation,
            block_c,
            block_l,
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        tensors = inputs[:expected_inputs]
        return all(
            isinstance(value, torch.Tensor)
            and value.device == tensors[0].device
            for value in tensors
        )

    from flag_dnn.ops.conv1d import conv1d_depthwise_kernel

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=conv1d_depthwise_kernel,
                grid=grid,
                static_args=(1, kernel_size - 1, 1, kernel_size),
                constexpr_kwargs={
                    "HAS_BIAS": has_bias,
                    "ACTIVATION": activation,
                },
                build_cached_call=build_cached_call,
            ),
            input_checks=checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


@register_prepared_run_fn("conv_dgrad")
def _prepare_conv_dgrad(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    backend_run_fn = _prepare_backend_conv(
        "conv_dgrad", attrs, input_specs, default_run_fn
    )
    if backend_run_fn is not None:
        return backend_run_fn

    del default_run_fn
    if len(input_specs) < 2:
        return None
    if not all(_is_runtime_device_spec(spec) for spec in input_specs[:2]):
        return None

    from flag_dnn.ops.conv_dgrad import conv_dgrad

    input_size = tuple(int(dim) for dim in attrs["input_size"])
    padding = attrs.get("padding")
    pre_padding = attrs.get("pre_padding")
    post_padding = attrs.get("post_padding")
    stride = attrs.get("stride", 1)
    dilation = attrs.get("dilation", 1)
    convolution_mode = attrs.get("convolution_mode", "CROSS_CORRELATION")
    compute_data_type = attrs.get("compute_data_type")
    name = attrs.get("name", "")
    groups = int(attrs.get("groups", 1))
    output_cache: PreparedTensorCache = {}

    def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, "conv_dgrad")
        loss = inputs[0]
        key = (
            loss.device.type,
            loss.device.index,
            loss.dtype,
            input_size,
        )
        output = get_cached_empty_tensor(
            output_cache,
            key,
            input_size,
            device=loss.device,
            dtype=loss.dtype,
        )
        return conv_dgrad(
            loss,
            inputs[1],
            input_size=input_size,
            padding=padding,
            pre_padding=pre_padding,
            post_padding=post_padding,
            stride=stride,
            dilation=dilation,
            convolution_mode=convolution_mode,
            compute_data_type=compute_data_type,
            name=name,
            groups=groups,
            _output=output,
        )

    return run


@register_prepared_run_fn("conv_wgrad")
def _prepare_conv_wgrad(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    backend_run_fn = _prepare_backend_conv(
        "conv_wgrad", attrs, input_specs, default_run_fn
    )
    if backend_run_fn is not None:
        return backend_run_fn

    del default_run_fn
    if len(input_specs) < 2:
        return None
    if not all(_is_runtime_device_spec(spec) for spec in input_specs[:2]):
        return None

    from flag_dnn.ops.conv_wgrad import (
        _conv_wgrad2d_1x1_atomic_nodiv_kernel,
        _conv_wgrad2d_1x1_reduce_kernel,
        _conv_wgrad2d_1x1_split_nodiv_kernel,
        _conv_wgrad2d_reduce_kernel,
        _conv_wgrad2d_stride2_3tap_atomic_kernel,
        _conv_wgrad2d_stride2_row4_split_kernel,
        _conv_wgrad_zero_kernel,
        conv_wgrad,
    )

    filter_size = tuple(int(dim) for dim in attrs["filter_size"])
    padding = attrs.get("padding")
    pre_padding = attrs.get("pre_padding")
    post_padding = attrs.get("post_padding")
    stride = attrs.get("stride", 1)
    dilation = attrs.get("dilation", 1)
    convolution_mode = attrs.get("convolution_mode", "CROSS_CORRELATION")
    compute_data_type = attrs.get("compute_data_type")
    name = attrs.get("name", "")
    groups = int(attrs.get("groups", 1))
    output_cache: PreparedTensorCache = {}
    workspace_cache: PreparedTensorCache = {}
    image_spec = input_specs[0]
    loss_spec = input_specs[1]
    if image_spec.device is not None and all(
        isinstance(dim, int) for dim in image_spec.shape + loss_spec.shape
    ):
        device = torch.device(image_spec.device)
        dtype = torch_dtype(image_spec.dtype)
        output_cache[(device.type, device.index, dtype, filter_size)] = (
            torch.empty(filter_size, device=device, dtype=dtype)
        )
        image_shape = tuple(int(dim) for dim in image_spec.shape)
        loss_shape = tuple(int(dim) for dim in loss_spec.shape)
        if (
            image_shape == (8, 64, 28, 28)
            and loss_shape == (8, 128, 28, 28)
            and filter_size == (128, 64, 1, 1)
            and groups == 1
        ):
            if dtype in (torch.float16, torch.bfloat16):
                partial_dtype = (
                    dtype if dtype == torch.float16 else torch.float32
                )
                workspace_cache[
                    (
                        device.type,
                        device.index,
                        partial_dtype,
                        ("2d_1x1_nodiv_split_v7", 32, 128, 64),
                    )
                ] = torch.empty(
                    (32, 128, 64), device=device, dtype=partial_dtype
                )
        if (
            image_shape == (8, 64, 56, 56)
            and loss_shape == (8, 128, 28, 28)
            and filter_size == (128, 64, 3, 3)
            and groups == 1
        ):
            if dtype in (torch.float16, torch.bfloat16):
                partial_dtype = (
                    dtype if dtype == torch.float16 else torch.float32
                )
                workspace_cache[
                    (
                        device.type,
                        device.index,
                        partial_dtype,
                        ("2d_stride2_row4_v1", 8, 128, 64, 9),
                    )
                ] = torch.empty(
                    (8, 128, 64, 9), device=device, dtype=partial_dtype
                )

        if (
            len(image_shape) == 4
            and len(loss_shape) == 4
            and len(filter_size) == 4
        ):
            stride_tuple = _tuple_n(stride, 2, "stride")
            dilation_tuple = _tuple_n(dilation, 2, "dilation")
            if pre_padding is None and post_padding is None:
                pad = _tuple_n(0 if padding is None else padding, 2, "padding")
                pre = post = pad
            else:
                pre = _tuple_n(pre_padding, 2, "pre_padding")
                post = _tuple_n(post_padding, 2, "post_padding")
            mode = str(convolution_mode).rsplit(".", 1)[-1].upper()
            exact_1x1 = (
                image_shape == (8, 64, 28, 28)
                and loss_shape == (8, 128, 28, 28)
                and filter_size == (128, 64, 1, 1)
                and stride_tuple == (1, 1)
                and pre == (0, 0)
                and post == (0, 0)
                and dilation_tuple == (1, 1)
                and mode == "CROSS_CORRELATION"
                and groups == 1
            )
            if exact_1x1:
                output = output_cache[
                    (device.type, device.index, dtype, filter_size)
                ]
                if dtype == torch.float32:

                    def run_exact_1x1_atomic(
                        inputs: Sequence[Any], _attrs: dict[str, Any]
                    ) -> Any:
                        image, loss = inputs
                        _conv_wgrad_zero_kernel[(8,)](
                            output,
                            8192,
                            BLOCK=1024,
                            num_warps=4,
                        )
                        _conv_wgrad2d_1x1_atomic_nodiv_kernel[(8, 32)](
                            image,
                            loss,
                            output,
                            784,
                            64,
                            128,
                            50176,
                            784,
                            100352,
                            784,
                            64,
                            1,
                            4,
                            BLOCK_CO=16,
                            BLOCK_CI=64,
                            BLOCK_M=64,
                            num_warps=4,
                            num_stages=3,
                        )
                        return output

                    return run_exact_1x1_atomic

                partial_dtype = (
                    dtype if dtype == torch.float16 else torch.float32
                )
                partial = workspace_cache[
                    (
                        device.type,
                        device.index,
                        partial_dtype,
                        ("2d_1x1_nodiv_split_v7", 32, 128, 64),
                    )
                ]

                def run_exact_1x1_split(
                    inputs: Sequence[Any], _attrs: dict[str, Any]
                ) -> Any:
                    image, loss = inputs
                    _conv_wgrad2d_1x1_split_nodiv_kernel[(8, 32)](
                        image,
                        loss,
                        partial,
                        784,
                        128,
                        64,
                        128,
                        50176,
                        784,
                        100352,
                        784,
                        4,
                        BLOCK_CO=16,
                        BLOCK_CI=64,
                        BLOCK_M=256,
                        num_warps=8,
                        num_stages=3,
                    )
                    if dtype == torch.bfloat16:
                        _conv_wgrad2d_1x1_reduce_kernel[(64, 1)](
                            partial,
                            output,
                            128,
                            64,
                            128,
                            64,
                            1,
                            32,
                            BLOCK_CO=8,
                            BLOCK_CI=16,
                            num_warps=4,
                            num_stages=1,
                        )
                    else:
                        _conv_wgrad2d_1x1_reduce_kernel[(32, 1)](
                            partial,
                            output,
                            128,
                            64,
                            128,
                            64,
                            1,
                            32,
                            BLOCK_CO=8,
                            BLOCK_CI=32,
                            num_warps=4,
                            num_stages=1,
                        )
                    return output

                return run_exact_1x1_split

            exact_stride2 = (
                image_shape == (8, 64, 56, 56)
                and loss_shape == (8, 128, 28, 28)
                and filter_size == (128, 64, 3, 3)
                and stride_tuple == (2, 2)
                and pre == (1, 1)
                and post == (1, 1)
                and dilation_tuple == (1, 1)
                and mode == "CROSS_CORRELATION"
                and groups == 1
            )
            if exact_stride2:
                output = output_cache[
                    (device.type, device.index, dtype, filter_size)
                ]
                if dtype == torch.float32:

                    def run_exact_stride2_atomic(
                        inputs: Sequence[Any], _attrs: dict[str, Any]
                    ) -> Any:
                        image, loss = inputs
                        _conv_wgrad_zero_kernel[(72,)](
                            output,
                            73728,
                            BLOCK=1024,
                            num_warps=4,
                        )
                        _conv_wgrad2d_stride2_3tap_atomic_kernel[(16, 3, 8)](
                            image,
                            loss,
                            output,
                            6272,
                            56,
                            56,
                            28,
                            28,
                            128,
                            64,
                            200704,
                            3136,
                            56,
                            1,
                            100352,
                            784,
                            28,
                            1,
                            576,
                            9,
                            3,
                            1,
                            8,
                            BLOCK_CO=16,
                            BLOCK_CI=32,
                            BLOCK_M=16,
                            num_warps=2,
                            num_stages=3,
                        )
                        return output

                    return run_exact_stride2_atomic

                partial_dtype = (
                    dtype if dtype == torch.float16 else torch.float32
                )
                partial = workspace_cache[
                    (
                        device.type,
                        device.index,
                        partial_dtype,
                        ("2d_stride2_row4_v1", 8, 128, 64, 9),
                    )
                ]

                def run_exact_stride2(
                    inputs: Sequence[Any], _attrs: dict[str, Any]
                ) -> Any:
                    image, loss = inputs
                    _conv_wgrad2d_stride2_row4_split_kernel[(16, 3, 8)](
                        image,
                        loss,
                        partial,
                        128,
                        64,
                        128,
                        200704,
                        3136,
                        56,
                        1,
                        100352,
                        784,
                        28,
                        1,
                        BLOCK_CO=16,
                        BLOCK_CI=32,
                        BLOCK_HW=128,
                        num_warps=4,
                        num_stages=2,
                    )
                    _conv_wgrad2d_reduce_kernel[(16, 9, 1)](
                        partial,
                        output,
                        128,
                        64,
                        128,
                        576,
                        9,
                        3,
                        1,
                        3,
                        3,
                        8,
                        BLOCK_CO=16,
                        BLOCK_CI=32,
                        num_warps=4,
                        num_stages=1,
                    )
                    return output

                return run_exact_stride2

    def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, "conv_wgrad")
        image = inputs[0]
        key = (
            image.device.type,
            image.device.index,
            image.dtype,
            filter_size,
        )
        output = get_cached_empty_tensor(
            output_cache,
            key,
            filter_size,
            device=image.device,
            dtype=image.dtype,
        )
        return conv_wgrad(
            image,
            inputs[1],
            filter_size=filter_size,
            padding=padding,
            pre_padding=pre_padding,
            post_padding=post_padding,
            stride=stride,
            dilation=dilation,
            convolution_mode=convolution_mode,
            compute_data_type=compute_data_type,
            name=name,
            groups=groups,
            _output=output,
            _workspace=workspace_cache,
        )

    return run


@register_prepared_run_fn("conv_fprop")
def _prepare_conv_fprop(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    backend_run_fn = _prepare_backend_conv(
        "conv_fprop", attrs, input_specs, default_run_fn
    )
    if backend_run_fn is not None:
        return backend_run_fn

    if len(input_specs) < 2:
        return None

    rank = _conv_rank(input_specs[0], input_specs[1])
    if rank not in (1, 2, 3):
        return None
    if not _is_cross_correlation(attrs.get("convolution_mode")):
        return None

    from flag_dnn.ops.conv1d import conv1d
    from flag_dnn.ops.conv2d import conv2d
    from flag_dnn.ops.conv3d import conv3d

    stride = _tuple_n(attrs.get("stride", 1), rank, "stride")
    dilation = _tuple_n(attrs.get("dilation", 1), rank, "dilation")
    groups = int(attrs.get("groups", 1))
    padding = _direct_padding(
        rank,
        attrs.get("padding"),
        attrs.get("pre_padding"),
        attrs.get("post_padding"),
    )
    if rank == 1:
        stride_arg = stride[0]
        dilation_arg = dilation[0]

        def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "conv_fprop")
            return conv1d(
                inputs[0],
                inputs[1],
                stride=stride_arg,
                padding=padding,
                dilation=dilation_arg,
                groups=groups,
            )

        return run

    if rank == 2:
        stride_2d = cast(tuple[int, int], stride)
        dilation_2d = cast(tuple[int, int], dilation)
        image_spec = input_specs[0]
        weight_spec = input_specs[1]
        workspace_cache: PreparedTensorCache = {}
        use_workspace_cache = False
        if (
            image_spec.layout == "contiguous"
            and weight_spec.contiguous is True
            and image_spec.dtype == "bfloat16"
            and not isinstance(padding, str)
            and all(isinstance(dim, int) for dim in image_spec.shape)
            and all(isinstance(dim, int) for dim in weight_spec.shape)
        ):
            n, cin, h, w = (int(dim) for dim in image_spec.shape)
            c_out, _, kh, kw = (int(dim) for dim in weight_spec.shape)
            padding_tuple: tuple[int, ...]
            if isinstance(padding, int):
                padding_tuple = (int(padding),) * 4
            else:
                raw_padding = tuple(int(dim) for dim in padding)
                if len(raw_padding) == 2:
                    padding_tuple = (
                        raw_padding[0],
                        raw_padding[0],
                        raw_padding[1],
                        raw_padding[1],
                    )
                else:
                    padding_tuple = raw_padding
            use_workspace_cache = (
                n == 1
                and h == 40
                and w == 40
                and kh == 3
                and kw == 3
                and cin >= 256
                and c_out >= 512
                and groups == 1
                and stride_2d == (2, 2)
                and dilation_2d == (1, 1)
                and padding_tuple == (1, 1, 1, 1)
            )

        def run_conv2d(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "conv_fprop")
            return conv2d(
                inputs[0],
                inputs[1],
                stride=stride_2d,
                padding=padding,
                dilation=dilation_2d,
                groups=groups,
                _workspace=workspace_cache if use_workspace_cache else None,
            )

        return run_conv2d

    stride_3d = cast(tuple[int, int, int], stride)
    dilation_3d = cast(tuple[int, int, int], dilation)

    def run_conv3d(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, "conv_fprop")
        return conv3d(
            inputs[0],
            inputs[1],
            stride=stride_3d,
            padding=padding,
            dilation=dilation_3d,
            groups=groups,
        )

    return run_conv3d
