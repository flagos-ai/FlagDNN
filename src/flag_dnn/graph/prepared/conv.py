from __future__ import annotations

from typing import Any, Optional, Sequence, cast

import torch

from flag_dnn.graph.prepared import (
    PreparedTensorCache,
    RunFn,
    get_cached_empty_tensor,
    register_prepared_run_fn,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _require_runtime_backend,
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


@register_prepared_run_fn("conv_dgrad")
def _prepare_conv_dgrad(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
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
                        _conv_wgrad2d_1x1_reduce_kernel[(8, 1)](
                            partial,
                            output,
                            128,
                            64,
                            128,
                            64,
                            1,
                            32,
                            BLOCK_CO=16,
                            BLOCK_CI=64,
                            num_warps=8,
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

        def run_conv2d(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "conv_fprop")
            return conv2d(
                inputs[0],
                inputs[1],
                stride=stride_2d,
                padding=padding,
                dilation=dilation_2d,
                groups=groups,
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
