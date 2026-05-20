from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from flag_dnn import runtime
from flag_dnn.graph.device import has_runtime_device_tensor
from flag_dnn.graph.tensor import TensorSpec

RunFn = Callable[[Sequence[Any], dict[str, Any]], Any]


def prepare_run_fn(
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> RunFn:
    if op_type == "conv_fprop":
        prepared = _prepare_conv_fprop(attrs, input_specs, default_run_fn)
        if prepared is not None:
            return prepared
    return default_run_fn


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
    runtime_specs = all(
        _is_runtime_device_spec(spec) for spec in input_specs[:2]
    )

    if rank == 1:
        stride_arg = stride[0]
        dilation_arg = dilation[0]

        def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            if runtime_specs or has_runtime_device_tensor(inputs):
                return conv1d(
                    inputs[0],
                    inputs[1],
                    stride=stride_arg,
                    padding=padding,
                    dilation=dilation_arg,
                    groups=groups,
                )
            return default_run_fn(inputs, attrs)

        return run

    if rank == 2:

        def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            if runtime_specs or has_runtime_device_tensor(inputs):
                return conv2d(
                    inputs[0],
                    inputs[1],
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
            return default_run_fn(inputs, attrs)

        return run

    def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        if runtime_specs or has_runtime_device_tensor(inputs):
            return conv3d(
                inputs[0],
                inputs[1],
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        return default_run_fn(inputs, attrs)

    return run


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


def _is_runtime_device_spec(spec: TensorSpec) -> bool:
    if spec.device is None:
        return False
    device = str(spec.device)
    runtime_name = runtime.device.name
    return device == runtime_name or device.startswith(runtime_name + ":")
