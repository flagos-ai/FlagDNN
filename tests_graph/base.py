from __future__ import annotations

import sys
from pathlib import Path

import pytest

cudnn = pytest.importorskip("cudnn", exc_type=ImportError)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

CUDNN_COMPARE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def get_cudnn():
    return cudnn


def cudnn_data_type(dtype):
    if dtype == torch.float16:
        return cudnn.data_type.HALF
    if dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    if dtype == torch.float32:
        return cudnn.data_type.FLOAT
    if dtype == torch.float64:
        return cudnn.data_type.DOUBLE
    if dtype == torch.bool:
        return cudnn.data_type.BOOLEAN
    raise TypeError(f"Unsupported dtype for cuDNN frontend: {dtype}")


def cudnn_graph(dtype, cudnn_handle):
    return cudnn.pygraph(
        io_data_type=cudnn_data_type(dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )


def skip_unsupported_cudnn_graph(exc, op_name):
    message = str(exc)
    if (
        isinstance(exc, cudnn.cudnnGraphNotSupportedError)
        or "CUDNN_STATUS_BAD_PARAM" in message
        or "No valid engine configs" in message
    ):
        pytest.skip(f"cuDNN frontend does not support {op_name}: {exc}")
    raise exc


def execute_cudnn_graph(
    graph,
    exec_tensors,
    output_value,
    output_template,
    cudnn_handle,
    op_name,
):
    output_value.set_output(True).set_data_type(
        cudnn_data_type(output_template.dtype)
    )
    try:
        graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
        skip_unsupported_cudnn_graph(exc, op_name)

    output = torch.empty_like(output_template)
    workspace = torch.empty(
        graph.get_workspace_size(),
        device=output_template.device,
        dtype=torch.uint8,
    )
    exec_tensors[output_value] = output
    graph.execute(exec_tensors, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    return output


def to_pair(value):
    if isinstance(value, int):
        return [value, value]
    return list(value)


def conv_out_dim(input_size, pad_before, pad_after, dilation, kernel, stride):
    return (
        input_size + pad_before + pad_after - dilation * (kernel - 1) - 1
    ) // stride + 1


def conv2d_output_shape(input_shape, weight_shape, stride, padding, dilation):
    stride = to_pair(stride)
    padding = to_pair(padding)
    dilation = to_pair(dilation)
    n, _, h, w = input_shape
    c_out, _, kh, kw = weight_shape
    oh = conv_out_dim(h, padding[0], padding[0], dilation[0], kh, stride[0])
    ow = conv_out_dim(w, padding[1], padding[1], dilation[1], kw, stride[1])
    return n, c_out, oh, ow


def channels_last_randn(shape, dtype, device):
    return torch.randn(shape, device=device, dtype=dtype).contiguous(
        memory_format=torch.channels_last
    )


def conv2d_output_template(
    input_shape,
    weight_shape,
    stride,
    padding,
    dilation,
    dtype,
    device,
):
    return torch.empty(
        conv2d_output_shape(
            input_shape, weight_shape, stride, padding, dilation
        ),
        device=device,
        dtype=dtype,
    ).contiguous(memory_format=torch.channels_last)
