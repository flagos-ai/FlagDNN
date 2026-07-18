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

import math

import pytest
from benchmark.base import (
    CudnnCompareBenchmark,
    cudnn_data_type,
    get_cudnn,
    skip_unsupported_cudnn_graph,
)
import torch

import flag_dnn
from benchmark import consts


def _to_spatial_tuple(value, rank):
    if isinstance(value, int):
        return (value,) * rank
    result = tuple(int(v) for v in value)
    if len(result) != rank:
        raise RuntimeError(f"expected length {rank}, got {value}")
    return result


def _to_cudnn_spatial(value, rank):
    return list(_to_spatial_tuple(value, rank))


def _normalize_conv_padding(
    rank, padding=0, pre_padding=None, post_padding=None
):
    if pre_padding is not None or post_padding is not None:
        if pre_padding is None or post_padding is None:
            raise RuntimeError(
                "both pre_padding and post_padding are required"
            )
        return (
            _to_spatial_tuple(pre_padding, rank),
            _to_spatial_tuple(post_padding, rank),
        )
    pad = _to_spatial_tuple(padding, rank)
    return pad, pad


def _conv_out_dim(input_size, pad_before, pad_after, dilation, kernel, stride):
    return (
        input_size + pad_before + pad_after - dilation * (kernel - 1) - 1
    ) // stride + 1


def _conv_loss_shape(
    input_shape,
    filter_size,
    stride,
    padding=0,
    dilation=1,
    pre_padding=None,
    post_padding=None,
):
    rank = len(input_shape) - 2
    stride = _to_spatial_tuple(stride, rank)
    dilation = _to_spatial_tuple(dilation, rank)
    pre, post = _normalize_conv_padding(
        rank, padding, pre_padding=pre_padding, post_padding=post_padding
    )
    out_dims = tuple(
        _conv_out_dim(
            input_shape[2 + axis],
            pre[axis],
            post[axis],
            dilation[axis],
            filter_size[2 + axis],
            stride[axis],
        )
        for axis in range(rank)
    )
    return input_shape[0], filter_size[0], *out_dims


def _randn_conv_tensor(shape, dtype, device):
    return torch.randn(shape, device=device, dtype=dtype).contiguous()


class ConvWgradBenchmark(CudnnCompareBenchmark):
    op_name = "conv_wgrad"
    shapes = consts.CONV_WGRAD_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_CONV_WGRAD_PERF_SHAPE_IDS"

    def make_inputs(self, case, dtype):
        self.case = case
        input_shape, filter_size, stride, padding, pre, post, dilation = case
        loss_shape = _conv_loss_shape(
            input_shape,
            filter_size,
            stride,
            0 if padding is None else padding,
            dilation,
            pre_padding=pre,
            post_padding=post,
        )
        image = _randn_conv_tensor(input_shape, dtype, flag_dnn.device)
        loss = _randn_conv_tensor(loss_shape, dtype, flag_dnn.device)
        return image, loss

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        (
            _,
            filter_size,
            stride,
            padding,
            pre_padding,
            post_padding,
            dilation,
        ) = self.case
        image, loss = inputs
        rank = image.dim() - 2
        io_dtype = cudnn_data_type(image.dtype)
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.cudnn_handle,
        )

        image_tensor = graph.tensor_like(image)
        loss_tensor = graph.tensor_like(loss)
        conv_kwargs = {
            "image": image_tensor,
            "loss": loss_tensor,
            "stride": _to_cudnn_spatial(stride, rank),
            "dilation": _to_cudnn_spatial(dilation, rank),
            "name": "conv_wgrad",
        }
        if pre_padding is None:
            conv_kwargs["padding"] = _to_cudnn_spatial(
                0 if padding is None else padding, rank
            )
        else:
            conv_kwargs["pre_padding"] = _to_cudnn_spatial(pre_padding, rank)
            conv_kwargs["post_padding"] = _to_cudnn_spatial(post_padding, rank)

        dw_tensor = graph.conv_wgrad(**conv_kwargs)
        dw = torch.empty(filter_size, device=image.device, dtype=image.dtype)
        dw_tensor.set_output(True).set_data_type(io_dtype).set_dim(
            list(dw.shape)
        ).set_stride(list(dw.stride()))

        try:
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
            skip_unsupported_cudnn_graph(exc, self.op_name)

        workspace = torch.empty(
            graph.get_workspace_size(), device=image.device, dtype=torch.uint8
        )

        def run():
            graph.execute(
                {image_tensor: image, loss_tensor: loss, dw_tensor: dw},
                workspace,
                handle=self.cudnn_handle,
            )
            return dw

        return run

    def build_flag_dnn_runner(self, inputs):
        (
            _,
            filter_size,
            stride,
            padding,
            pre_padding,
            post_padding,
            dilation,
        ) = self.case
        image, loss = inputs

        @flag_dnn.graph
        def flag_dnn_conv_wgrad_graph(image, loss):
            kwargs = {
                "stride": stride,
                "dilation": dilation,
                "convolution_mode": "CROSS_CORRELATION",
                "compute_data_type": "FLOAT",
                "name": "conv_wgrad",
            }
            if pre_padding is None:
                kwargs["padding"] = padding
            else:
                kwargs["pre_padding"] = pre_padding
                kwargs["post_padding"] = post_padding
            return flag_dnn.conv_wgrad(image, loss, filter_size, **kwargs)

        compiled = flag_dnn.compile(
            flag_dnn_conv_wgrad_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(image, "image"),
                flag_dnn.TensorSpec.from_tensor(loss, "loss"),
            ],
            options={"cache": None, "validate_inputs": False},
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            "conv_wgrad"
        ]

        def run():
            return compiled.run(image, loss)

        return run

    def transfer_bytes(self, inputs):
        image, loss = inputs
        filter_size = self.case[1]
        return (
            image.numel() * image.element_size()
            + loss.numel() * loss.element_size()
            + math.prod(filter_size) * image.element_size()
        )

    def shape_detail(self, inputs):
        _, filter_size, stride, padding, pre, post, dilation = self.case
        return {
            "image": tuple(inputs[0].shape),
            "loss": tuple(inputs[1].shape),
            "filter_size": tuple(filter_size),
            "stride": stride,
            "padding": padding,
            "pre_padding": pre,
            "post_padding": post,
            "dilation": dilation,
        }


@pytest.mark.conv_wgrad
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", ConvWgradBenchmark.dtypes)
def test_conv_wgrad(cudnn_handle, dtype):
    torch.manual_seed(0)
    ConvWgradBenchmark(cudnn_handle).run(dtype)
