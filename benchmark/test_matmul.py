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

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_MATMUL_MODES = (
    ("fp16", torch.float16, "float32", torch.float16),
    ("bf16", torch.bfloat16, "float32", torch.bfloat16),
    ("tf32", torch.float32, "tf32", torch.float32),
    (
        "fp8_e4m3_to_fp32",
        torch.float8_e4m3fn,
        "fast_float_for_fp8",
        torch.float32,
    ),
    (
        "fp8_e5m2_to_fp32",
        torch.float8_e5m2,
        "fast_float_for_fp8",
        torch.float32,
    ),
)


class MatmulBenchmark(CudnnCompareBenchmark):
    op_name = "matmul"
    shapes = consts.MATMUL_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_MATMUL_PERF_SHAPE_IDS"

    def __init__(
        self,
        cudnn_handle,
        *,
        mode_name,
        compute_mode,
        out_dtype,
    ):
        super().__init__(cudnn_handle)
        self.op_name = f"matmul_{mode_name}"
        self.compute_mode = compute_mode
        self.out_dtype = out_dtype

    def make_inputs(self, shape_pair, dtype):
        a_shape, b_shape = shape_pair
        input_dtype = torch.float32 if dtype in _FP8_DTYPES else dtype
        scale = 0.25 if dtype in _FP8_DTYPES else 1.0
        a = (
            torch.randn(a_shape, dtype=input_dtype, device=flag_dnn.device)
            .mul_(scale)
            .to(dtype)
        )
        b = (
            torch.randn(b_shape, dtype=input_dtype, device=flag_dnn.device)
            .mul_(scale)
            .to(dtype)
        )
        return a, b

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        a, b = inputs
        io_dtype = cudnn_data_type(a.dtype)
        compute_type = (
            cudnn.data_type.FAST_FLOAT_FOR_FP8
            if self.compute_mode == "fast_float_for_fp8"
            else cudnn.data_type.FLOAT
        )
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=compute_type,
            handle=self.cudnn_handle,
        )
        if a.dtype in _FP8_DTYPES:
            a_tensor = graph.tensor(
                dim=tuple(a.shape),
                stride=tuple(a.stride()),
                data_type=io_dtype,
            )
            b_tensor = graph.tensor(
                dim=tuple(b.shape),
                stride=tuple(b.stride()),
                data_type=io_dtype,
            )
        else:
            a_tensor = graph.tensor_like(a)
            b_tensor = graph.tensor_like(b)
        matmul_tensor = graph.matmul(
            A=a_tensor,
            B=b_tensor,
            compute_data_type=compute_type,
            name=self.op_name,
        )
        if a.dtype in _FP8_DTYPES:
            matmul_tensor.set_data_type(cudnn.data_type.FLOAT)
            y_tensor = graph.identity(
                input=matmul_tensor,
                compute_data_type=cudnn.data_type.FLOAT,
                name=f"{self.op_name}_output_cast",
            )
        else:
            y_tensor = matmul_tensor
        output_dtype = cudnn_data_type(self.out_dtype)
        y_tensor.set_output(True).set_data_type(output_dtype)

        try:
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
            skip_unsupported_cudnn_graph(exc, self.op_name)

        y = torch.empty(
            (*a.shape[:-1], b.shape[-1]),
            device=a.device,
            dtype=self.out_dtype,
        )
        workspace = torch.empty(
            graph.get_workspace_size(), device=a.device, dtype=torch.uint8
        )

        def run():
            graph.execute(
                {a_tensor: a, b_tensor: b, y_tensor: y},
                workspace,
                handle=self.cudnn_handle,
            )
            return y

        return run

    def build_flag_dnn_runner(self, inputs):
        a, b = inputs

        @flag_dnn.graph
        def flag_dnn_matmul_graph(a, b):
            return flag_dnn.matmul(
                a,
                b,
                compute_data_type=self.compute_mode,
                out_dtype=self.out_dtype,
                name=self.op_name,
            )

        compiled = flag_dnn.compile(
            flag_dnn_matmul_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(a, "a"),
                flag_dnn.TensorSpec.from_tensor(b, "b"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["matmul"]

        def run():
            return compiled.run(a, b)

        return run

    def transfer_bytes(self, inputs):
        a, b = inputs
        output_elements = math.prod((*a.shape[:-1], b.shape[-1]))
        return (
            a.numel() * a.element_size()
            + b.numel() * b.element_size()
            + output_elements
            * torch.empty((), dtype=self.out_dtype).element_size()
        )


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("mode_name", "dtype", "compute_mode", "out_dtype"),
    _MATMUL_MODES,
    ids=[mode[0] for mode in _MATMUL_MODES],
)
def test_matmul(
    cudnn_handle,
    mode_name,
    dtype,
    compute_mode,
    out_dtype,
):
    torch.manual_seed(0)
    benchmark = MatmulBenchmark(
        cudnn_handle,
        mode_name=mode_name,
        compute_mode=compute_mode,
        out_dtype=out_dtype,
    )
    benchmark.run(dtype)
