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

import pytest
import torch

import flag_dnn
from benchmark import consts
from benchmark.base import CudnnCompareBenchmark, get_cudnn

EPS = 1e-5


class RmsNormRhtAmaxBenchmark(CudnnCompareBenchmark):
    op_name = "rmsnorm_rht_amax_wrapper_sm100"
    dtypes = (torch.bfloat16,)
    shapes = consts.RMSNORM_RHT_AMAX_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_RMSNORM_RHT_AMAX_PERF_SHAPE_IDS"

    def make_inputs(self, case, dtype):
        m, n, rows_per_cta = case
        self.rows_per_cta = rows_per_cta
        x = torch.randn(
            (m, n), device=flag_dnn.device, dtype=dtype
        ).contiguous()
        w = torch.randn((n,), device=flag_dnn.device, dtype=dtype).contiguous()
        return x, w

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        try:
            wrapper = getattr(cudnn, "rmsnorm_rht_amax_wrapper_sm100")
        except (AttributeError, ImportError, RuntimeError) as exc:
            pytest.skip(
                f"cuDNN RMSNorm RHT amax wrapper is unavailable: {exc}"
            )
        x, w = inputs

        def run():
            return wrapper(x, w, eps=EPS, rows_per_cta=self.rows_per_cta)

        try:
            run()
            torch.cuda.synchronize()
        except (AssertionError, ImportError, RuntimeError, ValueError) as exc:
            pytest.skip(
                "cuDNN RMSNorm RHT amax wrapper is unsupported here: " f"{exc}"
            )
        return run

    def build_flag_dnn_runner(self, inputs):
        x, w = inputs

        @flag_dnn.graph
        def flag_dnn_rmsnorm_rht_amax_graph(x, w):
            return flag_dnn.rmsnorm_rht_amax_wrapper_sm100(
                x, w, eps=EPS, rows_per_cta=self.rows_per_cta
            )

        compiled = flag_dnn.compile(
            flag_dnn_rmsnorm_rht_amax_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(x, "x"),
                flag_dnn.TensorSpec.from_tensor(w, "w"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            "rmsnorm_rht_amax_wrapper_sm100"
        ]

        def run():
            return compiled.run(x, w)

        return run

    def transfer_bytes(self, inputs):
        x, w = inputs
        num_ctas = x.shape[0] // self.rows_per_cta
        return (
            x.numel() * x.element_size()
            + w.numel() * w.element_size()
            + x.numel() * x.element_size()
            + num_ctas * torch.empty((), dtype=torch.float32).element_size()
        )

    def shape_detail(self, inputs):
        x, w = inputs
        return [x.size(), w.size(), self.rows_per_cta]


@pytest.mark.rmsnorm_rht_amax
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", RmsNormRhtAmaxBenchmark.dtypes)
def test_rmsnorm_rht_amax(cudnn_handle, dtype):
    torch.manual_seed(0)
    RmsNormRhtAmaxBenchmark(cudnn_handle).run(dtype)
