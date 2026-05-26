import pytest
from benchmark_graph.base import CudnnCompareBenchmark
import torch

import flag_dnn
from benchmark_graph import consts
from tests.cudnn_legacy import CudnnBatchNormInference, CudnnLegacyError


class BatchNormInferenceBenchmark(CudnnCompareBenchmark):
    op_name = "batchnorm_inference"
    shapes = consts.BATCHNORM_INFERENCE_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_BATCHNORM_INFERENCE_PERF_SHAPE_IDS"

    def make_inputs(self, shape, dtype):
        x = torch.randn(
            shape, device=flag_dnn.device, dtype=dtype
        ).contiguous()
        channels = int(shape[1])
        mean = torch.randn(
            (1, channels, 1, 1), device=flag_dnn.device, dtype=torch.float32
        ).contiguous()
        inv_var = (
            torch.rand(
                (1, channels, 1, 1),
                device=flag_dnn.device,
                dtype=torch.float32,
            )
            + 0.5
        ).contiguous()
        scale = torch.randn_like(mean).contiguous()
        bias = torch.randn_like(mean).contiguous()
        return x, mean, inv_var, scale, bias

    def build_cudnn_runner(self, inputs):
        x, mean, inv_var, scale, bias = inputs
        try:
            runner = CudnnBatchNormInference(x, mean, inv_var, scale, bias)
        except CudnnLegacyError as exc:
            pytest.skip(f"legacy cuDNN batchnorm_inference unavailable: {exc}")

        def run():
            return runner()

        return run

    def build_flag_dnn_runner(self, inputs):
        x, mean, inv_var, scale, bias = inputs

        @flag_dnn.graph
        def flag_dnn_batchnorm_inference_graph(x, mean, inv_var, scale, bias):
            return flag_dnn.batchnorm_inference(
                x,
                mean,
                inv_var,
                scale,
                bias,
                compute_data_type="float32",
                name=self.op_name,
            )

        compiled = flag_dnn.compile(
            flag_dnn_batchnorm_inference_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(x, "x"),
                flag_dnn.TensorSpec.from_tensor(mean, "mean"),
                flag_dnn.TensorSpec.from_tensor(inv_var, "inv_var"),
                flag_dnn.TensorSpec.from_tensor(scale, "scale"),
                flag_dnn.TensorSpec.from_tensor(bias, "bias"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            "batchnorm_inference"
        ]

        def run():
            return compiled.run(x, mean, inv_var, scale, bias)

        return run

    def transfer_bytes(self, inputs):
        x, mean, inv_var, scale, bias = inputs
        return x.numel() * x.element_size() * 2 + sum(
            t.numel() * t.element_size() for t in (mean, inv_var, scale, bias)
        )


@pytest.mark.cudnn_legacy
@pytest.mark.batchnorm_inference
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", BatchNormInferenceBenchmark.dtypes)
def test_perf_graph_batchnorm_inference_vs_cudnn(cudnn_handle, dtype):
    torch.manual_seed(0)
    BatchNormInferenceBenchmark(cudnn_handle).run(dtype)
