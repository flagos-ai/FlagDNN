import pytest
import torch

import flag_dnn
from flag_dnn.graph import graph as graph_decorator
from flag_dnn.graph.backend import TritonAscendBackend
from flag_dnn.runtime.backend import vendor_module


pytestmark = pytest.mark.skipif(
    flag_dnn.vendor_name != "ascend",
    reason="Ascend backend tests require an Ascend runtime",
)


ASCEND_SIMPLE_CONFIGS = {
    "abs": {"BLOCK_SIZE": 256},
    "neg": {"BLOCK_SIZE": 256},
    "sqrt": {"BLOCK_SIZE": 256},
    "add_square": {"BLOCK_SIZE": 256, "TILES_PER_PROGRAM": 1},
    "unary": {"BLOCK_SIZE": 256, "TILES_PER_PROGRAM": 1},
    "pow": {"BLOCK_SIZE": 256},
    "tanh": {"BLOCK_SIZE": 256},
    "sigmoid": {"BLOCK_SIZE": 256},
    "sigmoid_backward": {"BLOCK_SIZE": 256},
    "relu": {"BLOCK_SIZE": 256},
    "leaky_relu": {"BLOCK_SIZE": 256},
    "elu": {"BLOCK_SIZE": 256},
    "gelu": {"BLOCK_SIZE": 256},
    "silu": {"BLOCK_SIZE": 256},
    "softplus": {"BLOCK_SIZE": 256},
    "layer_norm": {"BLOCK_SIZE": 256},
    "rms_norm": {"BLOCK_SIZE": 256},
    "reduction": {"BLOCK_M": 8, "BLOCK_N": 128},
}


def test_flagdnn_uses_package_qualified_ascend_backend() -> None:
    assert vendor_module.__name__ == "flag_dnn.runtime.backend._ascend"


def test_ascend_backend_aliases() -> None:
    for name in ("triton_ascend", "ascend", "npu"):
        assert isinstance(flag_dnn.resolve_backend(name), TritonAscendBackend)


def test_ascend_backend_rejects_non_npu_specs(monkeypatch) -> None:
    monkeypatch.setattr(torch.npu, "is_available", lambda: True)
    backend = TritonAscendBackend()
    npu_spec = flag_dnn.TensorSpec("x", (16,), "float32", device="npu:0")
    cpu_spec = flag_dnn.TensorSpec("x", (16,), "float32", device="cpu")

    assert backend.supports(flag_dnn.Graph(), [npu_spec])
    assert not backend.supports(flag_dnn.Graph(), [cpu_spec])


def test_graph_add_selects_ascend_candidate(monkeypatch) -> None:
    monkeypatch.setattr(torch.npu, "is_available", lambda: True)

    @graph_decorator
    def add_graph(lhs, rhs):
        return flag_dnn.add(
            lhs,
            rhs,
            compute_data_type="float32",
            name="add",
        )

    specs = [
        flag_dnn.TensorSpec("lhs", (1024,), "float32", device="npu:0"),
        flag_dnn.TensorSpec("rhs", (1024,), "float32", device="npu:0"),
    ]
    compiled = flag_dnn.compile(
        add_graph,
        inputs=specs,
        options={"backend": "npu", "cache": None},
    )

    candidates = compiled.plan.debug_info["kernel_candidates"]
    assert len(candidates) == 1
    assert candidates[0]["backend"] == "triton_ascend"
    assert candidates[0]["op_type"] == "add"


@pytest.mark.parametrize(
    ("op_name", "expected_meta"),
    ASCEND_SIMPLE_CONFIGS.items(),
)
def test_ascend_simple_tune_config(op_name, expected_meta) -> None:
    configs = flag_dnn.runtime.get_tuned_config(op_name)

    assert len(configs) == 1
    config = configs[0]
    assert config.kwargs == expected_meta
    assert config.num_warps == 4
    assert config.num_stages == 1
    assert config.num_ctas == 1
