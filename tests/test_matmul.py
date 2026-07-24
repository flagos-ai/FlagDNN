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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from tests.base import (
    CUDNN_COMPARE_DTYPES,
    cudnn,
    cudnn_data_type,
    execute_cudnn_graph,
)
import torch

import flag_dnn
from tests import accuracy_utils as utils
from tests import consts

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_MATMUL_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    *_FP8_DTYPES,
)


def _matmul_output_shape(a, b):
    batch_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return (*batch_shape, a.shape[-2], b.shape[-1])


def _cudnn_tensor(graph, value):
    if value.dtype in _FP8_DTYPES:
        return graph.tensor(
            dim=tuple(value.shape),
            stride=tuple(value.stride()),
            data_type=cudnn_data_type(value.dtype),
        )
    return graph.tensor_like(value)


def _cudnn_matmul(
    a,
    b,
    cudnn_handle,
    *,
    out_dtype=None,
    compute_mode="float32",
):
    out_dtype = a.dtype if out_dtype is None else out_dtype
    compute_type = (
        cudnn.data_type.FAST_FLOAT_FOR_FP8
        if compute_mode == "fast_float_for_fp8"
        else cudnn.data_type.FLOAT
    )
    graph = cudnn.pygraph(
        io_data_type=cudnn_data_type(a.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=compute_type,
        handle=cudnn_handle,
    )
    a_tensor = _cudnn_tensor(graph, a)
    b_tensor = _cudnn_tensor(graph, b)
    matmul_tensor = graph.matmul(
        A=a_tensor,
        B=b_tensor,
        compute_data_type=compute_type,
        name="matmul",
    )
    if a.dtype in _FP8_DTYPES:
        matmul_tensor.set_data_type(cudnn.data_type.FLOAT)
    y_tensor = graph.identity(
        input=matmul_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="output_cast",
    )
    y_tensor.set_output(True).set_data_type(cudnn_data_type(out_dtype))
    return execute_cudnn_graph(
        graph,
        {a_tensor: a, b_tensor: b},
        y_tensor,
        torch.empty(
            _matmul_output_shape(a, b),
            device=a.device,
            dtype=out_dtype,
        ),
        cudnn_handle,
        "matmul",
    )


def _cudnn_matmul_materialized_batch(
    a,
    b,
    cudnn_handle,
    *,
    out_dtype,
    compute_mode,
):
    batch_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a_matrix_shape = tuple(a.shape[-2:])
    b_matrix_shape = tuple(b.shape[-2:])
    a_batched = (
        a.expand((*batch_shape, *a_matrix_shape))
        .reshape(-1, *a_matrix_shape)
        .contiguous()
    )
    b_batched = (
        b.expand((*batch_shape, *b_matrix_shape))
        .reshape(-1, *b_matrix_shape)
        .contiguous()
    )
    output = _cudnn_matmul(
        a_batched,
        b_batched,
        cudnn_handle,
        out_dtype=out_dtype,
        compute_mode=compute_mode,
    )
    return output.reshape(_matmul_output_shape(a, b))


def _compile_flag_dnn_matmul_graph(
    a,
    b,
    *,
    out_dtype=None,
    compute_mode="float32",
):
    @flag_dnn.graph
    def flag_dnn_matmul_graph(a, b):
        return flag_dnn.matmul(
            a,
            b,
            compute_data_type=compute_mode,
            out_dtype=out_dtype,
            name="matmul",
        )

    compiled = flag_dnn.compile(
        flag_dnn_matmul_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(a, "a"),
            flag_dnn.TensorSpec.from_tensor(b, "b"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["matmul"]
    return compiled


def _run_flag_dnn_matmul_graph(
    a,
    b,
    *,
    out_dtype=None,
    compute_mode="float32",
):
    compiled = _compile_flag_dnn_matmul_graph(
        a, b, out_dtype=out_dtype, compute_mode=compute_mode
    )
    return compiled.run(a.clone(), b.clone())


def test_matmul_sm90_selector_requires_hopper_and_validated_key(monkeypatch):
    import flag_dnn.runtime.backend._nvidia.ops.matmul_sm90 as sm90

    key = sm90.Sm90MatmulKey(
        16,
        2048,
        2048,
        512,
        torch.float16,
        torch.float16,
        "float32",
    )
    config = sm90.Sm90MatmulConfig("lowp", 128, 256, 64, 3, 8, 1, 168, False)
    monkeypatch.setitem(sm90._VALIDATED_CONFIGS, key, config)
    assert sm90.select_sm90_matmul_config((8, 0), key) is None
    assert sm90.select_sm90_matmul_config((9, 0), key) == config
    assert sm90.select_sm90_matmul_config((10, 0), key) is None
    assert (
        sm90.select_sm90_matmul_config(
            (9, 0),
            sm90.Sm90MatmulKey(
                16,
                2048,
                2048,
                512,
                torch.float16,
                torch.float16,
                "ieee",
            ),
        )
        is None
    )


def test_matmul_sm90_id7_uses_exact_serial_kernel():
    from flag_dnn.runtime.backend._nvidia.ops import (
        matmul_sm90,
        matmul_sm90_gluon,
    )

    config = matmul_sm90.Sm90MatmulConfig(
        "lowp", 128, 256, 64, 3, 8, 2, 168, False
    )

    for dtype in (torch.float16, torch.bfloat16):
        key = matmul_sm90.Sm90MatmulKey(
            32, 1024, 1024, 4096, dtype, dtype, "float32"
        )
        assert matmul_sm90.select_sm90_matmul_config((9, 0), key) == config

    assert matmul_sm90_gluon._uses_id7_serial_kernel(
        32, 1024, 1024, 4096, config
    )
    assert not matmul_sm90_gluon._uses_id7_serial_kernel(
        16, 1024, 1024, 4096, config
    )
    assert not matmul_sm90_gluon._uses_id7_serial_kernel(
        32,
        1024,
        1024,
        4096,
        matmul_sm90.Sm90MatmulConfig("lowp", 128, 256, 64, 3, 8, 2, 168, True),
    )


@pytest.mark.parametrize(
    ("batch", "m", "n", "k"),
    (
        (32, 512, 512, 512),
        (16, 2048, 2048, 512),
    ),
    ids=("id2", "id6"),
)
def test_matmul_sm90_bf16_borderline_shapes_use_cublaslt(batch, m, n, k):
    from flag_dnn.runtime.backend._nvidia.ops import matmul_sm90

    key = matmul_sm90.Sm90MatmulKey(
        batch,
        m,
        n,
        k,
        torch.bfloat16,
        torch.bfloat16,
        "float32",
    )

    config = matmul_sm90.select_sm90_matmul_config((9, 0), key)

    assert config is not None
    assert config.family == "lowp_cublaslt"


@pytest.mark.parametrize("warp_specialized", (False, True))
def test_prepared_sm90_runner_uses_input_device_context(
    monkeypatch, warp_specialized
):
    from flag_dnn.runtime.backend._nvidia.ops import matmul_sm90_gluon
    from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90 import (
        Sm90MatmulConfig,
    )

    input_device = torch.device("cpu")
    other_device = torch.device("meta")
    current_device = [other_device]
    context_devices = []
    launch_devices = []

    class DeviceContext:
        def __init__(self, device):
            self.device = device
            self.previous = None

        def __enter__(self):
            self.previous = current_device[0]
            current_device[0] = self.device
            context_devices.append(self.device)

        def __exit__(self, *args):
            current_device[0] = self.previous

    class DeviceFn:
        @staticmethod
        def device(device):
            return DeviceContext(device)

    class FakeDescriptor:
        @staticmethod
        def from_tensor(*args):
            return object()

    class FakeCompiledKernel:
        def __getitem__(self, grid):
            def cached_launch(*args):
                launch_devices.append(current_device[0])

            return cached_launch

    class FakeKernel:
        def __getitem__(self, grid):
            def first_launch(*args, **kwargs):
                launch_devices.append(current_device[0])
                return FakeCompiledKernel()

            return first_launch

    layout = SimpleNamespace(get_default_for=lambda shape, dtype: object())
    fake_gl = SimpleNamespace(float16=object(), NVMMASharedLayout=layout)
    monkeypatch.setattr(matmul_sm90_gluon, "torch_device_fn", DeviceFn())
    monkeypatch.setattr(matmul_sm90_gluon, "TensorDescriptor", FakeDescriptor)
    monkeypatch.setattr(matmul_sm90_gluon, "gl", fake_gl)
    monkeypatch.setattr(
        matmul_sm90_gluon,
        "_validate_inputs",
        lambda a, b, c, config: (1, 1, 1, 1),
    )
    monkeypatch.setattr(
        matmul_sm90_gluon,
        "get_sm_count_for",
        lambda device: 1,
    )
    kernel_name = (
        "_matmul_sm90_ws_kernel" if warp_specialized else "_matmul_sm90_kernel"
    )
    monkeypatch.setattr(matmul_sm90_gluon, kernel_name, FakeKernel())

    a = torch.empty((1, 1, 1), dtype=torch.float16)
    b = torch.empty((1, 1, 1), dtype=torch.float16)
    c = torch.empty((1, 1, 1), dtype=torch.float16)
    config = Sm90MatmulConfig("lowp", 1, 1, 1, 2, 4, 1, 168, warp_specialized)
    runner = matmul_sm90_gluon.prepare_sm90_matmul(a, b, c, config=config)

    assert runner() is c
    assert current_device[0] == other_device
    assert runner() is c

    assert current_device[0] == other_device
    assert context_devices == [input_device, input_device]
    assert launch_devices == [input_device, input_device]


def test_matmul_selected_sm90_failure_is_not_silently_fallback(monkeypatch):
    import importlib

    matmul_module = importlib.import_module("flag_dnn.ops.matmul")

    def fail(*args, **kwargs):
        raise RuntimeError("selected SM90 kernel failed")

    def get_backend_hook(name):
        return fail if name == "matmul_3d_out" else None

    monkeypatch.setattr(
        matmul_module.runtime, "get_backend_hook", get_backend_hook
    )
    a = torch.randn((1, 32, 32), device=flag_dnn.device, dtype=torch.float16)
    b = torch.randn((1, 32, 32), device=flag_dnn.device, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="selected SM90 kernel failed"):
        _run_flag_dnn_matmul_graph(a, b)


def test_matmul_id7_uses_fixed_stage3_config(monkeypatch):
    import importlib

    matmul_module = importlib.import_module("flag_dnn.ops.matmul")
    calls = []

    class FakeKernel:
        def __init__(self):
            self.fn = self

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(matmul_module, "_batched_matmul_kernel", FakeKernel())
    monkeypatch.setattr(
        matmul_module.runtime, "get_backend_hook", lambda name: None
    )
    monkeypatch.setattr(
        matmul_module.torch_device_fn,
        "device",
        lambda device: nullcontext(),
    )
    a = torch.empty((32, 1024, 4096), device="meta", dtype=torch.float16)
    b = torch.empty((32, 4096, 1024), device="meta", dtype=torch.float16)
    c = torch.empty((32, 1024, 1024), device="meta", dtype=torch.float16)

    actual = matmul_module._batched_matmul_3d_out(
        a, b, c, compute_mode="float32"
    )

    assert actual is c
    assert len(calls) == 1
    grid, args, kwargs = calls[0]
    assert grid == (32, 32)
    assert args[0:3] == (a, b, c)
    assert kwargs == {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "GROUP_M": 4,
        "ROUND_F32_TO_TF32": False,
        "num_warps": 8,
        "num_stages": 3,
    }


@pytest.mark.matmul
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9,
    reason="Hopper is required",
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_matmul_sm90_gluon_lowp_direct(cudnn_handle, dtype):
    from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90 import (
        Sm90MatmulConfig,
    )
    from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90_gluon import (
        run_sm90_matmul,
    )

    torch.manual_seed(0)
    a = torch.randn((2, 128, 64), device=flag_dnn.device, dtype=dtype)
    b = torch.randn((2, 64, 256), device=flag_dnn.device, dtype=dtype)
    c = torch.empty((2, 128, 256), device=flag_dnn.device, dtype=dtype)
    config = Sm90MatmulConfig("lowp", 128, 256, 64, 3, 8, 1, 168, False)

    actual = run_sm90_matmul(a, b, c, config=config)
    expected = _cudnn_matmul(a, b, cudnn_handle)

    assert actual is c
    atol = 1e-1 if dtype == torch.bfloat16 else 5e-2
    utils.gems_assert_close(actual, expected, dtype, atol=atol)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 is required",
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_matmul_sm90_persistent_lowp_matches_cudnn(cudnn_handle, dtype):
    torch.manual_seed(37)
    a = torch.randn((32, 512, 512), device=flag_dnn.device, dtype=dtype)
    b = torch.randn((32, 512, 512), device=flag_dnn.device, dtype=dtype)

    expected = _cudnn_matmul(a, b, cudnn_handle)
    actual = _run_flag_dnn_matmul_graph(a, b)

    atol = 1e-1 if dtype == torch.bfloat16 else 5e-2
    utils.gems_assert_close(actual, expected, dtype, atol=atol)


@pytest.mark.matmul
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9,
    reason="Hopper is required",
)
@pytest.mark.parametrize("dtype", _FP8_DTYPES)
def test_matmul_sm90_gluon_fp8_direct(cudnn_handle, dtype):
    from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90 import (
        Sm90MatmulConfig,
    )
    from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90_gluon import (
        run_sm90_matmul,
    )

    torch.manual_seed(0)
    a = torch.randn((2, 128, 64), device=flag_dnn.device).mul_(0.25).to(dtype)
    b = torch.randn((2, 64, 256), device=flag_dnn.device).mul_(0.25).to(dtype)
    c = torch.empty((2, 128, 256), device=flag_dnn.device)
    config = Sm90MatmulConfig("fp8", 128, 64, 64, 3, 8, 1, 168, False)

    actual = run_sm90_matmul(a, b, c, config=config)
    expected = _cudnn_matmul(
        a,
        b,
        cudnn_handle,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )

    assert actual is c
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=0)


@pytest.mark.matmul
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9,
    reason="Hopper is required",
)
@pytest.mark.parametrize("dtype", _FP8_DTYPES)
def test_matmul_sm90_gluon_fp8_warp_specialized_direct(cudnn_handle, dtype):
    from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90 import (
        Sm90MatmulConfig,
    )
    from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90_gluon import (
        run_sm90_matmul,
    )

    torch.manual_seed(0)
    a = torch.randn((2, 128, 64), device=flag_dnn.device).mul_(0.25).to(dtype)
    b = torch.randn((2, 64, 256), device=flag_dnn.device).mul_(0.25).to(dtype)
    c = torch.empty((2, 128, 256), device=flag_dnn.device)
    config = Sm90MatmulConfig("fp8", 128, 128, 64, 3, 8, 1, 160, True)

    actual = run_sm90_matmul(a, b, c, config=config)
    expected = _cudnn_matmul(
        a,
        b,
        cudnn_handle,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )

    assert actual is c
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=0)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 is required",
)
@pytest.mark.parametrize("dtype", _FP8_DTYPES)
def test_matmul_sm90_fp8_tma_pretranspose_matches_cudnn(
    cudnn_handle,
    dtype,
):
    torch.manual_seed(29)
    a = torch.randint(-1, 2, (16, 1024, 1024), device=flag_dnn.device).to(
        dtype
    )
    b = torch.randint(-1, 2, (16, 1024, 1024), device=flag_dnn.device).to(
        dtype
    )

    expected = _cudnn_matmul(
        a,
        b,
        cudnn_handle,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )
    actual = _run_flag_dnn_matmul_graph(
        a,
        b,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )

    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=0)


def _torch_fp32_matmul(a, b):
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        try:
            return torch.matmul(a, b)
        except RuntimeError as exc:
            if "CUBLAS_STATUS_INVALID_VALUE" not in str(exc):
                raise

            # Some CUDA/driver combinations reject valid strided-batched
            # SGEMM while the equivalent individual SGEMMs remain supported.
            batch_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
            a_shape = tuple(a.shape[-2:])
            b_shape = tuple(b.shape[-2:])
            a_batch = a.expand((*batch_shape, *a_shape)).reshape(-1, *a_shape)
            b_batch = b.expand((*batch_shape, *b_shape)).reshape(-1, *b_shape)
            results = [
                torch.mm(a_item, b_item)
                for a_item, b_item in zip(a_batch, b_batch)
            ]
            return torch.stack(results).reshape(_matmul_output_shape(a, b))
    finally:
        torch.set_float32_matmul_precision(previous_precision)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape_pair", consts.MATMUL_CASES)
def test_matmul(cudnn_handle, dtype, shape_pair):
    torch.manual_seed(0)
    a_shape, b_shape = shape_pair
    a = torch.randn(a_shape, dtype=dtype, device=flag_dnn.device)
    b = torch.randn(b_shape, dtype=dtype, device=flag_dnn.device)

    flag_dnn_out = _run_flag_dnn_matmul_graph(a, b)

    if dtype == torch.bfloat16:
        expected = _cudnn_matmul(a, b, cudnn_handle)
        atol = 1e-1
    elif dtype == torch.float16:
        expected = _cudnn_matmul(a, b, cudnn_handle)
        atol = 5e-2
    else:
        expected = _torch_fp32_matmul(a, b)
        atol = 2e-4
    utils.gems_assert_close(flag_dnn_out, expected, dtype, atol=atol)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "shape_pair",
    (
        ((1, 512, 512), (1, 512, 512)),
        ((1, 1024, 512), (1, 512, 1024)),
    ),
    ids=("general_512", "direct_1024"),
)
def test_matmul_fp32_ieee_dispatch(shape_pair):
    torch.manual_seed(7)
    a_shape, b_shape = shape_pair
    a = torch.randn(a_shape, dtype=torch.float32, device=flag_dnn.device)
    b = torch.randn(b_shape, dtype=torch.float32, device=flag_dnn.device)

    expected = _torch_fp32_matmul(a, b)
    flag_dnn_out = _run_flag_dnn_matmul_graph(a, b)

    utils.gems_assert_close(flag_dnn_out, expected, torch.float32, atol=2e-4)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("input_dtype", _MATMUL_DTYPES)
@pytest.mark.parametrize("out_dtype", _MATMUL_DTYPES)
def test_matmul_input_output_dtype_matrix(
    cudnn_handle,
    input_dtype,
    out_dtype,
):
    torch.manual_seed(11)
    a = torch.randint(-1, 2, (1, 32, 32), device=flag_dnn.device).to(
        input_dtype
    )
    b = torch.randint(-1, 2, (1, 32, 32), device=flag_dnn.device).to(
        input_dtype
    )
    compute_mode = (
        "fast_float_for_fp8" if input_dtype in _FP8_DTYPES else "float32"
    )

    expected = _cudnn_matmul(
        a,
        b,
        cudnn_handle,
        out_dtype=out_dtype,
        compute_mode=compute_mode,
    )
    actual = _run_flag_dnn_matmul_graph(
        a,
        b,
        out_dtype=out_dtype,
        compute_mode=compute_mode,
    )

    assert actual.dtype == out_dtype
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=0, rtol=0
    )


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_matmul_fp32_compute_modes(cudnn_handle):
    torch.manual_seed(13)
    a = torch.randn((1, 128, 512), dtype=torch.float32, device=flag_dnn.device)
    b = torch.randn((1, 512, 128), dtype=torch.float32, device=flag_dnn.device)

    ieee = _run_flag_dnn_matmul_graph(a, b, compute_mode="float32")
    tf32 = _run_flag_dnn_matmul_graph(a, b, compute_mode="tf32")
    expected_ieee = _torch_fp32_matmul(a, b)
    expected_tf32 = _cudnn_matmul(a, b, cudnn_handle, compute_mode="tf32")

    utils.gems_assert_close(ieee, expected_ieee, torch.float32, atol=2e-4)
    # Both paths use TF32 inputs and FP32 accumulation, but cuDNN and Triton
    # use different K-reduction trees on Hopper.
    utils.gems_assert_close(tf32, expected_tf32, torch.float32, atol=5e-3)
    assert (ieee - tf32).abs().max().item() > 1e-4


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 is required",
)
def test_matmul_sm90_cublaslt_tf32_matches_cudnn(cudnn_handle):
    torch.manual_seed(31)
    a = torch.randn(
        (32, 512, 512), dtype=torch.float32, device=flag_dnn.device
    )
    b = torch.randn(
        (32, 512, 512), dtype=torch.float32, device=flag_dnn.device
    )

    expected = _cudnn_matmul(a, b, cudnn_handle, compute_mode="tf32")
    actual = _run_flag_dnn_matmul_graph(a, b, compute_mode="tf32")

    utils.gems_assert_close(actual, expected, torch.float32, atol=5e-3)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 is required",
)
def test_matmul_sm90_cublaslt_bf16_id6_matches_cudnn(cudnn_handle):
    torch.manual_seed(41)
    a = torch.randn(
        (16, 2048, 512), dtype=torch.bfloat16, device=flag_dnn.device
    )
    b = torch.randn(
        (16, 512, 2048), dtype=torch.bfloat16, device=flag_dnn.device
    )

    expected = _cudnn_matmul(a, b, cudnn_handle)
    actual = _run_flag_dnn_matmul_graph(a, b)

    utils.gems_assert_close(actual, expected, torch.bfloat16, atol=1e-1)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("input_dtype", _FP8_DTYPES)
@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    (
        ((32, 64), (64, 24)),
        ((2, 32, 64), (2, 64, 24)),
        ((2, 1, 32, 64), (3, 64, 24)),
    ),
    ids=("2d", "prepared_3d", "broadcast"),
)
def test_matmul_fp8_rank_routes_match_cudnn(
    cudnn_handle,
    input_dtype,
    a_shape,
    b_shape,
):
    torch.manual_seed(17)
    a = torch.randint(-1, 2, a_shape, device=flag_dnn.device).to(input_dtype)
    b = torch.randint(-1, 2, b_shape, device=flag_dnn.device).to(input_dtype)

    expected = _cudnn_matmul_materialized_batch(
        a,
        b,
        cudnn_handle,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )
    actual = _run_flag_dnn_matmul_graph(
        a,
        b,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("input_dtype", _FP8_DTYPES)
def test_matmul_fp8_k_contiguous_b_matches_cudnn(
    cudnn_handle,
    input_dtype,
):
    torch.manual_seed(19)
    a = torch.randint(-1, 2, (2, 32, 64), device=flag_dnn.device).to(
        input_dtype
    )
    b_storage = torch.randint(-1, 2, (2, 24, 64), device=flag_dnn.device).to(
        input_dtype
    )
    b = b_storage.transpose(-2, -1)
    assert not b.is_contiguous()
    assert b.stride(-2) == 1

    expected = _cudnn_matmul(
        a,
        b,
        cudnn_handle,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )
    actual = _run_flag_dnn_matmul_graph(
        a,
        b,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("input_dtype", _FP8_DTYPES)
def test_matmul_fp8_prepared_cache_preserves_output_dtype(
    cudnn_handle,
    input_dtype,
):
    torch.manual_seed(23)
    a = torch.randint(-1, 2, (2, 32, 64), device=flag_dnn.device).to(
        input_dtype
    )
    b = torch.randint(-1, 2, (2, 64, 24), device=flag_dnn.device).to(
        input_dtype
    )
    compiled = _compile_flag_dnn_matmul_graph(
        a,
        b,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )

    expected_first = _cudnn_matmul(
        a,
        b,
        cudnn_handle,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )
    replay_a = a.clone()
    replay_b = b.clone()
    replay = compiled.bind(replay_a, replay_b)
    first = replay()
    first_snapshot = first.clone()

    a_second = torch.flip(a.float(), dims=(-1,)).to(input_dtype)
    b_second = torch.roll(b.float(), shifts=1, dims=-1).to(input_dtype)
    expected_second = _cudnn_matmul(
        a_second,
        b_second,
        cudnn_handle,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )
    replay_a.copy_(a_second)
    replay_b.copy_(b_second)
    second = replay()

    assert first.dtype == second.dtype == torch.float32
    assert first.data_ptr() == second.data_ptr()
    torch.testing.assert_close(first_snapshot, expected_first, atol=0, rtol=0)
    torch.testing.assert_close(second, expected_second, atol=0, rtol=0)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("input_dtype", "compute_mode"),
    (
        (torch.float16, "tf32"),
        (torch.float32, "fast_float_for_fp8"),
    ),
)
def test_matmul_rejects_incompatible_compute_mode(
    input_dtype,
    compute_mode,
):
    a = torch.ones((1, 8, 8), device=flag_dnn.device, dtype=input_dtype)
    b = torch.ones((1, 8, 8), device=flag_dnn.device, dtype=input_dtype)

    with pytest.raises(
        RuntimeError, match="unsupported matmul compute_data_type"
    ) as exc:
        _run_flag_dnn_matmul_graph(a, b, compute_mode=compute_mode)

    message = str(exc.value)
    assert compute_mode in message
    assert str(input_dtype) in message


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_matmul_rejects_unsupported_output_dtype():
    a = torch.ones((1, 8, 8), device=flag_dnn.device)
    b = torch.ones((1, 8, 8), device=flag_dnn.device)

    with pytest.raises(RuntimeError, match="matmul out_dtype must be"):
        _run_flag_dnn_matmul_graph(a, b, out_dtype=torch.float64)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_matmul_rejects_mixed_input_dtypes():
    a = torch.ones((1, 8, 8), device=flag_dnn.device, dtype=torch.float16)
    b = torch.ones((1, 8, 8), device=flag_dnn.device, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="same dtype"):
        _run_flag_dnn_matmul_graph(a, b)
