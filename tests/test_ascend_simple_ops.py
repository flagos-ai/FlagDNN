import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu

import flag_dnn


_ASCEND_AVAILABLE = (
    flag_dnn.vendor_name == "ascend"
    and hasattr(torch, "npu")
    and torch.npu.is_available()
)

pytestmark = pytest.mark.skipif(
    not _ASCEND_AVAILABLE,
    reason="Ascend simple-operator tests require an available NPU",
)


@pytest.fixture(scope="module", autouse=True)
def select_npu():
    torch.npu.set_device(0)
    yield
    torch.npu.synchronize()


def _assert_result(config_key, actual, expected):
    assert actual.device.type == "npu", config_key
    assert actual.shape == expected.shape, config_key
    assert actual.dtype == expected.dtype, config_key
    torch.npu.synchronize()
    torch.testing.assert_close(
        actual.detach().cpu(),
        expected,
        atol=5e-3,
        rtol=5e-3,
    )


@pytest.mark.parametrize(
    ("config_key", "op_name"),
    (
        ("abs", "abs"),
        ("neg", "neg"),
        ("sqrt", "sqrt"),
        ("unary", "exp"),
        ("relu", "relu"),
        ("leaky_relu", "leaky_relu"),
        ("elu", "elu"),
        ("gelu", "gelu"),
        ("silu", "swish"),
        ("softplus", "softplus"),
    ),
)
def test_ascend_simple_pointwise(config_key, op_name):
    if op_name == "sqrt":
        input_cpu = torch.linspace(0.1, 4.0, 1024, dtype=torch.float32)
    else:
        input_cpu = torch.linspace(-2.0, 2.0, 1024, dtype=torch.float32)
    input_npu = input_cpu.to("npu:0")

    if op_name == "abs":
        actual, expected = flag_dnn.abs(input_npu), input_cpu.abs()
    elif op_name == "neg":
        actual, expected = flag_dnn.neg(input_npu), input_cpu.neg()
    elif op_name == "sqrt":
        actual, expected = flag_dnn.sqrt(input_npu), input_cpu.sqrt()
    elif op_name == "exp":
        actual, expected = flag_dnn.exp(input_npu), input_cpu.exp()
    elif op_name == "relu":
        actual, expected = flag_dnn.relu(input_npu), torch.relu(input_cpu)
    elif op_name == "leaky_relu":
        actual = flag_dnn.leaky_relu(input_npu, negative_slope=0.1)
        expected = torch.nn.functional.leaky_relu(
            input_cpu, negative_slope=0.1
        )
    elif op_name == "elu":
        actual = flag_dnn.elu(input_npu, alpha=1.0)
        expected = torch.nn.functional.elu(input_cpu, alpha=1.0)
    elif op_name == "gelu":
        actual = flag_dnn.gelu(input_npu)
        expected = torch.nn.functional.gelu(input_cpu)
    elif op_name == "swish":
        actual = flag_dnn.swish(input_npu)
        expected = torch.nn.functional.silu(input_cpu)
    elif op_name == "softplus":
        actual = flag_dnn.softplus(input_npu)
        expected = torch.nn.functional.softplus(input_cpu)
    else:
        raise AssertionError(f"unhandled pointwise op: {op_name}")

    _assert_result(config_key, actual, expected)


def test_ascend_add_square():
    left_cpu = torch.linspace(-2.0, 2.0, 1024, dtype=torch.float32)
    right_cpu = torch.linspace(0.1, 1.1, 1024, dtype=torch.float32)
    actual = flag_dnn.add_square(left_cpu.to("npu:0"), right_cpu.to("npu:0"))
    _assert_result("add_square", actual, left_cpu + right_cpu.square())


def test_ascend_layer_norm():
    generator = torch.Generator(device="cpu").manual_seed(0)
    input_cpu = torch.randn((8, 64), generator=generator)
    weight_cpu = torch.randn((64,), generator=generator)
    bias_cpu = torch.randn((64,), generator=generator)
    actual = flag_dnn.layer_norm(
        input_cpu.to("npu:0"),
        (64,),
        weight_cpu.to("npu:0"),
        bias_cpu.to("npu:0"),
    )
    expected = torch.nn.functional.layer_norm(
        input_cpu, (64,), weight_cpu, bias_cpu
    )
    _assert_result("layer_norm", actual, expected)


def test_ascend_rms_norm():
    generator = torch.Generator(device="cpu").manual_seed(0)
    input_cpu = torch.randn((8, 64), generator=generator)
    weight_cpu = torch.randn((64,), generator=generator)
    input_npu = input_cpu.to("npu:0")
    actual = flag_dnn.rms_norm(input_npu, (64,), weight_cpu.to("npu:0"))
    expected = (
        input_cpu
        * torch.rsqrt(input_cpu.square().mean(-1, keepdim=True) + 1e-5)
        * weight_cpu
    )
    _assert_result("rms_norm", actual, expected)
