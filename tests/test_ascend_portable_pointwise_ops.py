import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu

import flag_dnn


DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_ASCEND_AVAILABLE = (
    flag_dnn.vendor_name == "ascend"
    and hasattr(torch, "npu")
    and torch.npu.is_available()
)
pytestmark = pytest.mark.skipif(
    not _ASCEND_AVAILABLE,
    reason="portable pointwise tests require an available Ascend NPU",
)


@pytest.fixture(scope="module", autouse=True)
def select_npu():
    torch.npu.set_device(0)
    yield
    torch.npu.synchronize()


def _assert_close(actual, expected, dtype):
    assert actual.device.type == "npu"
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.npu.synchronize()
    tolerance = 1e-4 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(
        actual.detach().cpu(),
        expected,
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_ascend_tan_eager(dtype):
    input_cpu = torch.linspace(-1.0, 1.0, 1024).to(dtype)
    actual = flag_dnn.tan(input_cpu.to("npu:0"))
    _assert_close(actual, torch.tan(input_cpu), dtype)


def test_ascend_tan_out():
    input_cpu = torch.linspace(-1.0, 1.0, 1024)
    input_npu = input_cpu.to("npu:0")
    out = torch.empty_like(input_npu)
    actual = flag_dnn.tan(input_npu, out=out)
    assert actual is out
    _assert_close(actual, torch.tan(input_cpu), torch.float32)


def test_ascend_tan_graph():
    input_cpu = torch.linspace(-1.0, 1.0, 1024)
    input_npu = input_cpu.to("npu:0")

    @flag_dnn.graph
    def tan_graph(x):
        return flag_dnn.tan(
            x, compute_data_type="float32", name="tan"
        )

    compiled = flag_dnn.compile(
        tan_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(input_npu, "x")],
        options={"backend": "npu", "cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["tan"]
    actual = compiled.run(input_npu.clone())
    _assert_close(actual, torch.tan(input_cpu), torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "path",
    ("tensor_tensor", "broadcast", "tensor_scalar", "scalar_tensor"),
)
def test_ascend_pow_eager(dtype, path):
    if path == "tensor_tensor":
        base = torch.linspace(0.5, 2.0, 1024).to(dtype)
        exponent = torch.linspace(0.25, 1.5, 1024).to(dtype)
        actual = flag_dnn.pow(base.to("npu:0"), exponent.to("npu:0"))
        expected = torch.pow(base, exponent)
    elif path == "broadcast":
        base = torch.linspace(0.5, 2.0, 128).reshape(8, 1, 16).to(dtype)
        exponent = torch.linspace(0.25, 1.5, 4).reshape(1, 4, 1).to(dtype)
        actual = flag_dnn.pow(base.to("npu:0"), exponent.to("npu:0"))
        expected = torch.pow(base, exponent)
    elif path == "tensor_scalar":
        base = torch.linspace(0.5, 2.0, 1024).to(dtype)
        actual = flag_dnn.pow(base.to("npu:0"), 1.5)
        expected = torch.pow(base, 1.5)
    else:
        exponent = torch.linspace(0.25, 1.5, 1024).to(dtype)
        actual = flag_dnn.pow(1.25, exponent.to("npu:0"))
        expected = torch.pow(1.25, exponent)
    _assert_close(actual, expected, dtype)


def test_ascend_pow_out():
    base = torch.linspace(0.5, 2.0, 1024)
    exponent = torch.linspace(0.25, 1.5, 1024)
    base_npu, exponent_npu = base.to("npu:0"), exponent.to("npu:0")
    out = torch.empty_like(base_npu)
    actual = flag_dnn.pow(base_npu, exponent_npu, out=out)
    assert actual is out
    _assert_close(actual, torch.pow(base, exponent), torch.float32)


def test_ascend_pow_graph():
    base = torch.linspace(0.5, 2.0, 1024)
    exponent = torch.linspace(0.25, 1.5, 1024)
    base_npu, exponent_npu = base.to("npu:0"), exponent.to("npu:0")

    @flag_dnn.graph
    def pow_graph(left, right):
        return flag_dnn.pow(
            left,
            right,
            compute_data_type="float32",
            name="pow",
        )

    compiled = flag_dnn.compile(
        pow_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(base_npu, "left"),
            flag_dnn.TensorSpec.from_tensor(exponent_npu, "right"),
        ],
        options={"backend": "npu", "cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["pow"]
    actual = compiled.run(base_npu.clone(), exponent_npu.clone())
    _assert_close(actual, torch.pow(base, exponent), torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ascend_tanh_eager(dtype):
    input_cpu = torch.linspace(-4.0, 4.0, 1024).to(dtype)
    actual = flag_dnn.tanh(input_cpu.to("npu:0"))
    _assert_close(actual, torch.tanh(input_cpu), dtype)


def test_ascend_tanh_graph():
    input_cpu = torch.linspace(-4.0, 4.0, 1024)
    input_npu = input_cpu.to("npu:0")

    @flag_dnn.graph
    def tanh_graph(x):
        return flag_dnn.tanh(x)

    compiled = flag_dnn.compile(
        tanh_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(input_npu, "x")],
        options={"backend": "npu", "cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["tanh"]
    actual = compiled.run(input_npu.clone())
    _assert_close(actual, torch.tanh(input_cpu), torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ascend_sigmoid_eager(dtype):
    input_cpu = torch.linspace(-6.0, 6.0, 1024).to(dtype)
    actual = flag_dnn.sigmoid(input_cpu.to("npu:0"))
    _assert_close(actual, torch.sigmoid(input_cpu), dtype)


def test_ascend_sigmoid_out():
    input_cpu = torch.linspace(-6.0, 6.0, 1024)
    input_npu = input_cpu.to("npu:0")
    out = torch.empty_like(input_npu)
    actual = flag_dnn.sigmoid(input_npu, out=out)
    assert actual is out
    _assert_close(actual, torch.sigmoid(input_cpu), torch.float32)


def test_ascend_sigmoid_graph():
    input_cpu = torch.linspace(-6.0, 6.0, 1024)
    input_npu = input_cpu.to("npu:0")

    @flag_dnn.graph
    def sigmoid_graph(x):
        return flag_dnn.sigmoid(
            x, compute_data_type="float32", name="sigmoid"
        )

    compiled = flag_dnn.compile(
        sigmoid_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(input_npu, "x")],
        options={"backend": "npu", "cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["sigmoid"]
    actual = compiled.run(input_npu.clone())
    _assert_close(actual, torch.sigmoid(input_cpu), torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ascend_sigmoid_backward_eager(dtype):
    input_cpu = torch.linspace(-6.0, 6.0, 1024).to(dtype)
    loss_cpu = torch.linspace(-1.0, 1.0, 1024).to(dtype)
    y = torch.sigmoid(input_cpu)
    expected = loss_cpu * y * (1.0 - y)
    actual = flag_dnn.sigmoid_backward(
        loss_cpu.to("npu:0"), input_cpu.to("npu:0")
    )
    _assert_close(actual, expected, dtype)


def test_ascend_sigmoid_backward_out():
    input_cpu = torch.linspace(-6.0, 6.0, 1024)
    loss_cpu = torch.linspace(-1.0, 1.0, 1024)
    input_npu, loss_npu = input_cpu.to("npu:0"), loss_cpu.to("npu:0")
    out = torch.empty_like(input_npu)
    actual = flag_dnn.sigmoid_backward(loss_npu, input_npu, out=out)
    expected_y = torch.sigmoid(input_cpu)
    expected = loss_cpu * expected_y * (1.0 - expected_y)
    assert actual is out
    _assert_close(actual, expected, torch.float32)


def test_ascend_sigmoid_backward_graph():
    input_cpu = torch.linspace(-6.0, 6.0, 1024)
    loss_cpu = torch.linspace(-1.0, 1.0, 1024)
    input_npu, loss_npu = input_cpu.to("npu:0"), loss_cpu.to("npu:0")

    @flag_dnn.graph
    def backward_graph(loss, x):
        return flag_dnn.sigmoid_backward(
            loss,
            x,
            compute_data_type="float32",
            name="sigmoid_backward",
        )

    compiled = flag_dnn.compile(
        backward_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(loss_npu, "loss"),
            flag_dnn.TensorSpec.from_tensor(input_npu, "x"),
        ],
        options={"backend": "npu", "cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == [
        "sigmoid_backward"
    ]
    actual = compiled.run(loss_npu.clone(), input_npu.clone())
    expected_y = torch.sigmoid(input_cpu)
    expected = loss_cpu * expected_y * (1.0 - expected_y)
    _assert_close(actual, expected, torch.float32)
