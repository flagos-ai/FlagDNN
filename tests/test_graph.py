import pytest
import torch

import flag_dnn
from tests.base import cudnn, cudnn_graph, execute_cudnn_graph


def _contiguous_stride(shape):
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def _cudnn_add(x, y, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.tensor_like(y)
    out_tensor = graph.add(
        a=x_tensor,
        b=y_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="add",
    )
    output_shape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
    out_tensor.set_dim(output_shape).set_stride(
        _contiguous_stride(output_shape)
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x, y_tensor: y},
        out_tensor,
        torch.empty(output_shape, device=x.device, dtype=x.dtype),
        cudnn_handle,
        "add",
    )


def _cudnn_relu(x, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    out_tensor = graph.relu(
        input=x_tensor,
        negative_slope=0.0,
        lower_clip=0.0,
        compute_data_type=cudnn.data_type.FLOAT,
        name="relu",
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        out_tensor,
        torch.empty_like(x),
        cudnn_handle,
        "relu",
    )


def _cudnn_add_relu(x, y, cudnn_handle):
    return _cudnn_relu(_cudnn_add(x, y, cudnn_handle), cudnn_handle)


def test_capture_fuses_bias_relu_and_eliminates_dead_node():
    @flag_dnn.graph
    def fn(x, bias):
        y = flag_dnn.bias_add(x, bias)
        out = flag_dnn.relu(y)
        flag_dnn.relu(x)
        return out

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec("x", (2, 3, 4), "float32"),
            flag_dnn.TensorSpec("bias", (3,), "float32"),
        ],
        options={"cache": None},
    )

    op_types = [node.op_type for node in compiled.graph.nodes]
    assert op_types == ["fused_bias_relu"]
    assert compiled.plan.debug_info["fusion"]["bias_activation_fused"] == 1


def test_executor_rejects_cpu_without_torch_fallback():
    @flag_dnn.graph
    def fn(x, bias):
        return flag_dnn.relu(flag_dnn.bias_add(x, bias))

    x = torch.randn(2, 3, 4)
    bias = torch.randn(3)
    with pytest.raises(RuntimeError, match="no execution candidates"):
        flag_dnn.compile(fn, inputs=[x, bias], options={"cache": None})


def test_torch_backend_is_not_available():
    with pytest.raises(ValueError, match="no longer supports torch fallback"):
        flag_dnn.resolve_backend("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_operator_overload_capture(cudnn_handle):
    @flag_dnn.graph
    def fn(x, y):
        return flag_dnn.relu(x + y)

    x = torch.randn(1, 4, 5, device=flag_dnn.device)
    y = torch.randn(1, 4, 5, device=flag_dnn.device)
    compiled = flag_dnn.compile(fn, inputs=[x, y], options={"cache": None})

    op_types = [node.op_type for node in compiled.graph.nodes]
    assert op_types == ["add", "relu"]
    cudnn_out = _cudnn_add_relu(x, y, cudnn_handle)
    torch.testing.assert_close(compiled(x, y), cudnn_out)


def test_conv2d_bias_relu_fuses_in_ir():
    @flag_dnn.graph
    def fn(x, weight, bias):
        y = flag_dnn.conv2d(x, weight, padding=1)
        y = flag_dnn.bias_add(y, bias)
        return flag_dnn.relu(y)

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec("x", (1, 3, 8, 8), "float32"),
            flag_dnn.TensorSpec("weight", (4, 3, 3, 3), "float32"),
            flag_dnn.TensorSpec("bias", (4,), "float32"),
        ],
        options={"cache": None},
    )

    op_types = [node.op_type for node in compiled.graph.nodes]
    assert op_types == ["fused_conv2d_bias_relu"]
    assert len(compiled.graph.nodes[0].inputs) == 3
    assert (
        compiled.plan.debug_info["fusion"]["conv2d_bias_activation_fused"] == 1
    )
    assert compiled.plan.debug_info["kernel_candidates"][0]["op_type"] == (
        "fused_conv2d_bias_relu"
    )


def test_memory_plan_cache_hit():
    cache = flag_dnn.PlanCache(enable_disk=False)

    @flag_dnn.graph
    def fn(x, bias):
        return flag_dnn.relu(flag_dnn.bias_add(x, bias))

    specs = [
        flag_dnn.TensorSpec("x", (2, 3), "float32"),
        flag_dnn.TensorSpec("bias", (3,), "float32"),
    ]
    first = flag_dnn.compile(fn, inputs=specs, options={"cache": cache})
    second = flag_dnn.compile(fn, inputs=specs, options={"cache": cache})

    assert first.plan.workspace_size >= 0
    assert first.plan.memory_plan is not None
    assert second.plan.debug_info["cache_hit"] is True
    assert second.plan.debug_info["cache_layer"] == "memory"


def test_autotune_rejects_cpu_without_torch_fallback():
    @flag_dnn.graph
    def fn(x, bias):
        return flag_dnn.relu(flag_dnn.bias_add(x, bias))

    x = torch.randn(2, 3)
    bias = torch.randn(3)
    with pytest.raises(RuntimeError, match="no execution candidates"):
        flag_dnn.compile(
            fn,
            inputs=[x, bias],
            options={"cache": None, "autotune": True, "autotune_repeat": 1},
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ops_namespace_add_capture():
    @flag_dnn.graph
    def fn(x, y):
        return flag_dnn.ops.add(
            x,
            y,
            compute_data_type="float32",
            name="add",
        )

    x = torch.randn(1, 4, 5, device=flag_dnn.device)
    y = torch.randn(1, 4, 5, device=flag_dnn.device)
    compiled = flag_dnn.compile(fn, inputs=[x, y], options={"cache": None})

    assert [node.op_type for node in compiled.graph.nodes] == ["add"]
    assert compiled.graph.nodes[0].attrs["name"] == "add"
    torch.testing.assert_close(compiled(x, y), x + y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ops_namespace_sub_capture():
    @flag_dnn.graph
    def fn(x, y):
        return flag_dnn.ops.sub(
            x,
            y,
            compute_data_type="float32",
            name="sub",
        )

    x = torch.randn(1, 4, 5, device=flag_dnn.device)
    y = torch.randn(1, 4, 5, device=flag_dnn.device)
    compiled = flag_dnn.compile(fn, inputs=[x, y], options={"cache": None})

    assert [node.op_type for node in compiled.graph.nodes] == ["sub"]
    assert compiled.graph.nodes[0].attrs["name"] == "sub"
    torch.testing.assert_close(compiled(x, y), x - y)
