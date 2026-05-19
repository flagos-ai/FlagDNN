import torch

import flag_dnn


def test_graph_capture_fuses_bias_relu_and_eliminates_dead_node():
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


def test_graph_executor_cpu_fallback_matches_torch():
    @flag_dnn.graph
    def fn(x, bias):
        return flag_dnn.relu(flag_dnn.bias_add(x, bias))

    x = torch.randn(2, 3, 4)
    bias = torch.randn(3)
    compiled = flag_dnn.compile(fn, inputs=[x, bias], options={"cache": None})

    out = compiled.run(x, bias)
    ref = torch.relu(x + bias.reshape(1, 3, 1))
    torch.testing.assert_close(out, ref)


def test_graph_operator_overload_capture():
    @flag_dnn.graph
    def fn(x, y):
        return flag_dnn.relu(x + y)

    x = torch.randn(4, 5)
    y = torch.randn(4, 5)
    compiled = flag_dnn.compile(fn, inputs=[x, y], options={"cache": None})

    op_types = [node.op_type for node in compiled.graph.nodes]
    assert op_types == ["add", "relu"]
    torch.testing.assert_close(compiled(x, y), torch.relu(x + y))


def test_graph_conv2d_bias_relu_fuses_in_ir():
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


def test_graph_memory_plan_cache_hit():
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


def test_graph_autotune_framework_records_candidates_on_cpu():
    @flag_dnn.graph
    def fn(x, bias):
        return flag_dnn.relu(flag_dnn.bias_add(x, bias))

    x = torch.randn(2, 3)
    bias = torch.randn(3)
    compiled = flag_dnn.compile(
        fn,
        inputs=[x, bias],
        options={"cache": None, "autotune": True, "autotune_repeat": 1},
    )

    assert "autotune" in compiled.plan.debug_info
    assert compiled.plan.debug_info["autotune"]["enabled"] is True
    assert compiled.plan.debug_info["candidate_count"] >= 1

def test_graph_ops_namespace_add_capture():
    @flag_dnn.graph
    def fn(x, y):
        return flag_dnn.ops.add(
            x,
            y,
            compute_data_type="float32",
            name="add",
        )

    x = torch.randn(4, 5)
    y = torch.randn(4, 5)
    compiled = flag_dnn.compile(fn, inputs=[x, y], options={"cache": None})

    assert [node.op_type for node in compiled.graph.nodes] == ["add"]
    assert compiled.graph.nodes[0].attrs["name"] == "add"
    torch.testing.assert_close(compiled(x, y), x + y)
