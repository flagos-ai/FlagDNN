from __future__ import annotations

from flag_dnn.graph.graph import Graph
from flag_dnn.graph.node import OpNode


def apply_fusion_pass(graph: Graph) -> Graph:
    """Apply conservative first-phase graph fusion rewrites.

    Long patterns are matched first so short rewrites do not destroy a better
    candidate.  The graph IR can now represent conv2d+bias+relu as one fused
    node; planner/backend selection must choose a native FlagDNN Triton path.
    """

    stats = graph.attrs.setdefault("fusion", {})
    changed = True
    while changed:
        changed = False
        if _fuse_conv2d_bias_activation(graph):
            stats["conv2d_bias_activation_fused"] = (
                stats.get("conv2d_bias_activation_fused", 0) + 1
            )
            changed = True
            continue
        if _fold_conv2d_bias(graph):
            stats["conv2d_bias_folded"] = (
                stats.get("conv2d_bias_folded", 0) + 1
            )
            changed = True
            continue
        if _fuse_bias_activation(graph):
            stats["bias_activation_fused"] = (
                stats.get("bias_activation_fused", 0) + 1
            )
            changed = True
    return graph


def _fuse_conv2d_bias_activation(graph: Graph) -> bool:
    for act_node in list(graph.nodes):
        if act_node.op_type != "relu":
            continue
        if len(act_node.inputs) != 1 or len(act_node.outputs) != 1:
            continue

        bias_value = act_node.inputs[0]
        bias_node = graph.producer(bias_value)
        if bias_node is None or bias_node.op_type != "bias_add":
            continue
        if len(bias_node.inputs) != 2 or len(bias_node.outputs) != 1:
            continue
        if graph.num_users(bias_value) != 1:
            continue

        conv_value = bias_node.inputs[0]
        conv_node = graph.producer(conv_value)
        if conv_node is None or conv_node.op_type != "conv2d":
            continue
        if len(conv_node.inputs) != 2:
            continue
        if graph.num_users(conv_value) != 1:
            continue

        attrs = dict(conv_node.attrs)
        attrs["activation"] = "relu"
        fused_node = OpNode(
            id=graph.new_node_id(),
            op_type="fused_conv2d_bias_relu",
            inputs=[
                conv_node.inputs[0],
                conv_node.inputs[1],
                bias_node.inputs[1],
            ],
            outputs=list(act_node.outputs),
            attrs=attrs,
            name="fused_conv2d_bias_relu",
        )
        graph.replace_nodes_with(
            [conv_node.id, bias_node.id, act_node.id], fused_node
        )
        return True
    return False


def _fold_conv2d_bias(graph: Graph) -> bool:
    for bias_node in list(graph.nodes):
        if bias_node.op_type != "bias_add":
            continue
        if len(bias_node.inputs) != 2 or len(bias_node.outputs) != 1:
            continue

        conv_value = bias_node.inputs[0]
        conv_node = graph.producer(conv_value)
        if conv_node is None or conv_node.op_type != "conv2d":
            continue
        if len(conv_node.inputs) != 2:
            continue
        if graph.num_users(conv_value) != 1:
            continue

        bias_output = bias_node.outputs[0]
        conv_node.inputs = [
            conv_node.inputs[0],
            conv_node.inputs[1],
            bias_node.inputs[1],
        ]

        if bias_output in graph.outputs:
            graph.outputs = [
                conv_value if value_id == bias_output else value_id
                for value_id in graph.outputs
            ]
            graph.values[conv_value].spec = graph.values[
                conv_value
            ].spec.as_output()

        for user in graph.users(bias_output):
            user.inputs = [
                conv_value if value_id == bias_output else value_id
                for value_id in user.inputs
            ]

        graph.remove_nodes([bias_node.id])
        return True
    return False


def _fuse_bias_activation(graph: Graph) -> bool:
    for act_node in list(graph.nodes):
        if act_node.op_type not in ("relu", "gelu"):
            continue
        if len(act_node.inputs) != 1 or len(act_node.outputs) != 1:
            continue
        bias_value = act_node.inputs[0]
        bias_node = graph.producer(bias_value)
        if bias_node is None or bias_node.op_type != "bias_add":
            continue
        if len(bias_node.outputs) != 1 or graph.num_users(bias_value) != 1:
            continue

        fused_op = (
            "fused_bias_relu"
            if act_node.op_type == "relu"
            else "fused_bias_gelu"
        )
        attrs = dict(bias_node.attrs)
        attrs.update(act_node.attrs)
        attrs["activation"] = act_node.op_type
        fused_node = OpNode(
            id=graph.new_node_id(),
            op_type=fused_op,
            inputs=list(bias_node.inputs),
            outputs=list(act_node.outputs),
            attrs=attrs,
            name=fused_op,
        )
        graph.replace_nodes_with([bias_node.id, act_node.id], fused_node)
        return True
    return False
