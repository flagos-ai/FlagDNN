from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

from flag_dnn.graph.graph import Graph
from flag_dnn.graph.node import OpNode


def apply_fusion_pass(graph: Graph) -> Graph:
    """Apply conservative first-phase graph fusion rewrites.

    Long patterns are matched first so short rewrites do not destroy a better
    candidate.  The graph IR can now represent conv2d+bias+relu as one fused
    node; planner/backend selection must choose a native FlagDNN Triton path.
    """

    stats = graph.attrs.setdefault("fusion", {})
    patterns = (
        ("conv2d_bias_activation_fused", _fuse_conv2d_bias_activation),
        ("conv2d_bias_folded", _fold_conv2d_bias),
        ("bias_activation_fused", _fuse_bias_activation),
    )
    changed = True
    while changed:
        changed = False
        for stat_key, rewrite in patterns:
            if rewrite(graph):
                stats[stat_key] = stats.get(stat_key, 0) + 1
                changed = True
                break
    return graph


def _fuse_conv2d_bias_activation(graph: Graph) -> bool:
    for act_node in list(graph.nodes):
        if not _is_unary_node(act_node, "relu"):
            continue

        bias_value = act_node.inputs[0]
        bias_node = _producer_of(graph, bias_value, "bias_add")
        if (
            bias_node is None
            or len(bias_node.inputs) != 2
            or len(bias_node.outputs) != 1
        ):
            continue
        if not _has_single_user(graph, bias_value):
            continue

        conv_value = bias_node.inputs[0]
        conv_node = _producer_of(graph, conv_value, "conv2d")
        if conv_node is None or len(conv_node.inputs) != 2:
            continue
        if not _has_single_user(graph, conv_value):
            continue

        attrs = dict(conv_node.attrs)
        attrs["activation"] = "relu"
        fused_node = _make_node(
            graph,
            op_type="fused_conv2d_bias_relu",
            inputs=(
                conv_node.inputs[0],
                conv_node.inputs[1],
                bias_node.inputs[1],
            ),
            outputs=act_node.outputs,
            attrs=attrs,
        )
        _replace_nodes(graph, (conv_node, bias_node, act_node), fused_node)
        return True
    return False


def _fold_conv2d_bias(graph: Graph) -> bool:
    for bias_node in list(graph.nodes):
        if bias_node.op_type != "bias_add":
            continue
        if len(bias_node.inputs) != 2 or len(bias_node.outputs) != 1:
            continue

        conv_value = bias_node.inputs[0]
        conv_node = _producer_of(graph, conv_value, "conv2d")
        if conv_node is None or len(conv_node.inputs) != 2:
            continue
        if not _has_single_user(graph, conv_value):
            continue

        bias_output = bias_node.outputs[0]
        conv_node.inputs = [
            conv_node.inputs[0],
            conv_node.inputs[1],
            bias_node.inputs[1],
        ]
        _redirect_value_uses(
            graph,
            old_value_id=bias_output,
            new_value_id=conv_value,
            mark_new_output=True,
        )
        graph.remove_nodes([bias_node.id])
        return True
    return False


def _fuse_bias_activation(graph: Graph) -> bool:
    for act_node in list(graph.nodes):
        if act_node.op_type not in ("relu", "gelu"):
            continue
        if not _is_unary_node(act_node):
            continue

        bias_value = act_node.inputs[0]
        bias_node = _producer_of(graph, bias_value, "bias_add")
        if bias_node is None or len(bias_node.inputs) != 2:
            continue
        if len(bias_node.outputs) != 1 or not _has_single_user(
            graph, bias_value
        ):
            continue

        fused_op = (
            "fused_bias_relu"
            if act_node.op_type == "relu"
            else "fused_bias_gelu"
        )
        attrs = dict(bias_node.attrs)
        attrs.update(act_node.attrs)
        attrs["activation"] = act_node.op_type
        fused_node = _make_node(
            graph,
            op_type=fused_op,
            inputs=bias_node.inputs,
            outputs=act_node.outputs,
            attrs=attrs,
        )
        _replace_nodes(graph, (bias_node, act_node), fused_node)
        return True
    return False


def _is_unary_node(node: OpNode, op_type: Optional[str] = None) -> bool:
    if op_type is not None and node.op_type != op_type:
        return False
    return len(node.inputs) == 1 and len(node.outputs) == 1


def _producer_of(
    graph: Graph, value_id: int, op_type: str
) -> Optional[OpNode]:
    node = graph.producer(value_id)
    if node is None or node.op_type != op_type:
        return None
    return node


def _has_single_user(graph: Graph, value_id: int) -> bool:
    return graph.num_users(value_id) == 1


def _make_node(
    graph: Graph,
    *,
    op_type: str,
    inputs: Iterable[int],
    outputs: Iterable[int],
    attrs: dict[str, Any],
) -> OpNode:
    return OpNode(
        id=graph.new_node_id(),
        op_type=op_type,
        inputs=list(inputs),
        outputs=list(outputs),
        attrs=dict(attrs),
        name=op_type,
    )


def _replace_nodes(
    graph: Graph, old_nodes: Iterable[OpNode], new_node: OpNode
) -> None:
    graph.replace_nodes_with((node.id for node in old_nodes), new_node)


def _redirect_value_uses(
    graph: Graph,
    *,
    old_value_id: int,
    new_value_id: int,
    mark_new_output: bool = False,
) -> None:
    if old_value_id in graph.outputs:
        graph.outputs = [
            new_value_id if value_id == old_value_id else value_id
            for value_id in graph.outputs
        ]
        if mark_new_output:
            value = graph.values[new_value_id]
            value.spec = value.spec.as_output()

    for user in graph.users(old_value_id):
        user.inputs = [
            new_value_id if value_id == old_value_id else value_id
            for value_id in user.inputs
        ]
