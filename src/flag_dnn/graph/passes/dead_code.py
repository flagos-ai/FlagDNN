from __future__ import annotations

from flag_dnn.graph.graph import Graph


def eliminate_dead_nodes(graph: Graph) -> Graph:
    live_values: set[int] = set()
    live_nodes: set[int] = set()

    def visit_value(value_id: int) -> None:
        if value_id in live_values:
            return
        live_values.add(value_id)
        value = graph.values[value_id]
        if value.producer is not None:
            visit_node(value.producer)

    def visit_node(node_id: int) -> None:
        if node_id in live_nodes:
            return
        live_nodes.add(node_id)
        node = graph.get_node(node_id)
        for input_id in node.inputs:
            visit_value(input_id)

    for output_id in graph.outputs:
        visit_value(output_id)

    if len(live_nodes) != len(graph.nodes):
        graph.nodes = [node for node in graph.nodes if node.id in live_nodes]
        graph.rebuild_metadata()
        graph.attrs.setdefault("passes", {})["dead_code_eliminated"] = True
    return graph
