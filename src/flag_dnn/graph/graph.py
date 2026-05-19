from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Optional

import torch

from flag_dnn.graph.node import GraphValue, OpNode
from flag_dnn.graph.registry import get_op_schema
from flag_dnn.graph.tensor import GraphTensor, TensorSpec


class Graph:
    def __init__(self) -> None:
        self.inputs: list[int] = []
        self.outputs: list[int] = []
        self.nodes: list[OpNode] = []
        self.values: dict[int, GraphValue] = {}
        self.attrs: dict[str, Any] = {}
        self._next_value_id = 0
        self._next_node_id = 0

    def new_value_id(self) -> int:
        value_id = self._next_value_id
        self._next_value_id += 1
        return value_id

    def new_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def add_input(self, spec: TensorSpec) -> GraphTensor:
        value_id = self.new_value_id()
        name = spec.name or f"arg{len(self.inputs)}"
        input_spec = spec.with_name(name).as_input()
        self.values[value_id] = GraphValue(value_id, input_spec)
        self.inputs.append(value_id)
        return GraphTensor(value_id, self)

    def add_constant(
        self, value: Any, name_hint: str = "const"
    ) -> GraphTensor:
        if isinstance(value, torch.Tensor):
            spec = TensorSpec.from_tensor(
                value, name=f"{name_hint}{self._next_value_id}"
            )
        else:
            dtype = torch.tensor(value).dtype
            spec = TensorSpec(
                name=f"{name_hint}{self._next_value_id}",
                shape=(),
                dtype=dtype,
            )
        value_id = self.new_value_id()
        self.values[value_id] = GraphValue(
            value_id,
            spec,
            const_value=value,
            is_constant=True,
        )
        return GraphTensor(value_id, self)

    def add_op(
        self,
        op_type: str,
        inputs: list[int],
        attrs: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> GraphTensor:
        attrs = {} if attrs is None else dict(attrs)
        schema = get_op_schema(op_type)
        input_specs = [self.values[value_id].spec for value_id in inputs]
        node_id = self.new_node_id()
        output_specs = schema.infer_outputs(input_specs, attrs)

        outputs = []
        for index, output_spec in enumerate(output_specs):
            value_id = self.new_value_id()
            value_name = output_spec.name or f"{op_type}_{node_id}_{index}"
            self.values[value_id] = GraphValue(
                value_id,
                output_spec.with_name(value_name),
                producer=node_id,
            )
            outputs.append(value_id)

        node = OpNode(
            id=node_id,
            op_type=op_type,
            inputs=list(inputs),
            outputs=outputs,
            attrs=attrs,
            name=name,
        )
        self.nodes.append(node)
        for value_id in inputs:
            self.values[value_id].users.append(node_id)
        if len(outputs) != 1:
            raise NotImplementedError("multi-output graph ops are not enabled")
        return GraphTensor(outputs[0], self)

    def get_node(self, node_id: int) -> OpNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"graph node not found: {node_id}")

    def producer(self, value_id: int) -> Optional[OpNode]:
        producer_id = self.values[value_id].producer
        if producer_id is None:
            return None
        try:
            return self.get_node(producer_id)
        except KeyError:
            return None

    def users(self, value_id: int) -> list[OpNode]:
        result = []
        for node_id in self.values[value_id].users:
            try:
                result.append(self.get_node(node_id))
            except KeyError:
                continue
        return result

    def num_users(self, value_id: int) -> int:
        return len(self.users(value_id))

    def replace_input(self, old_value_id: int, new_value_id: int) -> None:
        for node in self.nodes:
            node.inputs = [
                new_value_id if value_id == old_value_id else value_id
                for value_id in node.inputs
            ]
        self.outputs = [
            new_value_id if value_id == old_value_id else value_id
            for value_id in self.outputs
        ]
        self.rebuild_metadata()

    def remove_nodes(self, node_ids: Iterable[int]) -> None:
        remove = set(node_ids)
        self.nodes = [node for node in self.nodes if node.id not in remove]
        self.rebuild_metadata()

    def replace_nodes_with(
        self,
        old_node_ids: Iterable[int],
        new_node: OpNode,
    ) -> None:
        old = set(old_node_ids)
        new_nodes: list[OpNode] = []
        inserted = False
        for node in self.nodes:
            if node.id in old:
                if not inserted:
                    new_nodes.append(new_node)
                    inserted = True
                continue
            new_nodes.append(node)
        if not inserted:
            new_nodes.append(new_node)
        self.nodes = new_nodes
        for output in new_node.outputs:
            self.values[output].producer = new_node.id
        self.rebuild_metadata()

    def rebuild_metadata(self) -> None:
        live_node_ids = {node.id for node in self.nodes}
        for value in self.values.values():
            value.users = []
            if (
                value.producer is not None
                and value.producer not in live_node_ids
            ):
                value.producer = None
        for node in self.nodes:
            for output in node.outputs:
                self.values[output].producer = node.id
            for value_id in node.inputs:
                if node.id not in self.values[value_id].users:
                    self.values[value_id].users.append(node.id)

    def mark_outputs(self, outputs: list[int]) -> None:
        self.outputs = list(outputs)
        for value_id in outputs:
            value = self.values[value_id]
            value.spec = value.spec.as_output()

    def topological_nodes(self) -> list[OpNode]:
        return list(self.nodes)

    def lint(self) -> None:
        node_ids = {node.id for node in self.nodes}
        for value_id in self.inputs + self.outputs:
            if value_id not in self.values:
                raise RuntimeError(
                    f"graph references missing value {value_id}"
                )
        for node in self.nodes:
            if node.id not in node_ids:
                raise RuntimeError(f"invalid graph node id {node.id}")
            for value_id in node.inputs + node.outputs:
                if value_id not in self.values:
                    raise RuntimeError(
                        f"node {node.id} references missing value {value_id}"
                    )
            for output in node.outputs:
                if self.values[output].producer != node.id:
                    raise RuntimeError(
                        f"value {output} producer metadata is inconsistent"
                    )

    def to_dict(self, include_values: bool = True) -> dict[str, Any]:
        data = {
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "nodes": [node.to_dict() for node in self.nodes],
            "attrs": self._jsonable(self.attrs),
        }
        if include_values:
            data["values"] = {
                str(value_id): value.to_dict(include_const=True)
                for value_id, value in sorted(self.values.items())
            }
        return data

    def graph_hash(self) -> str:
        payload = self.to_dict(include_values=True)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def dump_text(self) -> str:
        lines = ["Graph("]
        input_names = [
            self.values[value_id].spec.name for value_id in self.inputs
        ]
        output_names = [
            self.values[value_id].spec.name for value_id in self.outputs
        ]
        lines.append(f"  inputs={input_names}")
        for node in self.nodes:
            in_names = [
                self.values[value_id].spec.name for value_id in node.inputs
            ]
            out_names = [
                self.values[value_id].spec.name for value_id in node.outputs
            ]
            lines.append(
                f"  %{node.id} {node.op_type}({in_names}) -> {out_names} "
                f"attrs={node.attrs}"
            )
        lines.append(f"  outputs={output_names}")
        lines.append(")")
        return "\n".join(lines)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: Graph._jsonable(val) for key, val in value.items()}
        if isinstance(value, tuple):
            return [Graph._jsonable(item) for item in value]
        if isinstance(value, list):
            return [Graph._jsonable(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
