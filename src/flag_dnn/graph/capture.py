from __future__ import annotations

import functools
import json
import os
from typing import Any, Callable, Optional

import torch

from flag_dnn.graph.cache import PlanCacheKey, get_default_plan_cache
from flag_dnn.graph.graph import Graph
from flag_dnn.graph.passes import apply_fusion_pass, eliminate_dead_nodes
from flag_dnn.graph.planner import Planner
from flag_dnn.graph.registry import get_op_schema
from flag_dnn.graph.tensor import GraphTensor, TensorSpec

_CAPTURE_STACK: list["GraphCapture"] = []


def is_capturing() -> bool:
    return bool(_CAPTURE_STACK)


def current_capture() -> Optional["GraphCapture"]:
    return _CAPTURE_STACK[-1] if _CAPTURE_STACK else None


class GraphCapture:
    def __init__(self, input_specs: list[TensorSpec]):
        self.graph = Graph()
        self.input_specs = input_specs
        self.graph_inputs: list[GraphTensor] = []

    def __enter__(self) -> "GraphCapture":
        _CAPTURE_STACK.append(self)
        for index, spec in enumerate(self.input_specs):
            named = spec if spec.name else spec.with_name(f"arg{index}")
            self.graph_inputs.append(self.graph.add_input(named))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        popped = _CAPTURE_STACK.pop()
        if popped is not self:
            raise RuntimeError("FlagDNN graph capture stack is corrupted")

    def finalize(self, outputs: Any) -> tuple[Graph, Any]:
        output_ids, structure = flatten_outputs(outputs)
        self.graph.mark_outputs(output_ids)
        self.graph.lint()
        return self.graph, structure

    def is_tensor_like(self, value: Any) -> bool:
        return isinstance(value, (GraphTensor, torch.Tensor))

    def as_value(self, value: Any, name_hint: str = "value") -> int:
        if isinstance(value, GraphTensor):
            if value.graph is not self.graph:
                raise RuntimeError("cannot mix GraphTensor values from graphs")
            return value.value_id
        if isinstance(value, torch.Tensor):
            return self.graph.add_constant(value, name_hint=name_hint).value_id
        if isinstance(value, (int, float, bool)):
            return self.graph.add_constant(value, name_hint=name_hint).value_id
        raise TypeError(f"unsupported graph operand type: {type(value)!r}")

    def add_op_call(
        self,
        op_type: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphTensor:
        schema = get_op_schema(op_type)
        input_ids, attrs = schema.normalize(self, args, dict(kwargs))
        return self.graph.add_op(op_type, input_ids, attrs)

    def add_binary_op(
        self,
        op_type: str,
        left: Any,
        right: Any,
        reverse: bool = False,
    ) -> GraphTensor:
        attrs = {"op_type": op_type, "alpha": 1, "rounding_mode": None}
        if reverse and self.is_tensor_like(right):
            input_ids = [
                self.as_value(right, "other"),
                self.as_value(left, "input"),
            ]
        else:
            input_ids = [self.as_value(left, "input")]
            if self.is_tensor_like(right):
                input_ids.append(self.as_value(right, "other"))
            else:
                attrs["other"] = right
                attrs["reverse"] = reverse
        return self.graph.add_op(op_type, input_ids, attrs)


class GraphFunction:
    def __init__(self, fn: Callable[..., Any], **options: Any) -> None:
        self.fn = fn
        self.options = dict(options)
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


class CompiledGraph:
    def __init__(self, plan: Any):
        self.plan = plan
        self.graph = plan.graph

    def run(self, *inputs: Any) -> Any:
        return self.plan.run(*inputs)

    def __call__(self, *inputs: Any) -> Any:
        return self.run(*inputs)


def graph(fn: Optional[Callable[..., Any]] = None, **options: Any):
    if fn is None:
        return lambda inner: GraphFunction(inner, **options)
    return GraphFunction(fn, **options)


def compile(
    fn: Callable[..., Any],
    inputs: Optional[list[Any]] = None,
    input_specs: Optional[list[Any]] = None,
    options: Optional[dict[str, Any]] = None,
) -> CompiledGraph:
    compile_options = {}
    if isinstance(fn, GraphFunction):
        compile_options.update(fn.options)
        python_fn = fn.fn
    else:
        python_fn = fn
    if options:
        compile_options.update(options)

    input_values = input_specs if input_specs is not None else inputs
    specs = _normalize_input_specs(input_values)
    runtime_inputs = _runtime_inputs_from_values(input_values)
    if not specs:
        raise ValueError(
            "flag_dnn.compile requires input TensorSpec values or tensors"
        )

    with GraphCapture(specs) as ctx:
        outputs = python_fn(*ctx.graph_inputs)
    captured_graph, output_structure = ctx.finalize(outputs)
    _dump_graph(captured_graph, "capture")

    eliminate_dead_nodes(captured_graph)
    if _fusion_enabled(compile_options):
        apply_fusion_pass(captured_graph)
        eliminate_dead_nodes(captured_graph)
    captured_graph.lint()
    _dump_graph(captured_graph, "optimized")

    backend = str(compile_options.get("backend", "auto"))
    cache = compile_options.get("cache", get_default_plan_cache())
    key = PlanCacheKey.from_graph(
        captured_graph,
        specs,
        backend=backend,
        flagdnn_version=_flagdnn_version(),
    )
    plan = cache.get(key, captured_graph) if cache is not None else None
    if plan is None:
        planner = Planner(backend=backend, options=compile_options)
        plan = planner.build_plan(
            captured_graph,
            specs,
            output_structure,
            runtime_inputs=runtime_inputs,
        )
        if cache is not None:
            cache.put(key, plan)
    else:
        plan.debug_info["output_structure"] = output_structure
    return CompiledGraph(plan)


def flatten_outputs(outputs: Any) -> tuple[list[int], Any]:
    flat: list[int] = []

    def visit(value: Any) -> Any:
        if isinstance(value, GraphTensor):
            flat.append(value.value_id)
            return ("leaf",)
        if isinstance(value, tuple):
            return ("tuple", [visit(item) for item in value])
        if isinstance(value, list):
            return ("list", [visit(item) for item in value])
        if isinstance(value, dict):
            return (
                "dict",
                [(key, visit(item)) for key, item in value.items()],
            )
        raise TypeError(
            "FlagDNN graph functions must return GraphTensor values, "
            f"got {type(value)!r}"
        )

    structure = visit(outputs)
    return flat, structure


def _normalize_input_specs(values: Optional[list[Any]]) -> list[TensorSpec]:
    if values is None:
        return []
    specs = []
    for index, value in enumerate(values):
        if isinstance(value, TensorSpec):
            spec = value
        elif isinstance(value, torch.Tensor):
            spec = TensorSpec.from_tensor(value)
        else:
            raise TypeError(
                "flag_dnn.compile inputs must be TensorSpec or torch.Tensor "
                f"values, got {type(value)!r}"
            )
        if not spec.name:
            spec = spec.with_name(f"arg{index}")
        specs.append(spec)
    return specs


def _runtime_inputs_from_values(
    values: Optional[list[Any]],
) -> Optional[tuple[Any, ...]]:
    if values is None:
        return None
    if all(isinstance(value, torch.Tensor) for value in values):
        return tuple(values)
    return None


def _fusion_enabled(options: dict[str, Any]) -> bool:
    if str(os.getenv("FLAGDNN_GRAPH_FUSION", "1")).lower() in (
        "0",
        "false",
        "no",
    ):
        return False
    return bool(options.get("fusion", True))


def _dump_graph(graph_obj: Graph, stage: str) -> None:
    if str(os.getenv("FLAGDNN_DUMP_GRAPH", "0")).lower() in (
        "0",
        "false",
        "no",
    ):
        return
    dump_dir = os.getenv("FLAGDNN_DUMP_GRAPH_DIR", "/tmp/flagdnn_graphs")
    try:
        os.makedirs(dump_dir, exist_ok=True)
        graph_hash = graph_obj.graph_hash()[:12]
        json_path = os.path.join(dump_dir, f"{stage}_{graph_hash}.json")
        text_path = os.path.join(dump_dir, f"{stage}_{graph_hash}.txt")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(graph_obj.to_dict(), handle, indent=2, default=str)
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(graph_obj.dump_text())
    except OSError:
        return


def _flagdnn_version() -> str:
    try:
        import flag_dnn

        return str(getattr(flag_dnn, "__version__", "0.0.0"))
    except Exception:
        return "0.0.0"
