import sys
import types

from flag_dnn.graph.cache import (
    PlanCache,
    PlanCacheKey,
    get_default_plan_cache,
)
from flag_dnn.graph.autotune import AutotuneResult, GraphAutotuner
from flag_dnn.graph.backend import (
    AutoBackend,
    BackendCapability,
    GraphBackend,
    KernelCandidate,
    TritonAscendBackend,
    TritonCudaBackend,
    resolve_backend,
)
from flag_dnn.graph.capture import (
    CompiledGraph,
    GraphCapture,
    GraphFunction,
    compile,
    current_capture,
    graph,
    is_capturing,
)
from flag_dnn.graph.graph import Graph
from flag_dnn.graph.kernel_selector import KernelSelector
from flag_dnn.graph.memory import BufferBlock, MemoryPlan, TensorAllocation
from flag_dnn.graph.node import GraphValue, OpNode
from flag_dnn.graph.plan import ExecutionPlan, ExecutionStep
from flag_dnn.graph.registry import OpSchema, get_op_schema, register_op
from flag_dnn.graph.tensor import GraphTensor, TensorSpec
from flag_dnn.graph.wrappers import eager_bias_add, install_graph_wrappers


class _CallableGraphModule(types.ModuleType):
    def __call__(self, fn=None, **options):
        return graph(fn, **options)


sys.modules[__name__].__class__ = _CallableGraphModule

__all__ = [
    "AutoBackend",
    "AutotuneResult",
    "BackendCapability",
    "BufferBlock",
    "CompiledGraph",
    "ExecutionPlan",
    "ExecutionStep",
    "Graph",
    "GraphCapture",
    "GraphAutotuner",
    "GraphBackend",
    "GraphFunction",
    "GraphTensor",
    "GraphValue",
    "KernelCandidate",
    "KernelSelector",
    "MemoryPlan",
    "OpNode",
    "OpSchema",
    "PlanCache",
    "PlanCacheKey",
    "TensorAllocation",
    "TensorSpec",
    "TritonAscendBackend",
    "TritonCudaBackend",
    "compile",
    "current_capture",
    "eager_bias_add",
    "get_default_plan_cache",
    "get_op_schema",
    "graph",
    "install_graph_wrappers",
    "is_capturing",
    "register_op",
    "resolve_backend",
]
