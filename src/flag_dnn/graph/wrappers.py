# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
from typing import Any, Callable, Optional

import torch

from flag_dnn.graph.capture import current_capture, is_capturing
from flag_dnn.graph.registry.core import graph_wrapper_specs
from flag_dnn.graph.tensor import GraphTensor

# Names actually wrapped by the most recent install_graph_wrappers call.
# Populated lazily (the registry is the source of truth); kept for
# introspection and tests.
GRAPH_AWARE_OPS: tuple[str, ...] = ()


def eager_bias_add(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if bias.dim() == 1 and input.dim() >= 2:
        shape = [1] * input.dim()
        shape[1] = bias.numel()
        bias = bias.reshape(shape)
    from flag_dnn.ops.add import add

    return add(input, bias)


# Graph-only ops with no eager namespace function get their eager fallback
# from here when installing wrappers.
_SPECIAL_EAGER_FNS: dict[str, Callable[..., Any]] = {
    "bias_add": eager_bias_add,
}


def install_graph_wrappers(namespace: dict[str, Any]) -> None:
    """Install capture wrappers for every graph-aware op in ``namespace``.

    The set of ops and their dict-output keys come straight from the registry
    (``graph_wrapper_specs``), so there is no second list to keep in sync.
    """
    from flag_dnn.graph.registry import register_default_ops

    register_default_ops()
    global GRAPH_AWARE_OPS
    GRAPH_AWARE_OPS = tuple(graph_wrapper_specs())

    for op_type, spec in graph_wrapper_specs().items():
        eager_fn = namespace.get(spec.eager_name)
        if eager_fn is None:
            eager_fn = _SPECIAL_EAGER_FNS.get(op_type)
        if eager_fn is None:
            continue
        if getattr(eager_fn, "__flagdnn_graph_wrapped__", False):
            continue
        namespace[spec.eager_name] = make_graph_wrapper(
            op_type, eager_fn, spec.output_keys
        )


def make_graph_wrapper(
    op_type: str,
    eager_fn: Callable[..., Any],
    output_keys: Optional[tuple[str, ...]] = None,
):
    @functools.wraps(eager_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if (
            is_capturing()
            or _contains_graph_tensor(args)
            or _contains_graph_tensor(kwargs)
        ):
            ctx = current_capture()
            if ctx is None:
                raise RuntimeError(
                    f"FlagDNN graph op {op_type} used outside graph capture"
                )
            outputs = ctx.add_op_call(op_type, args, kwargs)
            if output_keys is not None:
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                if len(outputs) != len(output_keys):
                    raise RuntimeError(
                        f"FlagDNN graph op {op_type} returned "
                        f"{len(outputs)} outputs, expected {len(output_keys)}"
                    )
                return dict(zip(output_keys, outputs))
            return outputs
        return eager_fn(*args, **kwargs)

    setattr(wrapper, "__flagdnn_graph_wrapped__", True)
    setattr(wrapper, "__flagdnn_eager_fn__", eager_fn)
    return wrapper


def _contains_graph_tensor(value: Any) -> bool:
    if isinstance(value, GraphTensor):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_graph_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_graph_tensor(item) for item in value.values())
    return False
