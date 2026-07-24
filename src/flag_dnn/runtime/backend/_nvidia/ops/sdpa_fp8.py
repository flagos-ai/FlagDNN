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

import math
from typing import Any, Callable, Optional, Sequence

import torch

from flag_dnn.graph.prepared import (
    RunFn,
    runtime_tensor_checks_from_specs,
    runtime_tensor_checks_pass,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _static_shape,
)
from flag_dnn.graph.tensor import TensorSpec, torch_dtype
from flag_dnn.runtime.backend._nvidia.ops.sdpa import _causal_from_attrs
from flag_dnn.utils.device_info import get_device_capability_for

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_FORWARD_KEYS = {
    (4, 16, 16, 512, 512, 128, False, False),
    (1, 32, 32, 1024, 1024, 128, True, True),
    (2, 16, 16, 2048, 2048, 128, True, True),
    (8, 32, 32, 256, 256, 128, False, False),
    (1, 32, 8, 4096, 4096, 128, True, True),
    (2, 32, 32, 1024, 1024, 128, False, False),
    (4, 32, 32, 512, 512, 128, True, False),
    (1, 64, 8, 2048, 2048, 128, True, True),
}
_FP8_BACKWARD_KEYS = {key[:-1] for key in _FP8_FORWARD_KEYS}


def use_cudnn_fp8_sdpa(
    shape: tuple[int, int, int, int, int, int],
    dtype: torch.dtype,
    *,
    causal: bool,
    generate_stats: bool,
) -> bool:
    return (
        dtype == _FP8_DTYPE
        and (
            *shape,
            causal,
            generate_stats,
        )
        in _FP8_FORWARD_KEYS
    )


def use_cudnn_fp8_sdpa_backward(
    shape: tuple[int, int, int, int, int, int],
    dtype: torch.dtype,
    *,
    causal: bool,
) -> bool:
    return dtype == _FP8_DTYPE and (*shape, causal) in _FP8_BACKWARD_KEYS


def _load_cudnn():
    try:
        import cudnn

        cudnn.backend_version()
    except (ImportError, OSError, RuntimeError):
        return None
    return cudnn


def _attention_shape(
    input_specs: Sequence[TensorSpec],
) -> Optional[tuple[int, int, int, int, int, int]]:
    if len(input_specs) < 3:
        return None
    q_shape = _static_shape(input_specs[0])
    k_shape = _static_shape(input_specs[1])
    v_shape = _static_shape(input_specs[2])
    if (
        q_shape is None
        or k_shape is None
        or v_shape is None
        or len(q_shape) != 4
        or len(k_shape) != 4
        or len(v_shape) != 4
    ):
        return None
    batch, heads, sq, head_dim = q_shape
    if (
        k_shape[0] != batch
        or v_shape[0] != batch
        or k_shape[1] != v_shape[1]
        or k_shape[2] != v_shape[2]
        or k_shape[3] != head_dim
        or v_shape[3] != head_dim
    ):
        return None
    return batch, heads, k_shape[1], sq, k_shape[2], head_dim


def _runtime_supported(
    inputs: Sequence[Any],
    checks: Sequence[Any],
    *,
    validate_inputs: bool,
) -> bool:
    if validate_inputs and not runtime_tensor_checks_pass(inputs, checks):
        return False
    tensors = [value for value in inputs if isinstance(value, torch.Tensor)]
    if len(tensors) != len(inputs) or not tensors:
        return False
    device = tensors[0].device
    return all(value.device == device for value in tensors) and (
        get_device_capability_for(device) == (9, 0)
    )


def _build_graph(cudnn, graph) -> None:
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()


class _CudnnPlan:
    def __init__(self, cudnn, device: torch.device) -> None:
        self.cudnn = cudnn
        self.device = device
        self.handle = cudnn.create_handle()
        self.activate_stream()

    def activate_stream(self) -> None:
        self.cudnn.set_stream(
            handle=self.handle,
            stream=torch.cuda.current_stream(self.device).cuda_stream,
        )

    def __del__(self) -> None:
        handle = getattr(self, "handle", None)
        if handle is None:
            return
        try:
            self.cudnn.destroy_handle(handle)
        except Exception:
            pass
        self.handle = None


class _CudnnFp8ForwardPlan(_CudnnPlan):
    def __init__(
        self,
        cudnn,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attrs: dict[str, Any],
        *,
        causal: bool,
        generate_stats: bool,
        scale: float,
    ) -> None:
        super().__init__(cudnn, q.device)
        self.generate_stats = generate_stats
        self.output_shape = (q.shape[0], q.shape[1], q.shape[2], v.shape[3])
        self.output_dtype = q.dtype
        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.FP8_E4M3,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.handle,
        )

        def fp8_tensor(value: torch.Tensor):
            return graph.tensor(
                dim=tuple(value.shape),
                stride=tuple(value.stride()),
                data_type=cudnn.data_type.FP8_E4M3,
            )

        def scalar_tensor():
            return graph.tensor(
                dim=(1, 1, 1, 1),
                stride=(1, 1, 1, 1),
                data_type=cudnn.data_type.FLOAT,
            )

        self.q_tensor = fp8_tensor(q)
        self.k_tensor = fp8_tensor(k)
        self.v_tensor = fp8_tensor(v)
        scalar_names = (
            "descale_q",
            "descale_k",
            "descale_v",
            "descale_s",
            "scale_s",
            "scale_o",
        )
        self.scalar_descriptors = {
            name: scalar_tensor() for name in scalar_names
        }
        output = graph.sdpa_fp8(
            q=self.q_tensor,
            k=self.k_tensor,
            v=self.v_tensor,
            descale_q=self.scalar_descriptors["descale_q"],
            descale_k=self.scalar_descriptors["descale_k"],
            descale_v=self.scalar_descriptors["descale_v"],
            descale_s=self.scalar_descriptors["descale_s"],
            scale_s=self.scalar_descriptors["scale_s"],
            scale_o=self.scalar_descriptors["scale_o"],
            generate_stats=generate_stats,
            attn_scale=scale,
            use_causal_mask=causal,
        )
        (
            self.o_tensor,
            self.stats_tensor,
            self.amax_s_tensor,
            self.amax_o_tensor,
        ) = output
        batch, heads, sq, dim = self.output_shape
        self.o_tensor.set_output(True).set_dim(self.output_shape).set_stride(
            (heads * sq * dim, sq * dim, dim, 1)
        ).set_data_type(cudnn.data_type.FP8_E4M3)
        for tensor in (self.amax_s_tensor, self.amax_o_tensor):
            tensor.set_output(True).set_dim((1, 1, 1, 1)).set_stride(
                (1, 1, 1, 1)
            ).set_data_type(cudnn.data_type.FLOAT)
        if generate_stats:
            self.stats_tensor.set_output(True).set_dim(
                (batch, heads, sq, 1)
            ).set_stride((heads * sq, sq, 1, 1)).set_data_type(
                cudnn.data_type.FLOAT
            )
        _build_graph(cudnn, graph)
        self.graph = graph
        self.workspace = torch.empty(
            graph.get_workspace_size(), device=q.device, dtype=torch.uint8
        )
        self.scalar_values = {
            descriptor: torch.tensor(
                [float(attrs[name])], device=q.device, dtype=torch.float32
            )
            for name, descriptor in self.scalar_descriptors.items()
        }

    def _make_pack(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[dict[Any, torch.Tensor], tuple[torch.Tensor, ...]]:
        output = torch.empty(
            self.output_shape, device=q.device, dtype=self.output_dtype
        )
        amax_s = torch.empty(
            (1, 1, 1, 1), device=q.device, dtype=torch.float32
        )
        amax_o = torch.empty_like(amax_s)
        pack = dict(self.scalar_values)
        pack.update(
            {
                self.q_tensor: q,
                self.k_tensor: k,
                self.v_tensor: v,
                self.o_tensor: output,
                self.amax_s_tensor: amax_s,
                self.amax_o_tensor: amax_o,
            }
        )
        if self.generate_stats:
            stats = torch.empty(
                (*self.output_shape[:3], 1),
                device=q.device,
                dtype=torch.float32,
            )
            pack[self.stats_tensor] = stats
        outputs: tuple[torch.Tensor, ...]
        if self.generate_stats:
            outputs = (output, stats, amax_s, amax_o)
        else:
            outputs = (output, amax_s, amax_o)
        return pack, outputs

    def _execute(
        self,
        pack: dict[Any, torch.Tensor],
        outputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        self.activate_stream()
        self.graph.execute(pack, self.workspace, handle=self.handle)
        return outputs

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        pack, outputs = self._make_pack(q, k, v)
        return self._execute(pack, outputs)

    def bind(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable[[], tuple[torch.Tensor, ...]]:
        pack, outputs = self._make_pack(q, k, v)
        return lambda: self._execute(pack, outputs)


class _CudnnFp8BackwardPlan(_CudnnPlan):
    def __init__(
        self,
        cudnn,
        inputs: Sequence[torch.Tensor],
        attrs: dict[str, Any],
        *,
        causal: bool,
        scale: float,
    ) -> None:
        q, k, v, o, grad_o, stats = inputs
        super().__init__(cudnn, q.device)
        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.FP8_E4M3,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.handle,
        )

        def fp8_tensor(value: torch.Tensor):
            return graph.tensor(
                dim=tuple(value.shape),
                stride=tuple(value.stride()),
                data_type=cudnn.data_type.FP8_E4M3,
            )

        def scalar_tensor():
            return graph.tensor(
                dim=(1, 1, 1, 1),
                stride=(1, 1, 1, 1),
                data_type=cudnn.data_type.FLOAT,
            )

        self.q_tensor = fp8_tensor(q)
        self.k_tensor = fp8_tensor(k)
        self.v_tensor = fp8_tensor(v)
        self.o_tensor = fp8_tensor(o)
        self.grad_o_tensor = fp8_tensor(grad_o)
        self.stats_tensor = graph.tensor_like(stats)
        scalar_names = (
            "descale_q",
            "descale_k",
            "descale_v",
            "descale_o",
            "descale_dO",
            "descale_s",
            "descale_dP",
            "scale_s",
            "scale_dQ",
            "scale_dK",
            "scale_dV",
            "scale_dP",
        )
        self.scalar_descriptors = {
            name: scalar_tensor() for name in scalar_names
        }
        result = graph.sdpa_fp8_backward(
            q=self.q_tensor,
            k=self.k_tensor,
            v=self.v_tensor,
            o=self.o_tensor,
            dO=self.grad_o_tensor,
            stats=self.stats_tensor,
            descale_q=self.scalar_descriptors["descale_q"],
            descale_k=self.scalar_descriptors["descale_k"],
            descale_v=self.scalar_descriptors["descale_v"],
            descale_o=self.scalar_descriptors["descale_o"],
            descale_dO=self.scalar_descriptors["descale_dO"],
            descale_s=self.scalar_descriptors["descale_s"],
            descale_dP=self.scalar_descriptors["descale_dP"],
            scale_s=self.scalar_descriptors["scale_s"],
            scale_dQ=self.scalar_descriptors["scale_dQ"],
            scale_dK=self.scalar_descriptors["scale_dK"],
            scale_dV=self.scalar_descriptors["scale_dV"],
            scale_dP=self.scalar_descriptors["scale_dP"],
            attn_scale=scale,
            use_causal_mask=causal,
        )
        (
            self.dq_tensor,
            self.dk_tensor,
            self.dv_tensor,
            self.amax_dq_tensor,
            self.amax_dk_tensor,
            self.amax_dv_tensor,
            self.amax_dp_tensor,
        ) = result
        for tensor, value in (
            (self.dq_tensor, q),
            (self.dk_tensor, k),
            (self.dv_tensor, v),
        ):
            tensor.set_output(True).set_dim(tuple(value.shape)).set_stride(
                tuple(value.stride())
            ).set_data_type(cudnn.data_type.FP8_E4M3)
        for tensor in (
            self.amax_dq_tensor,
            self.amax_dk_tensor,
            self.amax_dv_tensor,
            self.amax_dp_tensor,
        ):
            tensor.set_output(True).set_dim((1, 1, 1, 1)).set_stride(
                (1, 1, 1, 1)
            ).set_data_type(cudnn.data_type.FLOAT)
        _build_graph(cudnn, graph)
        self.graph = graph
        self.workspace = torch.empty(
            graph.get_workspace_size(), device=q.device, dtype=torch.uint8
        )
        self.scalar_values = {
            descriptor: torch.tensor(
                [float(attrs[name])], device=q.device, dtype=torch.float32
            )
            for name, descriptor in self.scalar_descriptors.items()
        }

    def _make_pack(
        self,
        inputs: Sequence[torch.Tensor],
    ) -> tuple[dict[Any, torch.Tensor], tuple[torch.Tensor, ...]]:
        q, k, v, o, grad_o, stats = inputs
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        amax = [
            torch.empty((1, 1, 1, 1), device=q.device, dtype=torch.float32)
            for _ in range(4)
        ]
        pack = dict(self.scalar_values)
        pack.update(
            {
                self.q_tensor: q,
                self.k_tensor: k,
                self.v_tensor: v,
                self.o_tensor: o,
                self.grad_o_tensor: grad_o,
                self.stats_tensor: stats,
                self.dq_tensor: dq,
                self.dk_tensor: dk,
                self.dv_tensor: dv,
                self.amax_dq_tensor: amax[0],
                self.amax_dk_tensor: amax[1],
                self.amax_dv_tensor: amax[2],
                self.amax_dp_tensor: amax[3],
            }
        )
        return pack, (dq, dk, dv, *amax)

    def _execute(
        self,
        pack: dict[Any, torch.Tensor],
        outputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        self.activate_stream()
        self.graph.execute(pack, self.workspace, handle=self.handle)
        return outputs

    def run(
        self,
        inputs: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        pack, outputs = self._make_pack(inputs)
        return self._execute(pack, outputs)

    def bind(
        self,
        inputs: Sequence[torch.Tensor],
    ) -> Callable[[], tuple[torch.Tensor, ...]]:
        pack, outputs = self._make_pack(inputs)
        return lambda: self._execute(pack, outputs)


def _specs_are_fp8(
    input_specs: Sequence[TensorSpec],
    indices: Sequence[int],
) -> bool:
    return all(
        _is_runtime_device_spec(input_specs[index])
        and torch_dtype(input_specs[index].dtype) == _FP8_DTYPE
        and bool(input_specs[index].contiguous)
        for index in indices
    )


def prepare_sdpa_fp8(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 3 or attrs.get("has_bias"):
        return None
    shape = _attention_shape(input_specs)
    causal = _causal_from_attrs(attrs)
    generate_stats = bool(attrs.get("generate_stats"))
    if (
        shape is None
        or causal is None
        or not _specs_are_fp8(input_specs, (0, 1, 2))
        or not use_cudnn_fp8_sdpa(
            shape,
            torch_dtype(input_specs[0].dtype),
            causal=causal,
            generate_stats=generate_stats,
        )
    ):
        return None
    cudnn = _load_cudnn()
    if cudnn is None:
        return None
    checks = runtime_tensor_checks_from_specs(
        input_specs, (0, 1, 2), require_dtype=True
    )
    if checks is None:
        return None
    scale = attrs.get("attn_scale")
    if scale is None:
        scale = 1.0 / math.sqrt(shape[-1])
    scale = float(scale)
    validate_inputs = bool(attrs.get("_validate_inputs", True))
    plan: Optional[_CudnnFp8ForwardPlan] = None

    def get_plan(inputs: Sequence[Any]) -> _CudnnFp8ForwardPlan:
        nonlocal plan
        if plan is None:
            plan = _CudnnFp8ForwardPlan(
                cudnn,
                inputs[0],
                inputs[1],
                inputs[2],
                attrs,
                causal=causal,
                generate_stats=generate_stats,
                scale=scale,
            )
        return plan

    def execute(inputs: Sequence[Any]):
        return get_plan(inputs).run(inputs[0], inputs[1], inputs[2])

    def can_run(inputs: Sequence[Any]) -> bool:
        return _runtime_supported(
            inputs, checks, validate_inputs=validate_inputs
        )

    def bind(
        inputs: Sequence[Any], run_attrs: dict[str, Any]
    ) -> Callable[[], Any]:
        if not can_run(inputs):
            return lambda: default_run_fn(inputs, run_attrs)
        return get_plan(inputs).bind(inputs[0], inputs[1], inputs[2])

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if not can_run(inputs):
            return default_run_fn(inputs, run_attrs)
        return execute(inputs)

    setattr(run, "bind", bind)
    setattr(run, "_flagdnn_functional_output_safe", True)
    return run


def prepare_sdpa_fp8_backward(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 6 or attrs.get("use_deterministic_algorithm"):
        return None
    shape = _attention_shape(input_specs)
    causal = _causal_from_attrs(attrs)
    q_shape = _static_shape(input_specs[0])
    if (
        shape is None
        or causal is None
        or q_shape is None
        or not _specs_are_fp8(input_specs, range(5))
        or torch_dtype(input_specs[5].dtype) != torch.float32
        or not bool(input_specs[5].contiguous)
        or _static_shape(input_specs[3]) != q_shape
        or _static_shape(input_specs[4]) != q_shape
        or _static_shape(input_specs[5]) != (*q_shape[:3], 1)
        or not use_cudnn_fp8_sdpa_backward(
            shape,
            torch_dtype(input_specs[0].dtype),
            causal=causal,
        )
    ):
        return None
    cudnn = _load_cudnn()
    if cudnn is None:
        return None
    checks = runtime_tensor_checks_from_specs(
        input_specs, tuple(range(6)), require_dtype=True
    )
    if checks is None:
        return None
    scale = attrs.get("attn_scale")
    if scale is None:
        scale = 1.0 / math.sqrt(shape[-1])
    scale = float(scale)
    validate_inputs = bool(attrs.get("_validate_inputs", True))
    plan: Optional[_CudnnFp8BackwardPlan] = None

    def get_plan(inputs: Sequence[Any]) -> _CudnnFp8BackwardPlan:
        nonlocal plan
        tensor_inputs = tuple(inputs[:6])
        if plan is None:
            plan = _CudnnFp8BackwardPlan(
                cudnn,
                tensor_inputs,
                attrs,
                causal=causal,
                scale=scale,
            )
        return plan

    def execute(inputs: Sequence[Any]):
        return get_plan(inputs).run(tuple(inputs[:6]))

    def can_run(inputs: Sequence[Any]) -> bool:
        return _runtime_supported(
            inputs, checks, validate_inputs=validate_inputs
        )

    def bind(
        inputs: Sequence[Any], run_attrs: dict[str, Any]
    ) -> Callable[[], Any]:
        if not can_run(inputs):
            return lambda: default_run_fn(inputs, run_attrs)
        return get_plan(inputs).bind(tuple(inputs[:6]))

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if not can_run(inputs):
            return default_run_fn(inputs, run_attrs)
        return execute(inputs)

    setattr(run, "bind", bind)
    setattr(run, "_flagdnn_functional_output_safe", True)
    return run


__all__ = (
    "prepare_sdpa_fp8",
    "prepare_sdpa_fp8_backward",
    "use_cudnn_fp8_sdpa",
    "use_cudnn_fp8_sdpa_backward",
)
