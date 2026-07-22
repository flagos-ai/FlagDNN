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

from typing import Any

import torch

from .common import (
    NvidiaContext,
    PreparedCudnnOperation,
    build_cudnn_graph,
    cudnn,
    cudnn_graph,
)


_HADAMARD_SIZE = 16
_SM100_MAJOR = 10
_DEFAULT_NUM_THREADS_BY_N = {
    2048: 128,
    4096: 256,
    7168: 128,
    8192: 512,
    16384: 1024,
    32768: 512,
}
_RPC_CANDIDATES = (2, 4, 8)
_TARGET_MIN_CTAS = 148


def _best_num_threads(n: int) -> int | None:
    for num_threads in (1024, 512, 256, 128, 64):
        if n % num_threads != 0:
            continue
        elements_per_thread = n // num_threads
        if elements_per_thread >= 8 and elements_per_thread % 8 == 0:
            return num_threads
    return None


def _pick_rows_per_cta(m: int) -> int:
    for rows_per_cta in reversed(_RPC_CANDIDATES):
        if m % rows_per_cta != 0:
            continue
        if m // rows_per_cta >= _TARGET_MIN_CTAS:
            return rows_per_cta
    return _RPC_CANDIDATES[0]


def _hadamard_16(device: torch.device) -> torch.Tensor:
    value = torch.ones((1, 1), dtype=torch.float32)
    while value.shape[0] < _HADAMARD_SIZE:
        value = torch.cat(
            (
                torch.cat((value, value), dim=1),
                torch.cat((value, -value), dim=1),
            ),
            dim=0,
        )
    return (value * 0.25).to(device=device).reshape(1, 16, 16)


class _PreparedComposite:
    reference_name = "cuDNN standard composite"

    def __init__(
        self,
        operations: tuple[PreparedCudnnOperation, ...],
        output: dict[str, torch.Tensor],
    ) -> None:
        self._operations = operations
        self.output = output
        self._closed = False

    def run(self) -> dict[str, torch.Tensor]:
        if self._closed:
            raise RuntimeError("prepared cuDNN operation is closed")
        for operation in self._operations:
            operation.run()
        return self.output

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for operation in reversed(self._operations):
            operation.close()


class _PreparedNative:
    reference_name = "cuDNN native SM100"

    def __init__(
        self,
        wrapper: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        rows_per_cta: int,
        num_threads: int,
    ) -> None:
        self._wrapper = wrapper
        self._x = x
        self._weight = weight
        self._eps = eps
        self._rows_per_cta = rows_per_cta
        self._num_threads = num_threads
        self.output: dict[str, torch.Tensor] | None = None
        self._closed = False

    def run(self) -> dict[str, torch.Tensor]:
        if self._closed:
            raise RuntimeError("prepared cuDNN operation is closed")
        self.output = self._wrapper(
            self._x,
            self._weight,
            eps=self._eps,
            rows_per_cta=self._rows_per_cta,
            num_threads=self._num_threads,
        )
        return self.output

    def close(self) -> None:
        self._closed = True


class NvidiaRmsNormRhtAmaxOperation:
    name = "rmsnorm_rht_amax_wrapper_sm100"

    def __init__(self, context: NvidiaContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype == torch.bfloat16

    def run(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        prepared = self.prepare(*args, **kwargs)
        try:
            output = prepared.run()
            self._context.synchronize()
            return output
        finally:
            prepared.close()

    def prepare(
        self,
        x_tensor: torch.Tensor,
        w_tensor: torch.Tensor,
        eps: float = 1e-5,
        num_threads: int | None = None,
        rows_per_cta: int | None = None,
        current_stream: Any = None,
        **_: Any,
    ) -> Any:
        del current_stream
        x, weight, rows, threads = self._validate(
            x_tensor, w_tensor, rows_per_cta, num_threads
        )
        if torch.cuda.get_device_capability(x.device)[0] >= _SM100_MAJOR:
            try:
                wrapper = getattr(cudnn, self.name)
                native = _PreparedNative(
                    wrapper,
                    x,
                    weight,
                    float(eps),
                    rows,
                    threads,
                )
                native.run()
            except (
                AttributeError,
                ImportError,
                AssertionError,
                ValueError,
                TypeError,
            ):
                if "native" in locals():
                    native.close()
            else:
                torch.cuda.synchronize(x.device)
                self._context.last_device = x.device
                return native
        return self._prepare_composite(x, weight, float(eps), rows)

    def _validate(
        self,
        x_tensor: torch.Tensor,
        w_tensor: torch.Tensor,
        rows_per_cta: int | None,
        num_threads: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        x = (
            x_tensor.squeeze(-1)
            if x_tensor.dim() == 3 and x_tensor.shape[-1] == 1
            else x_tensor
        )
        weight = (
            w_tensor.squeeze(-1)
            if w_tensor.dim() == 2 and w_tensor.shape[-1] == 1
            else w_tensor
        )
        self._context.validate_tensor(self.name, x)
        self._context.validate_tensor(self.name, weight)
        if x.dim() != 2 or weight.dim() != 1:
            raise ValueError("RMSNorm RHT amax expects x 2D and weight 1D")
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError("RMSNorm RHT amax requires bfloat16 inputs")
        if x.device != weight.device:
            raise ValueError("RMSNorm RHT amax inputs must share a device")
        m, n = (int(value) for value in x.shape)
        if not x.is_contiguous() or not weight.is_contiguous():
            raise ValueError("RMSNorm RHT amax inputs must be contiguous")
        if tuple(weight.shape) != (n,):
            raise ValueError("weight length must match x.shape[1]")
        if n % _HADAMARD_SIZE:
            raise ValueError("x.shape[1] must be divisible by 16")
        rows = (
            _pick_rows_per_cta(m)
            if rows_per_cta is None
            else int(rows_per_cta)
        )
        if rows <= 0 or m % rows:
            raise ValueError("rows_per_cta must be positive and divide M")
        threads = (
            _DEFAULT_NUM_THREADS_BY_N.get(n, _best_num_threads(n))
            if num_threads is None
            else int(num_threads)
        )
        if threads is None:
            raise ValueError(f"No valid num_threads found for N={n}")
        if threads <= 0 or threads % 32 or threads > 1024:
            raise ValueError(f"invalid num_threads={threads}")
        if n % threads:
            raise ValueError(
                f"N={n} must be divisible by num_threads={threads}"
            )
        elements_per_thread = n // threads
        if elements_per_thread < 8 or elements_per_thread % 8:
            raise ValueError(
                f"EPT={elements_per_thread} must be >= 8 and divisible by 8"
            )
        return x, weight, rows, threads

    def _prepare_composite(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        rows_per_cta: int,
    ) -> _PreparedComposite:
        context = self._context
        m, n = (int(value) for value in x.shape)
        scale = weight.reshape(1, n)
        epsilon = torch.full((1, 1), eps, dtype=torch.float32, device="cpu")
        hadamard = _hadamard_16(x.device)

        with torch.cuda.device(x.device):
            context.activate_stream(x.device)
            norm_graph = cudnn_graph(x.dtype, context.handle)
            x_value = norm_graph.tensor_like(x)
            scale_value = norm_graph.tensor_like(scale)
            epsilon_value = norm_graph.tensor(
                dim=(1, 1),
                stride=(1, 1),
                data_type=cudnn.data_type.FLOAT,
                is_pass_by_value=True,
                name="rmsnorm_rht_epsilon",
            )
            norm_values = norm_graph.rmsnorm(
                cudnn.norm_forward_phase.INFERENCE,
                x_value,
                scale_value,
                bias=None,
                epsilon=epsilon_value,
                compute_data_type=cudnn.data_type.FLOAT,
                name="rmsnorm_rht_rmsnorm",
            )
            norm_value = (
                norm_values[0]
                if isinstance(norm_values, (tuple, list))
                else norm_values
            )
            norm_output = torch.empty(
                (m, n), device=x.device, dtype=torch.float32
            )
            norm_value.set_output(True).set_data_type(
                cudnn.data_type.FLOAT
            ).set_dim(list(norm_output.shape)).set_stride(
                list(norm_output.stride())
            )
            build_cudnn_graph(norm_graph, self.name)
            norm_workspace = torch.empty(
                norm_graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            norm = PreparedCudnnOperation(
                norm_graph,
                {
                    x_value: x,
                    scale_value: scale,
                    epsilon_value: epsilon,
                    norm_value: norm_output,
                },
                norm_workspace,
                norm_output,
                context.handle,
            )

            blocks = norm_output.reshape(1, m * n // 16, 16)
            rht_graph = cudnn_graph(torch.float32, context.handle)
            blocks_value = rht_graph.tensor_like(blocks)
            hadamard_value = rht_graph.tensor_like(hadamard)
            rht_value = rht_graph.matmul(
                A=blocks_value,
                B=hadamard_value,
                compute_data_type=cudnn.data_type.FLOAT,
                name="rmsnorm_rht_hadamard",
            )
            rht_output = torch.empty_like(blocks)
            rht_value.set_output(True).set_data_type(
                cudnn.data_type.FLOAT
            ).set_dim(list(rht_output.shape)).set_stride(
                list(rht_output.stride())
            )
            build_cudnn_graph(rht_graph, self.name)
            rht_workspace = torch.empty(
                rht_graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            rht = PreparedCudnnOperation(
                rht_graph,
                {
                    blocks_value: blocks,
                    hadamard_value: hadamard,
                    rht_value: rht_output,
                },
                rht_workspace,
                rht_output,
                context.handle,
            )

            rht_flat = rht_output.reshape(m, n)
            cast_graph = cudnn_graph(torch.float32, context.handle)
            cast_input = cast_graph.tensor_like(rht_output)
            cast_value = cast_graph.identity(
                input=cast_input,
                compute_data_type=cudnn.data_type.FLOAT,
                name="rmsnorm_rht_output_cast",
            )
            cast_output_blocks = torch.empty(
                rht_output.shape, device=x.device, dtype=torch.bfloat16
            )
            cast_output = cast_output_blocks.reshape(m, n)
            cast_value.set_output(True).set_data_type(
                cudnn.data_type.BFLOAT16
            ).set_dim(list(cast_output_blocks.shape)).set_stride(
                list(cast_output_blocks.stride())
            )
            build_cudnn_graph(cast_graph, self.name)
            cast_workspace = torch.empty(
                cast_graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            cast = PreparedCudnnOperation(
                cast_graph,
                {cast_input: rht_output, cast_value: cast_output_blocks},
                cast_workspace,
                cast_output,
                context.handle,
            )

            row_graph = cudnn_graph(torch.float32, context.handle)
            row_input = row_graph.tensor(
                dim=(m, n, 1, 1),
                stride=(n, 1, n, n),
                data_type=cudnn.data_type.FLOAT,
                name="rmsnorm_rht_rows",
            )
            absolute = row_graph.abs(
                input=row_input,
                compute_data_type=cudnn.data_type.FLOAT,
                name="rmsnorm_rht_abs",
            )
            row_value = row_graph.reduction(
                input=absolute,
                mode=cudnn.reduction_mode.MAX,
                compute_data_type=cudnn.data_type.FLOAT,
                name="rmsnorm_rht_row_amax",
            )
            row_output = torch.empty(
                (m,), device=x.device, dtype=torch.float32
            )
            row_value.set_output(True).set_data_type(
                cudnn.data_type.FLOAT
            ).set_dim([m, 1, 1, 1]).set_stride([1, 1, 1, 1])
            build_cudnn_graph(row_graph, self.name)
            row_workspace = torch.empty(
                row_graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            row_amax = PreparedCudnnOperation(
                row_graph,
                {row_input: rht_flat, row_value: row_output},
                row_workspace,
                row_output,
                context.handle,
            )

            cta_count = m // rows_per_cta
            cta_graph = cudnn_graph(torch.float32, context.handle)
            cta_input = cta_graph.tensor(
                dim=(cta_count, rows_per_cta, 1, 1),
                stride=(
                    rows_per_cta,
                    1,
                    rows_per_cta,
                    rows_per_cta,
                ),
                data_type=cudnn.data_type.FLOAT,
                name="rmsnorm_rht_cta_rows",
            )
            cta_value = cta_graph.reduction(
                input=cta_input,
                mode=cudnn.reduction_mode.MAX,
                compute_data_type=cudnn.data_type.FLOAT,
                name="rmsnorm_rht_cta_amax",
            )
            cta_output = torch.empty(
                (cta_count,), device=x.device, dtype=torch.float32
            )
            cta_value.set_output(True).set_data_type(
                cudnn.data_type.FLOAT
            ).set_dim([cta_count, 1, 1, 1]).set_stride([1, 1, 1, 1])
            build_cudnn_graph(cta_graph, self.name)
            cta_workspace = torch.empty(
                cta_graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            cta_amax = PreparedCudnnOperation(
                cta_graph,
                {cta_input: row_output, cta_value: cta_output},
                cta_workspace,
                cta_output,
                context.handle,
            )

        context.last_device = x.device
        output = {"o_tensor": cast_output, "amax_tensor": cta_output}
        return _PreparedComposite(
            (norm, rht, cast, row_amax, cta_amax), output
        )
