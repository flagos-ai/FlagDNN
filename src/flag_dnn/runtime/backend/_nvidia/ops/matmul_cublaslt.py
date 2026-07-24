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

import ctypes
import functools
import threading
from typing import Callable

import torch

_SUCCESS = 0
_CUDA_R_32F = 0
_CUDA_R_16BF = 14
_COMPUTE_32F = 68
_COMPUTE_32F_FAST_TF32 = 77
_LAYOUT_BATCH_COUNT = 5
_LAYOUT_STRIDED_BATCH_OFFSET = 6
_PREFERENCE_MAX_WORKSPACE_BYTES = 1
_MAX_WORKSPACE_BYTES = 32 * 1024 * 1024


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _HeuristicResult(ctypes.Structure):
    _fields_ = [
        ("algo", _Algo),
        ("workspace_size", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves_count", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


def _check(status: int, operation: str) -> None:
    if status != _SUCCESS:
        raise RuntimeError(f"{operation} failed with cuBLAS status {status}")


class _CublasLtApi:
    def __init__(self) -> None:
        load_error: OSError | None = None
        for library_name in ("libcublasLt.so.12", "libcublasLt.so"):
            try:
                self.lib = ctypes.CDLL(library_name)
                break
            except OSError as exc:
                load_error = exc
        else:
            assert load_error is not None
            raise load_error
        self._configure()

    def _configure(self) -> None:
        lib = self.lib
        void_pp = ctypes.POINTER(ctypes.c_void_p)
        lib.cublasLtCreate.argtypes = [void_pp]
        lib.cublasLtCreate.restype = ctypes.c_int
        lib.cublasLtDestroy.argtypes = [ctypes.c_void_p]
        lib.cublasLtDestroy.restype = ctypes.c_int
        lib.cublasLtMatmulDescCreate.argtypes = [
            void_pp,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.cublasLtMatmulDescCreate.restype = ctypes.c_int
        lib.cublasLtMatmulDescDestroy.argtypes = [ctypes.c_void_p]
        lib.cublasLtMatmulDescDestroy.restype = ctypes.c_int
        lib.cublasLtMatrixLayoutCreate.argtypes = [
            void_pp,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int64,
        ]
        lib.cublasLtMatrixLayoutCreate.restype = ctypes.c_int
        lib.cublasLtMatrixLayoutDestroy.argtypes = [ctypes.c_void_p]
        lib.cublasLtMatrixLayoutDestroy.restype = ctypes.c_int
        lib.cublasLtMatrixLayoutSetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.cublasLtMatrixLayoutSetAttribute.restype = ctypes.c_int
        lib.cublasLtMatmulPreferenceCreate.argtypes = [void_pp]
        lib.cublasLtMatmulPreferenceCreate.restype = ctypes.c_int
        lib.cublasLtMatmulPreferenceDestroy.argtypes = [ctypes.c_void_p]
        lib.cublasLtMatmulPreferenceDestroy.restype = ctypes.c_int
        lib.cublasLtMatmulPreferenceSetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.cublasLtMatmulPreferenceSetAttribute.restype = ctypes.c_int
        lib.cublasLtMatmulAlgoGetHeuristic.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_HeuristicResult),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.cublasLtMatmulAlgoGetHeuristic.restype = ctypes.c_int
        lib.cublasLtMatmul.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(_Algo),
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.cublasLtMatmul.restype = ctypes.c_int


_API: _CublasLtApi | None = None
_API_LOCK = threading.Lock()


def _get_api() -> _CublasLtApi:
    global _API
    if _API is None:
        with _API_LOCK:
            if _API is None:
                _API = _CublasLtApi()
    return _API


class _CublasLtPlan:
    def __init__(
        self,
        device_index: int,
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
    ) -> None:
        self.device_index = device_index
        self.dimensions = (batch, m, n, k)
        self.dtype = dtype
        if dtype == torch.float32:
            self.matrix_data_type = _CUDA_R_32F
            self.compute_type = _COMPUTE_32F_FAST_TF32
            self.mode_name = "TF32"
        elif dtype == torch.bfloat16:
            self.matrix_data_type = _CUDA_R_16BF
            self.compute_type = _COMPUTE_32F
            self.mode_name = "BF16"
        else:
            raise TypeError(f"cuBLASLt matmul does not support {dtype}")
        self.api = _get_api()
        self.handle = ctypes.c_void_p()
        self.operation = ctypes.c_void_p()
        self.layouts = [ctypes.c_void_p() for _ in range(4)]
        self.algo = _Algo()
        self.workspace_size = 0
        self.workspaces: dict[int, torch.Tensor] = {}
        self.alpha = ctypes.c_float(1.0)
        self.beta = ctypes.c_float(0.0)
        self.closed = False
        with torch.cuda.device(device_index):
            self._initialize()

    def _initialize(self) -> None:
        lib = self.api.lib
        preference = ctypes.c_void_p()
        _check(lib.cublasLtCreate(ctypes.byref(self.handle)), "cublasLtCreate")
        try:
            _check(
                lib.cublasLtMatmulDescCreate(
                    ctypes.byref(self.operation),
                    self.compute_type,
                    _CUDA_R_32F,
                ),
                "cublasLtMatmulDescCreate",
            )
            batch, m, n, k = self.dimensions
            # Row-major C=A@B is column-major C^T=B^T@A^T. This mapping
            # avoids layout-conversion kernels while using cuBLASLt's native
            # column-major descriptors.
            specs = (
                (n, k, n, k * n),
                (k, m, k, m * k),
                (n, m, n, m * n),
                (n, m, n, m * n),
            )
            for layout, (rows, cols, ld, stride) in zip(self.layouts, specs):
                _check(
                    lib.cublasLtMatrixLayoutCreate(
                        ctypes.byref(layout),
                        self.matrix_data_type,
                        rows,
                        cols,
                        ld,
                    ),
                    "cublasLtMatrixLayoutCreate",
                )
                batch_value = ctypes.c_int(batch)
                stride_value = ctypes.c_int64(stride)
                _check(
                    lib.cublasLtMatrixLayoutSetAttribute(
                        layout,
                        _LAYOUT_BATCH_COUNT,
                        ctypes.byref(batch_value),
                        ctypes.sizeof(batch_value),
                    ),
                    "cublasLtMatrixLayoutSetAttribute(batch)",
                )
                _check(
                    lib.cublasLtMatrixLayoutSetAttribute(
                        layout,
                        _LAYOUT_STRIDED_BATCH_OFFSET,
                        ctypes.byref(stride_value),
                        ctypes.sizeof(stride_value),
                    ),
                    "cublasLtMatrixLayoutSetAttribute(stride)",
                )
            _check(
                lib.cublasLtMatmulPreferenceCreate(ctypes.byref(preference)),
                "cublasLtMatmulPreferenceCreate",
            )
            max_workspace = ctypes.c_size_t(_MAX_WORKSPACE_BYTES)
            _check(
                lib.cublasLtMatmulPreferenceSetAttribute(
                    preference,
                    _PREFERENCE_MAX_WORKSPACE_BYTES,
                    ctypes.byref(max_workspace),
                    ctypes.sizeof(max_workspace),
                ),
                "cublasLtMatmulPreferenceSetAttribute",
            )
            results = (_HeuristicResult * 32)()
            returned = ctypes.c_int()
            _check(
                lib.cublasLtMatmulAlgoGetHeuristic(
                    self.handle,
                    self.operation,
                    self.layouts[0],
                    self.layouts[1],
                    self.layouts[2],
                    self.layouts[3],
                    preference,
                    len(results),
                    results,
                    ctypes.byref(returned),
                ),
                "cublasLtMatmulAlgoGetHeuristic",
            )
            selected = next(
                (
                    results[index]
                    for index in range(returned.value)
                    if results[index].state == _SUCCESS
                ),
                None,
            )
            if selected is None:
                raise RuntimeError(
                    f"cuBLASLt returned no {self.mode_name} algorithm"
                )
            ctypes.memmove(
                ctypes.byref(self.algo),
                ctypes.byref(selected.algo),
                ctypes.sizeof(self.algo),
            )
            self.workspace_size = int(selected.workspace_size)
            self._workspace_for_current_stream()
        except BaseException:
            self.close()
            raise
        finally:
            if preference:
                lib.cublasLtMatmulPreferenceDestroy(preference)

    def _workspace_for_current_stream(self) -> tuple[int, torch.Tensor | None]:
        stream_id = int(torch.cuda.current_stream().cuda_stream)
        if self.workspace_size == 0:
            return stream_id, None
        workspace = self.workspaces.get(stream_id)
        if workspace is None:
            workspace = torch.empty(
                self.workspace_size,
                device=torch.device("cuda", self.device_index),
                dtype=torch.uint8,
            )
            self.workspaces[stream_id] = workspace
        return stream_id, workspace

    def run(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        _validate_tensors(a, b, c, self.dimensions, self.dtype)
        with torch.cuda.device(self.device_index):
            stream_id, workspace = self._workspace_for_current_stream()
            workspace_ptr = (
                ctypes.c_void_p(workspace.data_ptr())
                if workspace is not None
                else None
            )
            _check(
                self.api.lib.cublasLtMatmul(
                    self.handle,
                    self.operation,
                    ctypes.byref(self.alpha),
                    ctypes.c_void_p(b.data_ptr()),
                    self.layouts[0],
                    ctypes.c_void_p(a.data_ptr()),
                    self.layouts[1],
                    ctypes.byref(self.beta),
                    ctypes.c_void_p(c.data_ptr()),
                    self.layouts[2],
                    ctypes.c_void_p(c.data_ptr()),
                    self.layouts[3],
                    ctypes.byref(self.algo),
                    workspace_ptr,
                    self.workspace_size,
                    ctypes.c_void_p(stream_id),
                ),
                "cublasLtMatmul",
            )
        return c

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        lib = self.api.lib
        for layout in self.layouts:
            if layout:
                lib.cublasLtMatrixLayoutDestroy(layout)
        if self.operation:
            lib.cublasLtMatmulDescDestroy(self.operation)
        if self.handle:
            lib.cublasLtDestroy(self.handle)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _validate_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    dimensions: tuple[int, int, int, int],
    dtype: torch.dtype,
) -> None:
    batch, m, n, k = dimensions
    device = a.device
    if (
        tuple(a.shape) != (batch, m, k)
        or tuple(b.shape) != (batch, k, n)
        or tuple(c.shape) != (batch, m, n)
    ):
        raise ValueError("cuBLASLt matmul received incompatible shapes")
    if a.dtype != dtype or b.dtype != dtype or c.dtype != dtype:
        raise TypeError(f"cuBLASLt matmul requires {dtype} tensors")
    if (
        device.type != "cuda"
        or b.device != device
        or c.device != device
        or not a.is_contiguous()
        or not b.is_contiguous()
        or not c.is_contiguous()
    ):
        raise ValueError("cuBLASLt matmul requires contiguous CUDA tensors")


@functools.lru_cache(maxsize=16)
def _get_plan(
    device_index: int,
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
) -> _CublasLtPlan:
    return _CublasLtPlan(device_index, batch, m, n, k, dtype)


def _plan_for(a: torch.Tensor, b: torch.Tensor) -> _CublasLtPlan:
    batch, m, k = map(int, a.shape)
    n = int(b.shape[2])
    device_index = a.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _get_plan(device_index, batch, m, n, k, a.dtype)


def run_cublaslt_tf32_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    return _plan_for(a, b).run(a, b, c)


def prepare_cublaslt_tf32_matmul_dynamic_output(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if output_dtype != torch.float32:
        raise TypeError("cuBLASLt TF32 matmul requires FP32 output")
    plan = _plan_for(a, b)

    def launch(output: torch.Tensor) -> torch.Tensor:
        return plan.run(a, b, output)

    return launch


def run_cublaslt_bf16_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    if a.dtype != torch.bfloat16:
        raise TypeError("cuBLASLt BF16 matmul requires BF16 input")
    return _plan_for(a, b).run(a, b, c)


def prepare_cublaslt_bf16_matmul_dynamic_output(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if a.dtype != torch.bfloat16 or output_dtype != torch.bfloat16:
        raise TypeError("cuBLASLt BF16 matmul requires BF16 tensors")
    plan = _plan_for(a, b)

    def launch(output: torch.Tensor) -> torch.Tensor:
        return plan.run(a, b, output)

    return launch


__all__ = (
    "prepare_cublaslt_bf16_matmul_dynamic_output",
    "prepare_cublaslt_tf32_matmul_dynamic_output",
    "run_cublaslt_bf16_matmul",
    "run_cublaslt_tf32_matmul",
)
