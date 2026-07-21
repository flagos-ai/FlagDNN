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
from typing import Any, Iterable

import torch

from ..common import (
    DTYPE_CODES,
    ERROR_BUFFER_SIZE,
    INT64_POINTER,
    AscendContext,
    PreparedAclnnOperation,
    metadata_array,
)


REDUCTION_OPERATION_CODES = {
    "ADD": 0,
    "AVG": 1,
    "MUL": 2,
}
_MODE_ALIASES = {"SUM": "ADD", "MEAN": "AVG", "PROD": "MUL"}


def configure_reduction(library: Any) -> None:
    create = library.flagdnn_aclnn_reduction_create
    create.argtypes = [
        ctypes.c_int32,
        ctypes.c_void_p,
        INT64_POINTER,
        INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_void_p,
        INT64_POINTER,
        INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_int32,
        INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    create.restype = ctypes.c_int
    for name in (
        "flagdnn_aclnn_reduction_run",
        "flagdnn_aclnn_reduction_destroy",
    ):
        function = getattr(library, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int


def _mode_name(mode: Any) -> str:
    name = getattr(mode, "name", None)
    if name is None:
        name = str(mode).rsplit(".", 1)[-1]
    normalized = str(name).upper()
    return _MODE_ALIASES.get(normalized, normalized)


def _dimensions(dim: int | Iterable[int] | None, rank: int) -> tuple[int, ...]:
    if dim is None:
        values = tuple(range(rank))
    elif isinstance(dim, int):
        values = (dim,)
    else:
        values = tuple(int(item) for item in dim)
    normalized = []
    for value in values:
        if value < 0:
            value += rank
        if value < 0 or value >= rank:
            raise IndexError(f"reduction dimension out of range: {value}")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("ACLNN reduction requires at least one dimension")
    return tuple(sorted(normalized))


class AscendReductionOperation:
    name = "reduction"

    def __init__(self, context: AscendContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in DTYPE_CODES

    def run(
        self,
        x: torch.Tensor,
        mode: Any,
        *,
        dim: int | Iterable[int] | None = None,
        keepdim: bool = True,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        prepared = self.prepare(
            x,
            mode,
            dim=dim,
            keepdim=keepdim,
            dtype=dtype,
            **kwargs,
        )
        try:
            return prepared.run()
        finally:
            prepared.close()

    def prepare(
        self,
        x: torch.Tensor,
        mode: Any,
        *,
        dim: int | Iterable[int] | None = None,
        keepdim: bool = True,
        dtype: torch.dtype | None = None,
        **_: Any,
    ) -> PreparedAclnnOperation:
        if not isinstance(x, torch.Tensor) or x.device.type != "npu":
            raise TypeError("aclnn reduction requires an NPU tensor")
        if x.layout != torch.strided or x.storage_offset() != 0:
            raise ValueError(
                "aclnn reduction requires a zero-offset strided tensor"
            )
        if x.dim() == 0 or x.dim() > 8 or x.numel() == 0:
            raise ValueError(
                "aclnn reduction requires a non-empty tensor of rank 1..8"
            )
        if not self.supports_dtype(x.dtype):
            raise TypeError(f"aclnn reduction does not support {x.dtype}")
        if dtype not in (None, x.dtype):
            raise TypeError(
                "aclnn reduction reference currently requires output dtype "
                "to match input dtype"
            )
        mode_name = _mode_name(mode)
        if mode_name not in REDUCTION_OPERATION_CODES:
            raise ValueError(
                f"aclnn reduction mode is not implemented: {mode_name}"
            )
        dimensions = _dimensions(dim, x.dim())
        if mode_name == "MUL" and len(dimensions) != 1:
            raise ValueError("aclnnProdDim supports one reduction dimension")
        output_shape = []
        for axis, size in enumerate(x.shape):
            if axis in dimensions:
                if keepdim:
                    output_shape.append(1)
            else:
                output_shape.append(int(size))
        if not output_shape:
            raise ValueError(
                "rank-0 ACLNN reduction outputs are not supported by this "
                "reference wrapper; use keepdim=True"
            )

        context = self._context
        npu = context.npu()
        with npu.device(x.device):
            output = torch.empty(
                tuple(output_shape), device=x.device, dtype=x.dtype
            )
            input_shape = metadata_array(x.shape)
            input_stride = metadata_array(x.stride())
            output_shape_array = metadata_array(output.shape)
            output_stride = metadata_array(output.stride())
            dimension_array = metadata_array(dimensions)
            stream_pointer = int(
                npu.current_stream(device=x.device).npu_stream
            )
            handle = ctypes.c_void_p()
            error_buffer = ctypes.create_string_buffer(ERROR_BUFFER_SIZE)
            library = context.get_library(configure_reduction)
            status = library.flagdnn_aclnn_reduction_create(
                ctypes.c_int32(REDUCTION_OPERATION_CODES[mode_name]),
                ctypes.c_void_p(int(x.data_ptr())),
                input_shape,
                input_stride,
                ctypes.c_uint64(x.dim()),
                ctypes.c_void_p(int(output.data_ptr())),
                output_shape_array,
                output_stride,
                ctypes.c_uint64(output.dim()),
                ctypes.c_int32(DTYPE_CODES[x.dtype]),
                dimension_array,
                ctypes.c_uint64(len(dimensions)),
                ctypes.c_bool(keepdim),
                ctypes.c_void_p(stream_pointer),
                ctypes.byref(handle),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
        if status != 0 or handle.value is None:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                "prepared aclnn reduction creation failed: "
                f"status={status}, detail={detail}, mode={mode_name}, "
                f"shape={tuple(x.shape)}, dim={dimensions}, "
                f"dtype={x.dtype}, device={x.device}"
            )
        context.last_device = x.device
        return PreparedAclnnOperation(
            library,
            handle,
            (x,),
            output,
            operation_name=f"aclnnReduction{mode_name}",
            symbol_name="aclnn_reduction",
        )
