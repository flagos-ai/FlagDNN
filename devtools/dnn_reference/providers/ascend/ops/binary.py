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
from typing import Any, Union

import torch

from ..common import (
    DTYPE_CODES,
    ERROR_BUFFER_SIZE,
    INT64_POINTER,
    AscendContext,
    PreparedAclnnOperation,
    metadata_array,
)


Number = Union[int, float]

BINARY_OPERATION_CODES = {
    "sub": 0,
    "mul": 1,
    "div": 2,
    "pow": 3,
    "max": 4,
    "min": 5,
    "mod": 6,
    "add_square": 7,
    "scale": 1,
    "cmp_eq": 8,
    "cmp_neq": 9,
    "cmp_gt": 10,
    "cmp_ge": 11,
    "cmp_lt": 12,
    "cmp_le": 13,
    "logical_and": 14,
    "logical_or": 15,
    "_sigmoid_backward_output": 16,
}

_COMPARISON_OPERATIONS = {
    "cmp_eq",
    "cmp_neq",
    "cmp_gt",
    "cmp_ge",
    "cmp_lt",
    "cmp_le",
}
_LOGICAL_OPERATIONS = {"logical_and", "logical_or"}
_BINARY_DTYPE_CODES = {**DTYPE_CODES, torch.bool: 3}


def configure_binary(library: Any) -> None:
    create = library.flagdnn_aclnn_binary_create
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
        ctypes.c_void_p,
        INT64_POINTER,
        INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    create.restype = ctypes.c_int
    for name in (
        "flagdnn_aclnn_binary_run",
        "flagdnn_aclnn_binary_destroy",
    ):
        function = getattr(library, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int


def _normalize_alpha(alpha: Number) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(
            "ACLNN binary reference alpha must be an int or float, "
            f"got {type(alpha).__name__}"
        )
    return float(alpha)


class AscendBinaryOperation:
    def __init__(self, name: str, context: AscendContext) -> None:
        if name not in BINARY_OPERATION_CODES:
            raise ValueError(f"unsupported generic ACLNN binary op: {name}")
        self.name = name
        self._context = context
        self._operation_code = BINARY_OPERATION_CODES[name]

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        if self.name in _LOGICAL_OPERATIONS:
            return dtype == torch.bool
        return dtype in DTYPE_CODES

    def _validate(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError(
                f"aclnn {self.name} reference expects two tensor inputs"
            )
        if x.layout != torch.strided or y.layout != torch.strided:
            raise ValueError(
                f"aclnn {self.name} reference requires strided tensors"
            )
        if x.device.type != "npu" or y.device.type != "npu":
            raise ValueError(
                f"aclnn {self.name} reference requires NPU tensors"
            )
        if x.device != y.device:
            raise ValueError(f"aclnn {self.name} inputs must use the same NPU")
        if x.dtype != y.dtype:
            raise TypeError(
                f"aclnn {self.name} inputs must have the same dtype"
            )
        if not self.supports_dtype(x.dtype):
            raise TypeError(
                f"aclnn {self.name} reference does not support {x.dtype}"
            )
        if x.dim() == 0 or y.dim() == 0 or x.dim() > 8 or y.dim() > 8:
            raise ValueError(
                f"aclnn {self.name} requires ranks from 1 through 8"
            )
        if x.storage_offset() != 0 or y.storage_offset() != 0:
            raise ValueError(
                f"aclnn {self.name} requires zero storage offsets"
            )

    def _output_shape(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[int, ...]:
        try:
            shape = tuple(torch.broadcast_shapes(x.shape, y.shape))
        except RuntimeError as exc:
            raise ValueError(
                f"aclnn {self.name} inputs are not broadcastable: "
                f"x_shape={tuple(x.shape)}, y_shape={tuple(y.shape)}"
            ) from exc
        if len(shape) > 8:
            raise ValueError(
                f"aclnn {self.name} requires output rank through 8"
            )
        return shape

    def run(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: Number = 1,
    ) -> torch.Tensor:
        prepared = self.prepare(x, y, alpha=alpha)
        try:
            return prepared.run()
        finally:
            prepared.close()

    def prepare(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: Number = 1,
    ) -> PreparedAclnnOperation:
        alpha_value = _normalize_alpha(alpha)
        self._validate(x, y)
        output_shape = self._output_shape(x, y)
        context = self._context
        npu = context.npu()
        with npu.device(x.device):
            output_dtype = (
                torch.bool
                if self.name in _COMPARISON_OPERATIONS
                or self.name in _LOGICAL_OPERATIONS
                else x.dtype
            )
            output = torch.empty(
                output_shape, device=x.device, dtype=output_dtype
            )
            metadata = (
                metadata_array(x.shape),
                metadata_array(x.stride()),
                metadata_array(y.shape),
                metadata_array(y.stride()),
                metadata_array(output.shape),
                metadata_array(output.stride()),
            )
            pointers = (x.data_ptr(), y.data_ptr(), output.data_ptr())
            if any(int(pointer) == 0 for pointer in pointers):
                raise ValueError(
                    f"prepared aclnn {self.name} received a null pointer"
                )
            stream_pointer = int(
                npu.current_stream(device=x.device).npu_stream
            )
            if stream_pointer == 0:
                raise RuntimeError("torch_npu returned a null current stream")
            handle = ctypes.c_void_p()
            error_buffer = ctypes.create_string_buffer(ERROR_BUFFER_SIZE)
            library = context.get_library(configure_binary)
            status = library.flagdnn_aclnn_binary_create(
                ctypes.c_int32(self._operation_code),
                ctypes.c_void_p(int(pointers[0])),
                metadata[0],
                metadata[1],
                ctypes.c_uint64(x.dim()),
                ctypes.c_void_p(int(pointers[1])),
                metadata[2],
                metadata[3],
                ctypes.c_uint64(y.dim()),
                ctypes.c_void_p(int(pointers[2])),
                metadata[4],
                metadata[5],
                ctypes.c_uint64(output.dim()),
                ctypes.c_int32(_BINARY_DTYPE_CODES[x.dtype]),
                ctypes.c_int32(_BINARY_DTYPE_CODES[output_dtype]),
                ctypes.c_double(alpha_value),
                ctypes.c_void_p(stream_pointer),
                ctypes.byref(handle),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
        if status != 0 or handle.value is None:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"prepared aclnn {self.name} creation failed: "
                f"status={status}, detail={detail}, "
                f"x_shape={tuple(x.shape)}, y_shape={tuple(y.shape)}, "
                f"dtype={x.dtype}, device={x.device}"
            )
        context.last_device = x.device
        return PreparedAclnnOperation(
            library,
            handle,
            (x, y),
            output,
            operation_name=f"aclnn{self.name.title()}",
            symbol_name="aclnn_binary",
        )


def create_binary_operations(
    context: AscendContext,
) -> tuple[AscendBinaryOperation, ...]:
    return tuple(
        AscendBinaryOperation(name, context)
        for name in BINARY_OPERATION_CODES
        if not name.startswith("_")
    )
