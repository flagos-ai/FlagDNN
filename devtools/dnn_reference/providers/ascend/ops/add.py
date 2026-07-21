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


def configure_add(library: Any) -> None:
    function = library.flagdnn_test_aclnn_add
    function.argtypes = [
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
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int

    create = library.flagdnn_aclnn_add_create
    create.argtypes = [
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
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    create.restype = ctypes.c_int
    for name in ("flagdnn_aclnn_add_run", "flagdnn_aclnn_add_destroy"):
        prepared = getattr(library, name)
        prepared.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        prepared.restype = ctypes.c_int


def _normalize_alpha(alpha: Number) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(
            "aclnnAdd reference alpha must be an int or float, "
            f"got {type(alpha).__name__}"
        )
    try:
        return float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "aclnnAdd reference alpha cannot be represented as a double"
        ) from exc


class AscendAddOperation:
    name = "add"

    def __init__(self, context: AscendContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in DTYPE_CODES

    def _validate(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError(
                "aclnnAdd reference expects two torch.Tensor inputs"
            )
        if x.layout != torch.strided or y.layout != torch.strided:
            raise ValueError(
                "aclnnAdd reference requires strided tensors, "
                f"got {x.layout} and {y.layout}"
            )
        if x.device.type != "npu" or y.device.type != "npu":
            raise ValueError(
                "aclnnAdd reference requires NPU tensors, "
                f"got {x.device} and {y.device}"
            )
        if x.device != y.device:
            raise ValueError(
                "aclnnAdd reference inputs must be on the same NPU, "
                f"got {x.device} and {y.device}"
            )
        if x.dtype != y.dtype:
            raise TypeError(
                "aclnnAdd reference inputs must have the same dtype, "
                f"got {x.dtype} and {y.dtype}"
            )
        if not self.supports_dtype(x.dtype):
            raise TypeError(f"aclnnAdd reference does not support {x.dtype}")
        if x.dim() == 0 or y.dim() == 0:
            raise ValueError(
                "aclnnAdd reference does not support rank-0 tensors"
            )
        if x.storage_offset() != 0 or y.storage_offset() != 0:
            raise ValueError(
                "aclnnAdd reference requires zero storage offsets, "
                f"got {x.storage_offset()} and {y.storage_offset()}"
            )

    @staticmethod
    def _output_shape(x: torch.Tensor, y: torch.Tensor) -> tuple[int, ...]:
        try:
            return tuple(torch.broadcast_shapes(x.shape, y.shape))
        except RuntimeError as exc:
            raise ValueError(
                "aclnnAdd inputs are not broadcastable: "
                f"x_shape={tuple(x.shape)}, y_shape={tuple(y.shape)}"
            ) from exc

    def run(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: Number = 1,
    ) -> torch.Tensor:
        alpha_value = _normalize_alpha(alpha)
        self._validate(x, y)
        output_shape = self._output_shape(x, y)
        context = self._context
        npu = context.npu()
        with npu.device(x.device):
            output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
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
                    "aclnnAdd reference received a null tensor data pointer"
                )
            stream_pointer = int(
                npu.current_stream(device=x.device).npu_stream
            )
            if stream_pointer == 0:
                raise RuntimeError("torch_npu returned a null current stream")
            error_buffer = ctypes.create_string_buffer(ERROR_BUFFER_SIZE)
            status = context.get_library(configure_add).flagdnn_test_aclnn_add(
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
                ctypes.c_int32(DTYPE_CODES[x.dtype]),
                ctypes.c_double(alpha_value),
                ctypes.c_void_p(stream_pointer),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
        if status != 0:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                "aclnnAdd reference failed: "
                f"status={status}, detail={detail}, "
                f"x_shape={tuple(x.shape)}, x_stride={tuple(x.stride())}, "
                f"y_shape={tuple(y.shape)}, y_stride={tuple(y.stride())}, "
                f"output_shape={tuple(output.shape)}, "
                f"output_stride={tuple(output.stride())}, "
                f"dtype={x.dtype}, device={x.device}, alpha={alpha}, "
                f"stream=0x{stream_pointer:x}"
            )
        context.last_device = x.device
        return output

    def prepare(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: Number = 1,
    ) -> PreparedAclnnOperation:
        alpha_value = _normalize_alpha(alpha)
        self._validate(x, y)
        if x.dim() > 8 or y.dim() > 8:
            raise ValueError(
                "prepared aclnnAdd requires input tensor rank from 1 through 8"
            )
        output_shape = self._output_shape(x, y)
        if len(output_shape) > 8:
            raise ValueError(
                "prepared aclnnAdd requires output tensor rank through 8"
            )

        context = self._context
        npu = context.npu()
        with npu.device(x.device):
            output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
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
                    "prepared aclnnAdd received a null tensor data pointer"
                )
            stream_pointer = int(
                npu.current_stream(device=x.device).npu_stream
            )
            if stream_pointer == 0:
                raise RuntimeError("torch_npu returned a null current stream")
            handle = ctypes.c_void_p()
            error_buffer = ctypes.create_string_buffer(ERROR_BUFFER_SIZE)
            library = context.get_library(configure_add)
            status = library.flagdnn_aclnn_add_create(
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
                ctypes.c_int32(DTYPE_CODES[x.dtype]),
                ctypes.c_double(alpha_value),
                ctypes.c_void_p(stream_pointer),
                ctypes.byref(handle),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
        if status != 0 or handle.value is None:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                "prepared aclnnAdd creation failed: "
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
            operation_name="aclnnAdd",
        )
