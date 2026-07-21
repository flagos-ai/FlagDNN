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
from typing import Any

import torch

from ..common import (
    DTYPE_CODES,
    ERROR_BUFFER_SIZE,
    INT64_POINTER,
    AscendContext,
    PreparedAclnnOperation,
    metadata_array,
)


def configure_abs(library: Any) -> None:
    function = library.flagdnn_test_aclnn_abs
    function.argtypes = [
        ctypes.c_void_p,
        INT64_POINTER,
        INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_void_p,
        INT64_POINTER,
        INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    create = library.flagdnn_aclnn_abs_create
    create.argtypes = [
        ctypes.c_void_p,
        INT64_POINTER,
        INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_void_p,
        INT64_POINTER,
        INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    create.restype = ctypes.c_int
    for name in ("flagdnn_aclnn_abs_run", "flagdnn_aclnn_abs_destroy"):
        prepared = getattr(library, name)
        prepared.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        prepared.restype = ctypes.c_int


class AscendAbsOperation:
    name = "abs"

    def __init__(self, context: AscendContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in DTYPE_CODES

    def _validate(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("aclnnAbs reference expects a torch.Tensor input")
        if x.layout != torch.strided:
            raise ValueError(
                "aclnnAbs reference requires a strided tensor, "
                f"got {x.layout}"
            )
        if x.device.type != "npu":
            raise ValueError(
                "aclnnAbs reference requires an NPU tensor, " f"got {x.device}"
            )
        if not self.supports_dtype(x.dtype):
            raise TypeError(f"aclnnAbs reference does not support {x.dtype}")
        if x.dim() == 0:
            raise ValueError(
                "aclnnAbs reference does not support rank-0 tensors"
            )
        if x.dim() > 8:
            raise ValueError(
                "aclnnAbs reference requires tensor rank from 1 through 8, "
                f"got {x.dim()}"
            )
        if x.storage_offset() != 0:
            raise ValueError(
                "aclnnAbs reference requires a zero storage offset, "
                f"got {x.storage_offset()}"
            )
        is_channels_last = x.dim() == 4 and x.is_contiguous(
            memory_format=torch.channels_last
        )
        if not x.is_contiguous() and not is_channels_last:
            raise ValueError(
                "aclnnAbs reference requires a contiguous or 4D "
                "channels-last dense tensor"
            )

    def _allocate_output(self, x: torch.Tensor) -> torch.Tensor:
        return torch.empty_strided(
            tuple(x.shape),
            tuple(x.stride()),
            device=x.device,
            dtype=x.dtype,
        )

    def run(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x)
        context = self._context
        npu = context.npu()
        with npu.device(x.device):
            output = self._allocate_output(x)
            metadata = (
                metadata_array(x.shape),
                metadata_array(x.stride()),
                metadata_array(output.shape),
                metadata_array(output.stride()),
            )
            pointers = (x.data_ptr(), output.data_ptr())
            if any(int(pointer) == 0 for pointer in pointers):
                raise ValueError(
                    "aclnnAbs reference received a null tensor data pointer"
                )
            stream_pointer = int(
                npu.current_stream(device=x.device).npu_stream
            )
            if stream_pointer == 0:
                raise RuntimeError("torch_npu returned a null current stream")
            error_buffer = ctypes.create_string_buffer(ERROR_BUFFER_SIZE)
            status = context.get_library(configure_abs).flagdnn_test_aclnn_abs(
                ctypes.c_void_p(int(pointers[0])),
                metadata[0],
                metadata[1],
                ctypes.c_uint64(x.dim()),
                ctypes.c_void_p(int(pointers[1])),
                metadata[2],
                metadata[3],
                ctypes.c_uint64(output.dim()),
                ctypes.c_int32(DTYPE_CODES[x.dtype]),
                ctypes.c_void_p(stream_pointer),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
        if status != 0:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                "aclnnAbs reference failed: "
                f"status={status}, detail={detail}, "
                f"x_shape={tuple(x.shape)}, x_stride={tuple(x.stride())}, "
                f"output_shape={tuple(output.shape)}, "
                f"output_stride={tuple(output.stride())}, "
                f"dtype={x.dtype}, device={x.device}, "
                f"stream=0x{stream_pointer:x}"
            )
        context.last_device = x.device
        return output

    def prepare(self, x: torch.Tensor) -> PreparedAclnnOperation:
        self._validate(x)
        context = self._context
        npu = context.npu()
        with npu.device(x.device):
            output = self._allocate_output(x)
            metadata = (
                metadata_array(x.shape),
                metadata_array(x.stride()),
                metadata_array(output.shape),
                metadata_array(output.stride()),
            )
            pointers = (x.data_ptr(), output.data_ptr())
            if any(int(pointer) == 0 for pointer in pointers):
                raise ValueError(
                    "prepared aclnnAbs received a null tensor data pointer"
                )
            stream_pointer = int(
                npu.current_stream(device=x.device).npu_stream
            )
            if stream_pointer == 0:
                raise RuntimeError("torch_npu returned a null current stream")
            handle = ctypes.c_void_p()
            error_buffer = ctypes.create_string_buffer(ERROR_BUFFER_SIZE)
            library = context.get_library(configure_abs)
            status = library.flagdnn_aclnn_abs_create(
                ctypes.c_void_p(int(pointers[0])),
                metadata[0],
                metadata[1],
                ctypes.c_uint64(x.dim()),
                ctypes.c_void_p(int(pointers[1])),
                metadata[2],
                metadata[3],
                ctypes.c_uint64(output.dim()),
                ctypes.c_int32(DTYPE_CODES[x.dtype]),
                ctypes.c_void_p(stream_pointer),
                ctypes.byref(handle),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
        if status != 0 or handle.value is None:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                "prepared aclnnAbs creation failed: "
                f"status={status}, detail={detail}, "
                f"x_shape={tuple(x.shape)}, dtype={x.dtype}, "
                f"device={x.device}"
            )
        context.last_device = x.device
        return PreparedAclnnOperation(
            library,
            handle,
            (x,),
            output,
            operation_name="aclnnAbs",
        )
