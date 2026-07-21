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


def configure_binary_select(library: Any) -> None:
    create = library.flagdnn_aclnn_binary_select_create
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
    for name in (
        "flagdnn_aclnn_binary_select_run",
        "flagdnn_aclnn_binary_select_destroy",
    ):
        function = getattr(library, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int


class AscendBinarySelectOperation:
    name = "binary_select"

    def __init__(self, context: AscendContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in DTYPE_CODES

    def _validate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if not all(isinstance(item, torch.Tensor) for item in (x, y, mask)):
            raise TypeError(
                "aclnn binary_select expects x, y, and mask tensors"
            )
        if any(item.layout != torch.strided for item in (x, y, mask)):
            raise ValueError("aclnn binary_select requires strided tensors")
        if any(item.device.type != "npu" for item in (x, y, mask)):
            raise ValueError("aclnn binary_select requires NPU tensors")
        if x.device != y.device or x.device != mask.device:
            raise ValueError(
                "aclnn binary_select inputs must use the same NPU"
            )
        if x.dtype != y.dtype:
            raise TypeError(
                "aclnn binary_select x and y must have the same dtype"
            )
        if not self.supports_dtype(x.dtype):
            raise TypeError(f"aclnn binary_select does not support {x.dtype}")
        if mask.dtype != torch.bool:
            raise TypeError(
                "aclnn binary_select mask must have dtype torch.bool"
            )
        if any(item.dim() == 0 or item.dim() > 8 for item in (x, y, mask)):
            raise ValueError(
                "aclnn binary_select requires ranks from 1 through 8"
            )
        if any(item.storage_offset() != 0 for item in (x, y, mask)):
            raise ValueError(
                "aclnn binary_select requires zero storage offsets"
            )

    def _output_shape(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[int, ...]:
        try:
            shape = tuple(torch.broadcast_shapes(x.shape, y.shape, mask.shape))
        except RuntimeError as exc:
            raise ValueError(
                "aclnn binary_select inputs are not broadcastable: "
                f"x={tuple(x.shape)}, y={tuple(y.shape)}, "
                f"mask={tuple(mask.shape)}"
            ) from exc
        if len(shape) > 8:
            raise ValueError(
                "aclnn binary_select output rank must not exceed 8"
            )
        return shape

    def run(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        prepared = self.prepare(x, y, mask)
        try:
            return prepared.run()
        finally:
            prepared.close()

    def prepare(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> PreparedAclnnOperation:
        self._validate(x, y, mask)
        output_shape = self._output_shape(x, y, mask)
        context = self._context
        npu = context.npu()
        with npu.device(x.device):
            output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
            tensors = (x, y, mask, output)
            metadata = tuple(
                item
                for tensor in tensors
                for item in (
                    metadata_array(tensor.shape),
                    metadata_array(tensor.stride()),
                )
            )
            if any(int(tensor.data_ptr()) == 0 for tensor in tensors):
                raise ValueError(
                    "prepared aclnn binary_select received a null pointer"
                )
            stream_pointer = int(
                npu.current_stream(device=x.device).npu_stream
            )
            if stream_pointer == 0:
                raise RuntimeError("torch_npu returned a null current stream")
            handle = ctypes.c_void_p()
            error_buffer = ctypes.create_string_buffer(ERROR_BUFFER_SIZE)
            library = context.get_library(configure_binary_select)
            status = library.flagdnn_aclnn_binary_select_create(
                ctypes.c_void_p(int(x.data_ptr())),
                metadata[0],
                metadata[1],
                ctypes.c_uint64(x.dim()),
                ctypes.c_void_p(int(y.data_ptr())),
                metadata[2],
                metadata[3],
                ctypes.c_uint64(y.dim()),
                ctypes.c_void_p(int(mask.data_ptr())),
                metadata[4],
                metadata[5],
                ctypes.c_uint64(mask.dim()),
                ctypes.c_void_p(int(output.data_ptr())),
                metadata[6],
                metadata[7],
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
                "prepared aclnn binary_select creation failed: "
                f"status={status}, detail={detail}, "
                f"x_shape={tuple(x.shape)}, y_shape={tuple(y.shape)}, "
                f"mask_shape={tuple(mask.shape)}, dtype={x.dtype}, "
                f"device={x.device}"
            )
        context.last_device = x.device
        return PreparedAclnnOperation(
            library,
            handle,
            (x, y, mask),
            output,
            operation_name="aclnnSWhere",
            symbol_name="aclnn_binary_select",
        )
