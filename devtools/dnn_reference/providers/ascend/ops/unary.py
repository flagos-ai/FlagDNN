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


UNARY_OPERATION_CODES = {
    "neg": 0,
    "sqrt": 1,
    "rsqrt": 2,
    "reciprocal": 3,
    "ceil": 4,
    "floor": 5,
    "exp": 6,
    "log": 7,
    "erf": 8,
    "sin": 9,
    "cos": 10,
    "tan": 11,
    "relu": 12,
    "sigmoid": 13,
    "tanh": 14,
    "logical_not": 15,
    "leaky_relu": 16,
    "elu": 17,
    "gelu": 18,
    "gelu_approx_tanh": 19,
    "swish": 20,
    "softplus": 21,
}

_UNARY_DTYPE_CODES = {**DTYPE_CODES, torch.bool: 3}


def configure_unary(library: Any) -> None:
    create = library.flagdnn_aclnn_unary_create
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
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    create.restype = ctypes.c_int
    for name in (
        "flagdnn_aclnn_unary_run",
        "flagdnn_aclnn_unary_destroy",
    ):
        function = getattr(library, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int


class AscendUnaryOperation:
    def __init__(self, name: str, context: AscendContext) -> None:
        if name not in UNARY_OPERATION_CODES:
            raise ValueError(f"unsupported generic ACLNN unary op: {name}")
        self.name = name
        self._context = context
        self._operation_code = UNARY_OPERATION_CODES[name]

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        if self.name == "logical_not":
            return dtype == torch.bool
        return dtype in DTYPE_CODES

    def _validate(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"aclnn {self.name} reference expects a torch.Tensor input"
            )
        if x.layout != torch.strided:
            raise ValueError(
                f"aclnn {self.name} reference requires a strided tensor"
            )
        if x.device.type != "npu":
            raise ValueError(
                f"aclnn {self.name} reference requires an NPU tensor"
            )
        if not self.supports_dtype(x.dtype):
            raise TypeError(
                f"aclnn {self.name} reference does not support {x.dtype}"
            )
        if x.dim() == 0 or x.dim() > 8:
            raise ValueError(
                f"aclnn {self.name} reference requires rank from 1 through 8"
            )
        if x.storage_offset() != 0:
            raise ValueError(
                f"aclnn {self.name} reference requires zero storage offset"
            )

    def _parameters(
        self,
        *,
        negative_slope: float = 0.01,
        alpha: float = 1.0,
        swish_beta: float | None = None,
        beta: float = 1.0,
        threshold: float = 20.0,
    ) -> tuple[float, float, float]:
        if self.name == "leaky_relu":
            return float(negative_slope), 0.0, 0.0
        if self.name == "elu":
            return float(alpha), 1.0, 1.0
        if self.name == "swish":
            value = 1.0 if swish_beta is None else float(swish_beta)
            return value, 0.0, 0.0
        if self.name == "softplus":
            beta_value = float(beta)
            if beta_value <= 0.0:
                raise ValueError(
                    f"softplus beta must be positive, got {beta_value}"
                )
            return beta_value, float(threshold), 0.0
        return 0.0, 0.0, 0.0

    def run(
        self,
        x: torch.Tensor,
        *,
        negative_slope: float = 0.01,
        alpha: float = 1.0,
        swish_beta: float | None = None,
        beta: float = 1.0,
        threshold: float = 20.0,
        approximate: str = "none",
    ) -> torch.Tensor:
        if self.name == "gelu" and approximate != "none":
            raise ValueError(
                "gelu provider expects approximate='none'; use "
                "gelu_approx_tanh for the tanh approximation"
            )
        prepared = self.prepare(
            x,
            negative_slope=negative_slope,
            alpha=alpha,
            swish_beta=swish_beta,
            beta=beta,
            threshold=threshold,
            approximate=approximate,
        )
        try:
            return prepared.run()
        finally:
            prepared.close()

    def prepare(
        self,
        x: torch.Tensor,
        *,
        negative_slope: float = 0.01,
        alpha: float = 1.0,
        swish_beta: float | None = None,
        beta: float = 1.0,
        threshold: float = 20.0,
        approximate: str = "none",
    ) -> PreparedAclnnOperation:
        if self.name == "gelu" and approximate != "none":
            raise ValueError(
                "gelu provider expects approximate='none'; use "
                "gelu_approx_tanh for the tanh approximation"
            )
        self._validate(x)
        parameters = self._parameters(
            negative_slope=negative_slope,
            alpha=alpha,
            swish_beta=swish_beta,
            beta=beta,
            threshold=threshold,
        )
        context = self._context
        npu = context.npu()
        with npu.device(x.device):
            output = torch.empty_strided(
                tuple(x.shape),
                tuple(x.stride()),
                device=x.device,
                dtype=x.dtype,
            )
            input_shape = metadata_array(x.shape)
            input_stride = metadata_array(x.stride())
            output_shape = metadata_array(output.shape)
            output_stride = metadata_array(output.stride())
            if int(x.data_ptr()) == 0 or int(output.data_ptr()) == 0:
                raise ValueError(
                    f"prepared aclnn {self.name} received a null data pointer"
                )
            stream_pointer = int(
                npu.current_stream(device=x.device).npu_stream
            )
            if stream_pointer == 0:
                raise RuntimeError("torch_npu returned a null current stream")
            handle = ctypes.c_void_p()
            error_buffer = ctypes.create_string_buffer(ERROR_BUFFER_SIZE)
            library = context.get_library(configure_unary)
            status = library.flagdnn_aclnn_unary_create(
                ctypes.c_int32(self._operation_code),
                ctypes.c_void_p(int(x.data_ptr())),
                input_shape,
                input_stride,
                ctypes.c_uint64(x.dim()),
                ctypes.c_void_p(int(output.data_ptr())),
                output_shape,
                output_stride,
                ctypes.c_uint64(output.dim()),
                ctypes.c_int32(_UNARY_DTYPE_CODES[x.dtype]),
                ctypes.c_double(parameters[0]),
                ctypes.c_double(parameters[1]),
                ctypes.c_double(parameters[2]),
                ctypes.c_void_p(stream_pointer),
                ctypes.byref(handle),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
        if status != 0 or handle.value is None:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"prepared aclnn {self.name} creation failed: "
                f"status={status}, detail={detail}, shape={tuple(x.shape)}, "
                f"dtype={x.dtype}, device={x.device}"
            )
        context.last_device = x.device
        return PreparedAclnnOperation(
            library,
            handle,
            (x,),
            output,
            operation_name=f"aclnn{self.name.title()}",
            symbol_name="aclnn_unary",
        )


def create_unary_operations(
    context: AscendContext,
) -> tuple[AscendUnaryOperation, ...]:
    return tuple(
        AscendUnaryOperation(name, context) for name in UNARY_OPERATION_CODES
    )
