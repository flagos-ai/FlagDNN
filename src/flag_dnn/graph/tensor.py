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

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Optional, Tuple, Union

import torch

ShapeDim = Union[int, str]

_DTYPE_ALIASES = {
    "torch.float16": "float16",
    "half": "float16",
    "fp16": "float16",
    "float16": "float16",
    "torch.bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "torch.float32": "float32",
    "float": "float32",
    "fp32": "float32",
    "float32": "float32",
    "torch.float64": "float64",
    "double": "float64",
    "fp64": "float64",
    "float64": "float64",
    "torch.int8": "int8",
    "int8": "int8",
    "torch.int16": "int16",
    "int16": "int16",
    "torch.int32": "int32",
    "int32": "int32",
    "torch.int64": "int64",
    "long": "int64",
    "int64": "int64",
    "torch.uint8": "uint8",
    "uint8": "uint8",
    "torch.bool": "bool",
    "bool": "bool",
    "torch.float8_e4m3fn": "float8_e4m3fn",
    "float8_e4m3fn": "float8_e4m3fn",
    "fp8_e4m3": "float8_e4m3fn",
    "torch.float8_e5m2": "float8_e5m2",
    "float8_e5m2": "float8_e5m2",
    "fp8_e5m2": "float8_e5m2",
}

_TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}

_ITEM_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}


def canonical_dtype(dtype: Any) -> str:
    if isinstance(dtype, torch.dtype):
        return _DTYPE_ALIASES[str(dtype)]
    if dtype is None:
        return "unknown"
    key = str(dtype).lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"unsupported dtype in graph TensorSpec: {dtype}")
    return _DTYPE_ALIASES[key]


def torch_dtype(dtype: Any) -> torch.dtype:
    name = canonical_dtype(dtype)
    if name not in _TORCH_DTYPES:
        raise ValueError(f"cannot convert dtype to torch.dtype: {dtype}")
    return _TORCH_DTYPES[name]


def dtype_itemsize(dtype: Any) -> int:
    return _ITEM_SIZES.get(canonical_dtype(dtype), 0)


def normalize_shape(shape: Any) -> Tuple[ShapeDim, ...]:
    if shape is None:
        return tuple()
    if isinstance(shape, torch.Size):
        return tuple(int(dim) for dim in shape)
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def infer_layout(tensor: torch.Tensor) -> Optional[str]:
    if tensor.dim() == 4 and tensor.is_contiguous(
        memory_format=torch.channels_last
    ):
        return "nhwc"
    if tensor.is_contiguous():
        return "contiguous"
    return "strided"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[ShapeDim, ...]
    dtype: str
    stride: Optional[Tuple[int, ...]] = None
    layout: Optional[str] = None
    device: Optional[str] = None
    is_input: bool = False
    is_output: bool = False
    contiguous: Optional[bool] = None

    def __init__(
        self,
        name: str,
        shape: Any,
        dtype: Any,
        stride: Optional[Any] = None,
        layout: Optional[str] = None,
        device: Optional[Any] = None,
        is_input: bool = False,
        is_output: bool = False,
        contiguous: Optional[bool] = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "shape", normalize_shape(shape))
        object.__setattr__(self, "dtype", canonical_dtype(dtype))
        object.__setattr__(
            self,
            "stride",
            None if stride is None else tuple(int(s) for s in stride),
        )
        object.__setattr__(self, "layout", layout)
        object.__setattr__(
            self, "device", None if device is None else str(device)
        )
        object.__setattr__(self, "is_input", bool(is_input))
        object.__setattr__(self, "is_output", bool(is_output))
        object.__setattr__(self, "contiguous", contiguous)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, name: str = "") -> "TensorSpec":
        return cls(
            name=name,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            stride=tuple(tensor.stride()),
            layout=infer_layout(tensor),
            device=tensor.device,
            contiguous=tensor.is_contiguous(),
        )

    def with_name(self, name: str) -> "TensorSpec":
        return TensorSpec(
            name=name,
            shape=self.shape,
            dtype=self.dtype,
            stride=self.stride,
            layout=self.layout,
            device=self.device,
            is_input=self.is_input,
            is_output=self.is_output,
            contiguous=self.contiguous,
        )

    def as_input(self) -> "TensorSpec":
        return TensorSpec(
            name=self.name,
            shape=self.shape,
            dtype=self.dtype,
            stride=self.stride,
            layout=self.layout,
            device=self.device,
            is_input=True,
            is_output=self.is_output,
            contiguous=self.contiguous,
        )

    def as_output(self) -> "TensorSpec":
        return TensorSpec(
            name=self.name,
            shape=self.shape,
            dtype=self.dtype,
            stride=self.stride,
            layout=self.layout,
            device=self.device,
            is_input=self.is_input,
            is_output=True,
            contiguous=self.contiguous,
        )

    def numel(self) -> Optional[int]:
        if any(not isinstance(dim, int) for dim in self.shape):
            return None
        if not self.shape:
            return 1
        return reduce(mul, self.shape, 1)

    def nbytes(self) -> Optional[int]:
        numel = self.numel()
        if numel is None:
            return None
        return numel * dtype_itemsize(self.dtype)

    def signature(self) -> dict:
        return {
            "shape": tuple(self.shape),
            "dtype": self.dtype,
            "stride": self.stride,
            "layout": self.layout,
            "device": self.device,
            "contiguous": self.contiguous,
        }


class GraphTensor:
    def __init__(self, value_id: int, graph: Any):
        self.value_id = value_id
        self.graph = graph

    @property
    def spec(self) -> TensorSpec:
        return self.graph.values[self.value_id].spec

    @property
    def shape(self) -> Tuple[ShapeDim, ...]:
        return self.spec.shape

    @property
    def dtype(self) -> str:
        return self.spec.dtype

    @property
    def device(self) -> Optional[str]:
        return self.spec.device

    def dim(self) -> int:
        return len(self.spec.shape)

    def numel(self) -> Optional[int]:
        return self.spec.numel()

    def _binary(self, op_type: str, other: Any, reverse: bool = False):
        from flag_dnn.graph.capture import current_capture

        ctx = current_capture()
        if ctx is None:
            raise RuntimeError("GraphTensor operation used outside capture")
        return ctx.add_binary_op(op_type, self, other, reverse=reverse)

    def __add__(self, other: Any):
        return self._binary("add", other)

    def __radd__(self, other: Any):
        return self._binary("add", other, reverse=True)

    def __sub__(self, other: Any):
        return self._binary("sub", other)

    def __rsub__(self, other: Any):
        return self._binary("sub", other, reverse=True)

    def __mul__(self, other: Any):
        return self._binary("mul", other)

    def __rmul__(self, other: Any):
        return self._binary("mul", other, reverse=True)

    def __truediv__(self, other: Any):
        return self._binary("div", other)

    def __rtruediv__(self, other: Any):
        return self._binary("div", other, reverse=True)

    def __mod__(self, other: Any):
        return self._binary("mod", other)

    def __rmod__(self, other: Any):
        return self._binary("mod", other, reverse=True)

    def __matmul__(self, other: Any):
        from flag_dnn.graph.capture import current_capture

        ctx = current_capture()
        if ctx is None:
            raise RuntimeError("GraphTensor operation used outside capture")
        return ctx.add_op_call("mm", (self, other), {})

    def __repr__(self) -> str:
        spec = self.spec
        return (
            f"GraphTensor(value_id={self.value_id}, "
            f"shape={spec.shape}, dtype={spec.dtype})"
        )
