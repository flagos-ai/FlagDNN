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

import builtins
import ctypes
from typing import Any, Iterable, Sequence

import torch

from .binary import AscendBinaryOperation
from ..common import (
    DTYPE_CODES,
    ERROR_BUFFER_SIZE,
    INT64_POINTER,
    AscendContext,
    PreparedAclnnOperation,
    metadata_array,
)


UTILITY_OPERATION_CODES = {
    "transpose": 1,
    "slice": 2,
    "concatenate": 3,
    "gen_index": 4,
}


def configure_utility(library: Any) -> None:
    create = library.flagdnn_aclnn_utility_create
    tensor_arguments = [
        ctypes.c_void_p,
        INT64_POINTER,
        INT64_POINTER,
        ctypes.c_uint64,
    ]
    create.argtypes = [
        ctypes.c_int32,
        *tensor_arguments,
        *tensor_arguments,
        *tensor_arguments,
        ctypes.c_uint64,
        ctypes.c_void_p,
        INT64_POINTER,
        INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_int32,
        INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    create.restype = ctypes.c_int
    for name in (
        "flagdnn_aclnn_utility_run",
        "flagdnn_aclnn_utility_destroy",
    ):
        function = getattr(library, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int


class PreparedViewOperation:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self._closed = False

    def run(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("prepared logical view operation is closed")
        return self.output

    def __call__(self) -> torch.Tensor:
        return self.run()

    def close(self) -> None:
        self._closed = True


def _normalize_axis(axis: int, rank: int) -> int:
    value = int(axis)
    if value < 0:
        value += rank
    if value < 0 or value >= rank:
        raise IndexError(f"axis out of range for rank {rank}: {axis}")
    return value


def _normalize_permutation(
    permutation: Iterable[int], rank: int
) -> tuple[int, ...]:
    values = tuple(_normalize_axis(item, rank) for item in permutation)
    if len(values) != rank or len(set(values)) != rank:
        raise ValueError(
            f"invalid permutation {values} for tensor rank {rank}"
        )
    return values


def _normalize_slices(slices: Any, shape: Sequence[int]) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    rank = len(shape)
    if slices is None:
        specs: tuple[Any, ...] = ()
    elif isinstance(slices, builtins.slice):
        specs = (slices,)
    else:
        specs = tuple(slices)
    if len(specs) > rank:
        raise IndexError(
            f"too many slice specs for tensor rank {rank}: {len(specs)}"
        )
    specs += (builtins.slice(None),) * (rank - len(specs))
    starts: list[int] = []
    ends: list[int] = []
    steps: list[int] = []
    output_shape: list[int] = []
    for dim, spec in zip(shape, specs):
        if isinstance(spec, (tuple, list)):
            if len(spec) != 3:
                raise TypeError(
                    "slice tuple specs must be (start, stop, step)"
                )
            spec = builtins.slice(*spec)
        if not isinstance(spec, builtins.slice):
            raise TypeError(f"invalid slice spec: {spec!r}")
        start, end, step = spec.indices(int(dim))
        if step <= 0:
            raise ValueError("slice step must be greater than zero")
        starts.append(start)
        ends.append(end)
        steps.append(step)
        output_shape.append(len(range(start, end, step)))
    return (
        tuple(starts),
        tuple(ends),
        tuple(range(rank)),
        tuple(steps),
        tuple(output_shape),
    )


class AscendReshapeOperation:
    name = "reshape"

    def __init__(self, context: AscendContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in DTYPE_CODES

    def prepare(
        self,
        x: torch.Tensor,
        shape: Iterable[int],
        **_: Any,
    ) -> PreparedViewOperation:
        _validate_tensor(self.name, x)
        target_shape = tuple(int(dim) for dim in shape)
        try:
            output = x.view(target_shape)
        except RuntimeError as exc:
            raise ValueError(
                "DNN reshape reference requires a logical view"
            ) from exc
        self._context.last_device = x.device
        return PreparedViewOperation(output)

    def run(
        self,
        x: torch.Tensor,
        shape: Iterable[int],
        **kwargs: Any,
    ) -> torch.Tensor:
        prepared = self.prepare(x, shape, **kwargs)
        try:
            return prepared.run()
        finally:
            prepared.close()


class AscendIdentityOperation:
    name = "identity"

    def __init__(self, context: AscendContext) -> None:
        self._context = context
        self._multiply = AscendBinaryOperation("mul", context)
        self._ones: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return self._multiply.supports_dtype(dtype)

    def _one(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        value = self._ones.get(key)
        if value is None:
            value = torch.ones((1,), device=x.device, dtype=x.dtype)
            self._ones[key] = value
        return value

    def prepare(self, x: torch.Tensor, **_: Any) -> PreparedAclnnOperation:
        _validate_tensor(self.name, x)
        return self._multiply.prepare(x, self._one(x))

    def run(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        prepared = self.prepare(x, **kwargs)
        try:
            return prepared.run()
        finally:
            prepared.close()


def _validate_tensor(op_name: str, tensor: Any) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"aclnn {op_name} expects torch.Tensor inputs")
    if tensor.layout != torch.strided:
        raise ValueError(f"aclnn {op_name} requires strided tensors")
    if tensor.device.type != "npu":
        raise ValueError(f"aclnn {op_name} requires NPU tensors")
    if tensor.dtype not in DTYPE_CODES:
        raise TypeError(f"aclnn {op_name} does not support {tensor.dtype}")
    if tensor.dim() == 0 or tensor.dim() > 8:
        raise ValueError(f"aclnn {op_name} requires rank from 1 through 8")
    if tensor.storage_offset() != 0:
        raise ValueError(f"aclnn {op_name} requires zero storage offset")
    if tensor.numel() == 0:
        raise ValueError(f"aclnn {op_name} does not support empty tensors")
    return tensor


class AscendUtilityOperation:
    def __init__(self, name: str, context: AscendContext) -> None:
        if name not in UTILITY_OPERATION_CODES:
            raise ValueError(f"unsupported ACLNN utility op: {name}")
        self.name = name
        self._context = context
        self._operation_code = UTILITY_OPERATION_CODES[name]

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in DTYPE_CODES

    def _prepare_arguments(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[
        tuple[torch.Tensor, ...],
        torch.Tensor,
        tuple[int, ...],
    ]:
        if self.name == "identity":
            (x,) = args
            x = _validate_tensor(self.name, x)
            output = torch.empty_strided(
                tuple(x.shape),
                tuple(x.stride()),
                device=x.device,
                dtype=x.dtype,
            )
            return (x,), output, ()

        if self.name == "transpose":
            x = args[0]
            permutation = (
                args[1] if len(args) > 1 else kwargs.pop("permutation")
            )
            x = _validate_tensor(self.name, x)
            permutation = _normalize_permutation(permutation, x.dim())
            transpose_output_shape = tuple(
                int(x.shape[dim]) for dim in permutation
            )
            output = torch.empty(
                transpose_output_shape, device=x.device, dtype=x.dtype
            )
            return (x,), output, permutation

        if self.name == "slice":
            x = args[0]
            slices = args[1] if len(args) > 1 else kwargs.pop("slices", ())
            x = _validate_tensor(self.name, x)
            normalized = _normalize_slices(slices, x.shape)
            starts, ends, axes, steps, slice_output_shape = normalized
            output = torch.empty(
                slice_output_shape, device=x.device, dtype=x.dtype
            )
            return (x,), output, starts + ends + axes + steps

        if self.name == "concatenate":
            values = tuple(args[0])
            axis = kwargs.pop("axis", args[1] if len(args) > 1 else None)
            if axis is None:
                raise TypeError("concatenate requires axis")
            if not 1 <= len(values) <= 3:
                raise ValueError(
                    "aclnn concatenate supports one through three inputs"
                )
            tensors = tuple(
                _validate_tensor(self.name, item) for item in values
            )
            first = tensors[0]
            axis = _normalize_axis(int(axis), first.dim())
            concatenate_output_shape = list(first.shape)
            concatenate_output_shape[axis] = 0
            for item in tensors:
                if item.device != first.device or item.dtype != first.dtype:
                    raise ValueError(
                        "concatenate inputs must share device and dtype"
                    )
                if item.dim() != first.dim():
                    raise ValueError("concatenate inputs must share rank")
                for dim in range(first.dim()):
                    if dim != axis and item.shape[dim] != first.shape[dim]:
                        raise ValueError(
                            "concatenate non-axis dimensions must match"
                        )
                concatenate_output_shape[axis] += int(item.shape[axis])
            output = torch.empty(
                tuple(concatenate_output_shape),
                device=first.device,
                dtype=first.dtype,
            )
            return tensors, output, (axis,)

        x = args[0]
        axis = args[1] if len(args) > 1 else kwargs.pop("axis")
        x = _validate_tensor(self.name, x)
        axis = _normalize_axis(int(axis), x.dim())
        requested_dtype = kwargs.pop("compute_data_type", None)
        if requested_dtype not in (None, x.dtype):
            raise TypeError(
                "aclnn gen_index currently requires output dtype to match "
                "input"
            )
        scratch = torch.empty(
            (int(x.shape[axis]),), device=x.device, dtype=x.dtype
        )
        output = torch.empty(tuple(x.shape), device=x.device, dtype=x.dtype)
        return (x, scratch), output, (axis,)

    def prepare(self, *args: Any, **kwargs: Any) -> PreparedAclnnOperation:
        call_kwargs = dict(kwargs)
        tensors, output, parameters = self._prepare_arguments(
            args, call_kwargs
        )
        if call_kwargs:
            unknown = ", ".join(sorted(call_kwargs))
            raise TypeError(f"unexpected {self.name} arguments: {unknown}")
        context = self._context
        npu = context.npu()
        device = tensors[0].device
        with npu.device(device):
            stream_pointer = int(npu.current_stream(device=device).npu_stream)
            if stream_pointer == 0:
                raise RuntimeError("torch_npu returned a null current stream")
            metadata = [
                (
                    metadata_array(tensor.shape),
                    metadata_array(tensor.stride()),
                )
                for tensor in tensors
            ]
            while len(metadata) < 3:
                metadata.append((None, None))
            padded_tensors: list[torch.Tensor | None] = list(tensors)
            padded_tensors.extend([None] * (3 - len(padded_tensors)))
            output_shape = metadata_array(output.shape)
            output_stride = metadata_array(output.stride())
            parameter_values = (
                metadata_array(parameters) if parameters else None
            )
            handle = ctypes.c_void_p()
            error_buffer = ctypes.create_string_buffer(ERROR_BUFFER_SIZE)
            library = context.get_library(configure_utility)
            tensor_arguments: list[Any] = []
            for tensor, tensor_metadata in zip(padded_tensors, metadata):
                tensor_arguments.extend(
                    (
                        ctypes.c_void_p(
                            0 if tensor is None else int(tensor.data_ptr())
                        ),
                        tensor_metadata[0],
                        tensor_metadata[1],
                        ctypes.c_uint64(0 if tensor is None else tensor.dim()),
                    )
                )
            status = library.flagdnn_aclnn_utility_create(
                ctypes.c_int32(self._operation_code),
                *tensor_arguments,
                ctypes.c_uint64(len(tensors)),
                ctypes.c_void_p(int(output.data_ptr())),
                output_shape,
                output_stride,
                ctypes.c_uint64(output.dim()),
                ctypes.c_int32(DTYPE_CODES[output.dtype]),
                parameter_values,
                ctypes.c_uint64(len(parameters)),
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
                f"shapes={[tuple(item.shape) for item in tensors]}, "
                f"dtype={output.dtype}, device={device}"
            )
        context.last_device = device
        return PreparedAclnnOperation(
            library,
            handle,
            tensors,
            output,
            operation_name=f"aclnn{self.name.title()}",
            symbol_name="aclnn_utility",
        )

    def run(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        prepared = self.prepare(*args, **kwargs)
        try:
            return prepared.run()
        finally:
            prepared.close()


def create_utility_operations(
    context: AscendContext,
) -> tuple[Any, ...]:
    return (
        AscendIdentityOperation(context),
        AscendReshapeOperation(context),
        *(
            AscendUtilityOperation(name, context)
            for name in UTILITY_OPERATION_CODES
        ),
    )
