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
from typing import Any, Iterable, Sequence

import torch

from .add import AscendAddOperation
from .binary import AscendBinaryOperation
from ..common import (
    DTYPE_CODES,
    ERROR_BUFFER_SIZE,
    INT64_POINTER,
    AscendContext,
    PreparedAclnnOperation,
    metadata_array,
)


class _NormTensorArgument(ctypes.Structure):
    _fields_ = (
        ("data", ctypes.c_void_p),
        ("shape", INT64_POINTER),
        ("strides", INT64_POINTER),
        ("rank", ctypes.c_uint64),
        ("dtype_code", ctypes.c_int32),
    )


def configure_norm(library: Any) -> None:
    create = library.flagdnn_aclnn_norm_create
    create.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(_NormTensorArgument),
        ctypes.c_uint64,
        ctypes.POINTER(_NormTensorArgument),
        ctypes.c_uint64,
        INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    create.restype = ctypes.c_int
    for name in (
        "flagdnn_aclnn_norm_run",
        "flagdnn_aclnn_norm_destroy",
    ):
        function = getattr(library, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int


def _validate_tensor(name: str, tensor: Any) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "npu":
        raise TypeError(f"aclnn {name} requires NPU tensor inputs")
    if tensor.layout != torch.strided or tensor.storage_offset() != 0:
        raise ValueError(f"aclnn {name} requires zero-offset strided tensors")
    if tensor.dtype not in DTYPE_CODES:
        raise TypeError(f"aclnn {name} does not support {tensor.dtype}")
    if tensor.dim() == 0 or tensor.dim() > 8 or tensor.numel() == 0:
        raise ValueError(
            f"aclnn {name} requires non-empty tensors of rank 1..8"
        )
    return tensor


def _arguments(
    tensors: Sequence[torch.Tensor],
) -> tuple[Any, tuple[Any, ...]]:
    metadata: list[Any] = []
    values = []
    for tensor in tensors:
        shape = metadata_array(tensor.shape)
        strides = metadata_array(tensor.stride())
        metadata.extend((shape, strides))
        values.append(
            _NormTensorArgument(
                ctypes.c_void_p(int(tensor.data_ptr())),
                shape,
                strides,
                ctypes.c_uint64(tensor.dim()),
                ctypes.c_int32(DTYPE_CODES[tensor.dtype]),
            )
        )
    array_type = _NormTensorArgument * len(values)
    return array_type(*values), tuple(metadata)


def _prepare_native_norm(
    context: AscendContext,
    operation_code: int,
    operation_name: str,
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    *,
    int_parameters: Iterable[int] = (),
    parameter0: float = 0.0,
    parameter1: float = 0.0,
) -> PreparedAclnnOperation:
    device = inputs[0].device
    if any(item.device != device for item in (*inputs, *outputs)):
        raise ValueError(f"aclnn {operation_name} tensors must share a device")
    input_arguments, input_metadata = _arguments(inputs)
    output_arguments, output_metadata = _arguments(outputs)
    parameters = tuple(int(value) for value in int_parameters)
    parameter_array = metadata_array(parameters) if parameters else None
    npu = context.npu()
    with npu.device(device):
        stream_pointer = int(npu.current_stream(device=device).npu_stream)
        handle = ctypes.c_void_p()
        error_buffer = ctypes.create_string_buffer(ERROR_BUFFER_SIZE)
        library = context.get_library(configure_norm)
        status = library.flagdnn_aclnn_norm_create(
            ctypes.c_int32(operation_code),
            input_arguments,
            ctypes.c_uint64(len(inputs)),
            output_arguments,
            ctypes.c_uint64(len(outputs)),
            parameter_array,
            ctypes.c_uint64(len(parameters)),
            ctypes.c_double(parameter0),
            ctypes.c_double(parameter1),
            ctypes.c_void_p(stream_pointer),
            ctypes.byref(handle),
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
    del input_metadata, output_metadata
    if status != 0 or handle.value is None:
        detail = error_buffer.value.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"prepared aclnn {operation_name} creation failed: "
            f"status={status}, detail={detail}, "
            f"input_shapes={[tuple(item.shape) for item in inputs]}, "
            f"input_dtypes={[item.dtype for item in inputs]}, device={device}"
        )
    context.last_device = device
    output: Any = outputs[0] if len(outputs) == 1 else tuple(outputs)
    return PreparedAclnnOperation(
        library,
        handle,
        tuple(inputs),
        output,
        operation_name=operation_name,
        symbol_name="aclnn_norm",
    )


def _normalized_shape(x: torch.Tensor, scale: torch.Tensor) -> tuple[int, ...]:
    aligned = (1,) * (x.dim() - scale.dim()) + tuple(scale.shape)
    axes = tuple(index for index, size in enumerate(aligned) if size != 1)
    if not axes:
        axes = (x.dim() - 1,)
    trailing = tuple(range(x.dim() - len(axes), x.dim()))
    if axes != trailing:
        raise ValueError("norm scale must describe trailing dimensions")
    result = tuple(int(x.shape[index]) for index in axes)
    if scale.numel() != torch.Size(result).numel():
        raise ValueError("norm scale size does not match normalized shape")
    return result


class _PreparedCompositeNorm:
    def __init__(self, first: Any, second: Any, output: Any) -> None:
        self._first = first
        self._second = second
        self.output = output

    def run(self) -> Any:
        self._first.run()
        self._second.run()
        return self.output

    def __call__(self) -> Any:
        return self.run()

    def close(self) -> None:
        try:
            self._second.close()
        finally:
            self._first.close()


class _PreparedNormSequence:
    def __init__(self, prepared: Sequence[Any], output: Any) -> None:
        self._prepared = tuple(prepared)
        self.output = output

    def run(self) -> Any:
        for prepared in self._prepared:
            prepared.run()
        return self.output

    def __call__(self) -> Any:
        return self.run()

    def close(self) -> None:
        first_error: BaseException | None = None
        for prepared in reversed(self._prepared):
            try:
                prepared.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


class AscendLayerNormOperation:
    name = "layernorm"

    def __init__(self, context: AscendContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in DTYPE_CODES

    def prepare(
        self,
        norm_forward_phase: Any,
        x: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        epsilon: float,
        **_: Any,
    ) -> PreparedAclnnOperation:
        del norm_forward_phase
        x = _validate_tensor(self.name, x)
        scale = _validate_tensor(self.name, scale)
        bias = _validate_tensor(self.name, bias)
        normalized = _normalized_shape(x, scale)
        scale_flat = scale.reshape(-1)
        bias_flat = bias.reshape(-1)
        stat_shape = tuple(x.shape[: x.dim() - len(normalized)]) + (1,) * len(
            normalized
        )
        outputs = (
            torch.empty_like(x),
            torch.empty(stat_shape, device=x.device, dtype=torch.float32),
            torch.empty(stat_shape, device=x.device, dtype=torch.float32),
        )
        return _prepare_native_norm(
            self._context,
            0,
            "aclnnLayerNorm",
            (x, scale_flat, bias_flat),
            outputs,
            int_parameters=normalized,
            parameter0=float(epsilon),
        )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        prepared = self.prepare(*args, **kwargs)
        try:
            return prepared.run()
        finally:
            prepared.close()


class AscendRmsNormOperation:
    name = "rmsnorm"

    def __init__(self, context: AscendContext) -> None:
        self._context = context
        self._add = AscendAddOperation(context)

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in DTYPE_CODES

    def prepare(
        self,
        norm_forward_phase: Any,
        x: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        epsilon: float = 1e-5,
        **_: Any,
    ) -> Any:
        del norm_forward_phase
        x = _validate_tensor(self.name, x)
        scale = _validate_tensor(self.name, scale)
        normalized = _normalized_shape(x, scale)
        if len(normalized) != 1:
            raise ValueError("aclnnRmsNorm supports the last dimension")
        stat_shape = tuple(x.shape[:-1]) + (1,)
        native_outputs = (
            torch.empty_like(x),
            torch.empty(stat_shape, device=x.device, dtype=torch.float32),
        )
        native = _prepare_native_norm(
            self._context,
            1,
            "aclnnRmsNorm",
            (x, scale.reshape(-1)),
            native_outputs,
            parameter0=float(epsilon),
        )
        if bias is None:
            return native
        bias = _validate_tensor(self.name, bias)
        if bias.numel() != scale.numel():
            native.close()
            raise ValueError("rmsnorm bias size must match scale")
        try:
            add = self._add.prepare(native_outputs[0], bias.reshape(-1))
        except Exception:
            native.close()
            raise
        return _PreparedCompositeNorm(
            native,
            add,
            (add.output, native_outputs[1]),
        )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        prepared = self.prepare(*args, **kwargs)
        try:
            return prepared.run()
        finally:
            prepared.close()


class AscendBatchNormOperation:
    name = "batchnorm"

    def __init__(self, context: AscendContext) -> None:
        self._context = context
        self._add = AscendAddOperation(context)
        self._binary = {
            name: AscendBinaryOperation(name, context)
            for name in ("max", "mul", "pow", "sub")
        }

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        # aclnnBatchNormStats on the supported Ascend stack does not accept
        # BF16 inputs. Keep this restriction local to training BatchNorm.
        return dtype in (torch.float16, torch.float32)

    def prepare(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        epsilon: float,
        momentum: float,
        peer_stats: Sequence[torch.Tensor] | None = None,
        **_: Any,
    ) -> Any:
        x = _validate_tensor(self.name, x)
        scale = _validate_tensor(self.name, scale)
        bias = _validate_tensor(self.name, bias)
        running_mean = _validate_tensor(self.name, running_mean)
        running_var = _validate_tensor(self.name, running_var)
        peers = tuple(peer_stats or ())
        if len(peers) > 1:
            raise ValueError(
                "batchnorm supports at most one peer_stats tensor"
            )
        channels = int(x.shape[1])
        if any(
            item.numel() != channels
            for item in (scale, bias, running_mean, running_var)
        ):
            raise ValueError("batchnorm channel parameter size mismatch")
        if (
            running_mean.dtype != torch.float32
            or running_var.dtype != torch.float32
        ):
            raise TypeError(
                "aclnn batchnorm running statistics must be float32"
            )
        weight = scale.reshape(-1).to(dtype=torch.float32)
        bias_float = bias.reshape(-1).to(dtype=torch.float32)
        saved_mean = torch.empty(
            (channels,), device=x.device, dtype=torch.float32
        )
        saved_invstd = torch.empty_like(saved_mean)
        y = torch.empty_like(x)
        epsilon_value = float(epsilon)
        momentum_value = float(momentum)
        sample_count = x.numel() // channels
        correction = (
            sample_count / (sample_count - 1) if sample_count > 1 else 1.0
        )
        constant_values = {
            "epsilon": epsilon_value,
            "minus_two": -2.0,
            "momentum": momentum_value,
            "one_minus_momentum": 1.0 - momentum_value,
            "correction": correction,
            "zero": 0.0,
        }
        constants = {
            name: torch.full((1,), value, device=x.device, dtype=torch.float32)
            for name, value in constant_values.items()
        }
        prepared: list[Any] = []
        try:
            prepared.append(
                _prepare_native_norm(
                    self._context,
                    2,
                    "aclnnBatchNormStats",
                    (x,),
                    (saved_mean, saved_invstd),
                    parameter0=epsilon_value,
                )
            )
            prepared.append(
                _prepare_native_norm(
                    self._context,
                    3,
                    "aclnnBatchNormElemt",
                    (x, weight, bias_float, saved_mean, saved_invstd),
                    (y,),
                    parameter0=0.0,
                )
            )

            def binary(
                name: str, lhs: torch.Tensor, rhs: torch.Tensor
            ) -> torch.Tensor:
                operation = self._binary[name].prepare(lhs, rhs)
                prepared.append(operation)
                return operation.output

            variance = binary("pow", saved_invstd, constants["minus_two"])
            variance = binary("sub", variance, constants["epsilon"])
            variance = binary("max", variance, constants["zero"])
            if sample_count > 1:
                variance = binary("mul", variance, constants["correction"])
            previous_mean = binary(
                "mul",
                running_mean.reshape(-1),
                constants["one_minus_momentum"],
            )
            current_mean = binary("mul", saved_mean, constants["momentum"])
            mean_next_operation = self._add.prepare(
                previous_mean, current_mean
            )
            prepared.append(mean_next_operation)
            previous_var = binary(
                "mul",
                running_var.reshape(-1),
                constants["one_minus_momentum"],
            )
            current_var = binary("mul", variance, constants["momentum"])
            var_next_operation = self._add.prepare(previous_var, current_var)
            prepared.append(var_next_operation)
        except Exception:
            for operation in reversed(prepared):
                operation.close()
            raise
        stat_shape = tuple(running_mean.shape)
        output = (
            y,
            saved_mean.view(stat_shape),
            saved_invstd.view(stat_shape),
            mean_next_operation.output.view(stat_shape),
            var_next_operation.output.view(stat_shape),
        )
        return _PreparedNormSequence(prepared, output)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        prepared = self.prepare(*args, **kwargs)
        try:
            return prepared.run()
        finally:
            prepared.close()


class AscendBatchNormInferenceOperation:
    name = "batchnorm_inference"

    def __init__(self, context: AscendContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in DTYPE_CODES

    def prepare(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        inv_variance: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        **_: Any,
    ) -> PreparedAclnnOperation:
        tensors = tuple(
            _validate_tensor(self.name, item)
            for item in (x, scale, bias, mean, inv_variance)
        )
        channels = int(x.shape[1])
        if any(item.numel() != channels for item in tensors[1:]):
            raise ValueError(
                "batchnorm_inference channel parameter size mismatch"
            )
        output = torch.empty_like(x)
        return _prepare_native_norm(
            self._context,
            3,
            "aclnnBatchNormElemt",
            (
                x,
                scale.reshape(-1),
                bias.reshape(-1),
                mean.reshape(-1),
                inv_variance.reshape(-1),
            ),
            (output,),
            parameter0=0.0,
        )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        prepared = self.prepare(*args, **kwargs)
        try:
            return prepared.run()
        finally:
            prepared.close()


def create_norm_operations(context: AscendContext) -> tuple[Any, ...]:
    return (
        AscendLayerNormOperation(context),
        AscendRmsNormOperation(context),
        AscendBatchNormOperation(context),
        AscendBatchNormInferenceOperation(context),
    )
