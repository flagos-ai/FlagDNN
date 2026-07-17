from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Union

import torch

from .build import build_aclnn_oracle, find_cann_layout


_DTYPE_CODES = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}
_ERROR_BUFFER_SIZE = 4096
_INT64_POINTER = ctypes.POINTER(ctypes.c_int64)
_Number = Union[int, float]


def _configure_add(library: Any) -> None:
    function = library.flagdnn_test_aclnn_add
    function.argtypes = [
        ctypes.c_void_p,
        _INT64_POINTER,
        _INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_void_p,
        _INT64_POINTER,
        _INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_void_p,
        _INT64_POINTER,
        _INT64_POINTER,
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
        _INT64_POINTER,
        _INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_void_p,
        _INT64_POINTER,
        _INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_void_p,
        _INT64_POINTER,
        _INT64_POINTER,
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
        prepared_function = getattr(library, name)
        prepared_function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        prepared_function.restype = ctypes.c_int


def _configure_abs(library: Any) -> None:
    function = library.flagdnn_test_aclnn_abs
    function.argtypes = [
        ctypes.c_void_p,
        _INT64_POINTER,
        _INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_void_p,
        _INT64_POINTER,
        _INT64_POINTER,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int


def _configure_library(library: Any) -> Any:
    _configure_add(library)
    _configure_abs(library)
    return library


def _load_cdll(path: Path, mode: int, description: str) -> Any:
    try:
        return ctypes.CDLL(str(path), mode=mode)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load {description} from {path}: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def _load_library() -> Any:
    layout = find_cann_layout()
    mode = os.RTLD_NOW | os.RTLD_GLOBAL
    dependencies = (
        _load_cdll(
            layout.lib_dir / "libascendcl.so",
            mode,
            "CANN libascendcl.so",
        ),
        _load_cdll(
            layout.lib_dir / "libopapi.so",
            mode,
            "CANN libopapi.so",
        ),
    )
    wrapper_path = build_aclnn_oracle(layout=layout)
    library = _configure_library(
        _load_cdll(wrapper_path, mode, "ACLNN test wrapper")
    )
    library._flagdnn_dependency_handles = dependencies
    return library


def _metadata_array(values: Iterable[int]) -> Any:
    items = tuple(int(value) for value in values)
    return (ctypes.c_int64 * len(items))(*items)


def _npu_module() -> Any:
    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError(
            "torch_npu has not registered torch.npu for the Ascend oracle"
        )
    return npu


def _normalize_alpha(alpha: _Number) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(
            "aclnnAdd oracle alpha must be an int or float, "
            f"got {type(alpha).__name__}"
        )
    try:
        return float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "aclnnAdd oracle alpha cannot be represented as a double"
        ) from exc


class PreparedAclnnAdd:
    def __init__(
        self,
        library: Any,
        handle: ctypes.c_void_p,
        x: torch.Tensor,
        y: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        self._library = library
        self._handle = handle
        self._x = x
        self._y = y
        self.output = output

    def run(self) -> torch.Tensor:
        handle = self._handle
        if handle.value is None:
            raise RuntimeError("prepared ACLNN Add runner is closed")
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        status = self._library.flagdnn_aclnn_add_run(
            handle,
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
        if status != 0:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                "prepared aclnnAdd execution failed: "
                f"status={status}, detail={detail}"
            )
        return self.output

    def __call__(self) -> torch.Tensor:
        return self.run()

    def close(self) -> None:
        handle = self._handle
        if handle.value is None:
            return
        self._handle = ctypes.c_void_p()
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        status = self._library.flagdnn_aclnn_add_destroy(
            handle,
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
        if status != 0:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                "prepared aclnnAdd cleanup failed: "
                f"status={status}, detail={detail}"
            )


class AscendDnnProvider:
    vendor_name = "ascend"
    implementation = "aclnn"
    display_name = "ACLNN"

    def __init__(
        self,
        library_loader: Callable[[], Any] = _load_library,
    ) -> None:
        self._library_loader = library_loader
        self._library: Any = None
        self._last_device: Any = None

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in _DTYPE_CODES

    def _get_library(self) -> Any:
        if self._library is None:
            self._library = self._library_loader()
        return self._library

    def _validate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("aclnnAdd oracle expects two torch.Tensor inputs")
        if x.layout != torch.strided or y.layout != torch.strided:
            raise ValueError(
                "aclnnAdd oracle requires strided tensors, "
                f"got {x.layout} and {y.layout}"
            )
        if x.device.type != "npu" or y.device.type != "npu":
            raise ValueError(
                "aclnnAdd oracle requires NPU tensors, "
                f"got {x.device} and {y.device}"
            )
        if x.device != y.device:
            raise ValueError(
                "aclnnAdd oracle inputs must be on the same NPU, "
                f"got {x.device} and {y.device}"
            )
        if x.dtype != y.dtype:
            raise TypeError(
                "aclnnAdd oracle inputs must have the same dtype, "
                f"got {x.dtype} and {y.dtype}"
            )
        if not self.supports_dtype(x.dtype):
            raise TypeError(f"aclnnAdd oracle does not support {x.dtype}")
        if x.dim() == 0 or y.dim() == 0:
            raise ValueError("aclnnAdd oracle does not support rank-0 tensors")
        if x.storage_offset() != 0 or y.storage_offset() != 0:
            raise ValueError(
                "aclnnAdd oracle requires zero storage offsets, "
                f"got {x.storage_offset()} and {y.storage_offset()}"
            )

    def _validate_unary(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("aclnnAbs oracle expects a torch.Tensor input")
        if x.layout != torch.strided:
            raise ValueError(
                "aclnnAbs oracle requires a strided tensor, " f"got {x.layout}"
            )
        if x.device.type != "npu":
            raise ValueError(
                "aclnnAbs oracle requires an NPU tensor, " f"got {x.device}"
            )
        if not self.supports_dtype(x.dtype):
            raise TypeError(f"aclnnAbs oracle does not support {x.dtype}")
        if x.dim() == 0:
            raise ValueError("aclnnAbs oracle does not support rank-0 tensors")
        if x.dim() > 8:
            raise ValueError(
                "aclnnAbs oracle requires tensor rank from 1 through 8, "
                f"got {x.dim()}"
            )
        if x.storage_offset() != 0:
            raise ValueError(
                "aclnnAbs oracle requires a zero storage offset, "
                f"got {x.storage_offset()}"
            )
        is_channels_last = x.dim() == 4 and x.is_contiguous(
            memory_format=torch.channels_last
        )
        if not x.is_contiguous() and not is_channels_last:
            raise ValueError(
                "aclnnAbs oracle requires a contiguous or 4D "
                "channels-last dense tensor"
            )

    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: _Number = 1,
    ) -> torch.Tensor:
        alpha_value = _normalize_alpha(alpha)
        self._validate(x, y)
        try:
            output_shape = tuple(torch.broadcast_shapes(x.shape, y.shape))
        except RuntimeError as exc:
            raise ValueError(
                "aclnnAdd oracle inputs are not broadcastable: "
                f"x_shape={tuple(x.shape)}, y_shape={tuple(y.shape)}"
            ) from exc

        npu = _npu_module()
        with npu.device(x.device):
            output = torch.empty(
                output_shape,
                device=x.device,
                dtype=x.dtype,
            )
            x_shape = _metadata_array(x.shape)
            x_strides = _metadata_array(x.stride())
            y_shape = _metadata_array(y.shape)
            y_strides = _metadata_array(y.stride())
            output_shape_data = _metadata_array(output.shape)
            output_strides = _metadata_array(output.stride())
            x_pointer = int(x.data_ptr())
            y_pointer = int(y.data_ptr())
            output_pointer = int(output.data_ptr())
            if x_pointer == 0 or y_pointer == 0 or output_pointer == 0:
                raise ValueError(
                    "aclnnAdd oracle received a null tensor data pointer"
                )
            stream = npu.current_stream(device=x.device)
            stream_pointer = int(stream.npu_stream)
            if stream_pointer == 0:
                raise RuntimeError("torch_npu returned a null current stream")

            error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
            library = self._get_library()
            status = library.flagdnn_test_aclnn_add(
                ctypes.c_void_p(x_pointer),
                x_shape,
                x_strides,
                ctypes.c_uint64(x.dim()),
                ctypes.c_void_p(y_pointer),
                y_shape,
                y_strides,
                ctypes.c_uint64(y.dim()),
                ctypes.c_void_p(output_pointer),
                output_shape_data,
                output_strides,
                ctypes.c_uint64(output.dim()),
                ctypes.c_int32(_DTYPE_CODES[x.dtype]),
                ctypes.c_double(alpha_value),
                ctypes.c_void_p(stream_pointer),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
        if status != 0:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                "aclnnAdd oracle failed: "
                f"status={status}, detail={detail}, "
                f"x_shape={tuple(x.shape)}, x_stride={tuple(x.stride())}, "
                f"y_shape={tuple(y.shape)}, y_stride={tuple(y.stride())}, "
                f"output_shape={tuple(output.shape)}, "
                f"output_stride={tuple(output.stride())}, "
                f"dtype={x.dtype}, device={x.device}, alpha={alpha}, "
                f"stream=0x{stream_pointer:x}"
            )
        self._last_device = x.device
        return output

    def prepare_add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: _Number = 1,
    ) -> PreparedAclnnAdd:
        alpha_value = _normalize_alpha(alpha)
        self._validate(x, y)
        if x.dim() > 8 or y.dim() > 8:
            raise ValueError(
                "prepared aclnnAdd requires input tensor rank from 1 through 8"
            )
        try:
            output_shape = tuple(torch.broadcast_shapes(x.shape, y.shape))
        except RuntimeError as exc:
            raise ValueError(
                "aclnnAdd inputs are not broadcastable: "
                f"x_shape={tuple(x.shape)}, y_shape={tuple(y.shape)}"
            ) from exc
        if len(output_shape) > 8:
            raise ValueError(
                "prepared aclnnAdd requires output tensor rank through 8"
            )

        npu = _npu_module()
        with npu.device(x.device):
            output = torch.empty(
                output_shape,
                device=x.device,
                dtype=x.dtype,
            )
            x_shape = _metadata_array(x.shape)
            x_strides = _metadata_array(x.stride())
            y_shape = _metadata_array(y.shape)
            y_strides = _metadata_array(y.stride())
            output_shape_data = _metadata_array(output.shape)
            output_strides = _metadata_array(output.stride())
            pointers = (x.data_ptr(), y.data_ptr(), output.data_ptr())
            if any(int(pointer) == 0 for pointer in pointers):
                raise ValueError(
                    "prepared aclnnAdd received a null tensor data pointer"
                )
            stream = npu.current_stream(device=x.device)
            stream_pointer = int(stream.npu_stream)
            if stream_pointer == 0:
                raise RuntimeError("torch_npu returned a null current stream")

            handle = ctypes.c_void_p()
            error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
            library = self._get_library()
            status = library.flagdnn_aclnn_add_create(
                ctypes.c_void_p(int(pointers[0])),
                x_shape,
                x_strides,
                ctypes.c_uint64(x.dim()),
                ctypes.c_void_p(int(pointers[1])),
                y_shape,
                y_strides,
                ctypes.c_uint64(y.dim()),
                ctypes.c_void_p(int(pointers[2])),
                output_shape_data,
                output_strides,
                ctypes.c_uint64(output.dim()),
                ctypes.c_int32(_DTYPE_CODES[x.dtype]),
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
        self._last_device = x.device
        return PreparedAclnnAdd(library, handle, x, y, output)

    def abs(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_unary(x)

        npu = _npu_module()
        with npu.device(x.device):
            output = torch.empty_strided(
                tuple(x.shape),
                tuple(x.stride()),
                device=x.device,
                dtype=x.dtype,
            )
            x_shape = _metadata_array(x.shape)
            x_strides = _metadata_array(x.stride())
            output_shape = _metadata_array(output.shape)
            output_strides = _metadata_array(output.stride())
            x_pointer = int(x.data_ptr())
            output_pointer = int(output.data_ptr())
            if x_pointer == 0 or output_pointer == 0:
                raise ValueError(
                    "aclnnAbs oracle received a null tensor data pointer"
                )
            stream = npu.current_stream(device=x.device)
            stream_pointer = int(stream.npu_stream)
            if stream_pointer == 0:
                raise RuntimeError("torch_npu returned a null current stream")

            error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
            library = self._get_library()
            status = library.flagdnn_test_aclnn_abs(
                ctypes.c_void_p(x_pointer),
                x_shape,
                x_strides,
                ctypes.c_uint64(x.dim()),
                ctypes.c_void_p(output_pointer),
                output_shape,
                output_strides,
                ctypes.c_uint64(output.dim()),
                ctypes.c_int32(_DTYPE_CODES[x.dtype]),
                ctypes.c_void_p(stream_pointer),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
        if status != 0:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                "aclnnAbs oracle failed: "
                f"status={status}, detail={detail}, "
                f"x_shape={tuple(x.shape)}, x_stride={tuple(x.stride())}, "
                f"output_shape={tuple(output.shape)}, "
                f"output_stride={tuple(output.stride())}, "
                f"dtype={x.dtype}, device={x.device}, "
                f"stream=0x{stream_pointer:x}"
            )
        self._last_device = x.device
        return output

    def synchronize(self) -> None:
        if self._last_device is not None:
            _npu_module().synchronize(device=self._last_device)

    def close(self) -> None:
        return None


def create_provider() -> AscendDnnProvider:
    return AscendDnnProvider()


class AscendDnnOracle(AscendDnnProvider):
    """Compatibility name for existing correctness-test integrations."""

    implementation = "aclnnAdd"


def create_oracle() -> AscendDnnOracle:
    return AscendDnnOracle()
