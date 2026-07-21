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
import ctypes.util
import os
import sys
from typing import Any, Iterable

import torch

from .common import CUDNN_COMPARE_DTYPES, NvidiaContext


_CUDNN_DATA_TYPES = {
    torch.float16: 2,
    torch.float32: 0,
    torch.bfloat16: 9,
}
_CUDNN_BATCHNORM_SPATIAL = 1
_LIBRARY: Any = None


class CudnnLegacyError(RuntimeError):
    pass


def _candidate_library_paths() -> Iterable[str]:
    seen = set()
    for base in sys.path:
        if not base:
            continue
        path = os.path.join(base, "nvidia", "cudnn", "lib", "libcudnn.so.9")
        if path not in seen:
            seen.add(path)
            yield path
    found = ctypes.util.find_library("cudnn")
    if found and found not in seen:
        seen.add(found)
        yield found
    for name in ("libcudnn.so.9", "libcudnn.so"):
        if name not in seen:
            yield name


def _configure_api(library: Any) -> None:
    library.cudnnGetErrorString.argtypes = [ctypes.c_int]
    library.cudnnGetErrorString.restype = ctypes.c_char_p
    library.cudnnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.cudnnCreate.restype = ctypes.c_int
    library.cudnnDestroy.argtypes = [ctypes.c_void_p]
    library.cudnnDestroy.restype = ctypes.c_int
    library.cudnnSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.cudnnSetStream.restype = ctypes.c_int
    library.cudnnCreateTensorDescriptor.argtypes = [
        ctypes.POINTER(ctypes.c_void_p)
    ]
    library.cudnnCreateTensorDescriptor.restype = ctypes.c_int
    library.cudnnDestroyTensorDescriptor.argtypes = [ctypes.c_void_p]
    library.cudnnDestroyTensorDescriptor.restype = ctypes.c_int
    library.cudnnSetTensor4dDescriptorEx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.cudnnSetTensor4dDescriptorEx.restype = ctypes.c_int
    library.cudnnBatchNormalizationForwardInference.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_double,
    ]
    library.cudnnBatchNormalizationForwardInference.restype = ctypes.c_int


def _load_library() -> Any:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    errors = []
    for path in _candidate_library_paths():
        try:
            library = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        _configure_api(library)
        _LIBRARY = library
        return library
    detail = "; ".join(errors) if errors else "no candidates found"
    raise CudnnLegacyError(f"failed to load libcudnn: {detail}")


def _check(status: int, name: str) -> None:
    if status == 0:
        return
    message = _load_library().cudnnGetErrorString(status).decode("utf-8")
    raise CudnnLegacyError(f"{name} failed: {message} ({status})")


def _data_type(dtype: torch.dtype) -> int:
    try:
        return _CUDNN_DATA_TYPES[dtype]
    except KeyError as exc:
        raise CudnnLegacyError(f"unsupported cuDNN dtype: {dtype}") from exc


def _channel_tensor(
    tensor: torch.Tensor, channels: int, name: str
) -> torch.Tensor:
    if tensor.numel() != channels:
        raise CudnnLegacyError(
            f"{name} must contain {channels} values, got {tensor.numel()}"
        )
    if tensor.device.type != "cuda":
        raise CudnnLegacyError(f"{name} must be a CUDA tensor")
    return (
        tensor.to(dtype=torch.float32).contiguous().reshape(1, channels, 1, 1)
    )


class PreparedCudnnBatchNormInference:
    def __init__(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        inv_variance: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        if x.ndim != 4:
            raise CudnnLegacyError(
                "cuDNN batchnorm_inference reference expects 4D NCHW"
            )
        self._library = _load_library()
        self._x = x
        self.output = torch.empty_strided(
            tuple(x.shape), tuple(x.stride()), device=x.device, dtype=x.dtype
        )
        channels = int(x.shape[1])
        self._mean = _channel_tensor(mean, channels, "mean")
        inv = _channel_tensor(inv_variance, channels, "inv_variance")
        self._variance = inv.reciprocal().square().contiguous()
        self._scale = _channel_tensor(scale, channels, "scale")
        self._bias = _channel_tensor(bias, channels, "bias")
        self._handle = ctypes.c_void_p()
        self._descriptors: list[ctypes.c_void_p] = []
        self._closed = False
        _check(
            self._library.cudnnCreate(ctypes.byref(self._handle)),
            "cudnnCreate",
        )
        try:
            self._x_descriptor = self._create_descriptor(self._x)
            self._y_descriptor = self._create_descriptor(self.output)
            self._parameter_descriptor = self._create_descriptor(self._scale)
        except Exception:
            self.close()
            raise
        self._alpha = ctypes.c_float(1.0)
        self._beta = ctypes.c_float(0.0)

    def _create_descriptor(self, tensor: torch.Tensor) -> ctypes.c_void_p:
        descriptor = ctypes.c_void_p()
        _check(
            self._library.cudnnCreateTensorDescriptor(
                ctypes.byref(descriptor)
            ),
            "cudnnCreateTensorDescriptor",
        )
        self._descriptors.append(descriptor)
        n, c, h, w = (int(value) for value in tensor.shape)
        ns, cs, hs, ws = (int(value) for value in tensor.stride())
        _check(
            self._library.cudnnSetTensor4dDescriptorEx(
                descriptor,
                _data_type(tensor.dtype),
                n,
                c,
                h,
                w,
                ns,
                cs,
                hs,
                ws,
            ),
            "cudnnSetTensor4dDescriptorEx",
        )
        return descriptor

    def run(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("prepared cuDNN batchnorm_inference is closed")
        stream = ctypes.c_void_p(
            torch.cuda.current_stream(device=self._x.device).cuda_stream
        )
        _check(
            self._library.cudnnSetStream(self._handle, stream),
            "cudnnSetStream",
        )
        _check(
            self._library.cudnnBatchNormalizationForwardInference(
                self._handle,
                _CUDNN_BATCHNORM_SPATIAL,
                ctypes.byref(self._alpha),
                ctypes.byref(self._beta),
                self._x_descriptor,
                ctypes.c_void_p(self._x.data_ptr()),
                self._y_descriptor,
                ctypes.c_void_p(self.output.data_ptr()),
                self._parameter_descriptor,
                ctypes.c_void_p(self._scale.data_ptr()),
                ctypes.c_void_p(self._bias.data_ptr()),
                ctypes.c_void_p(self._mean.data_ptr()),
                ctypes.c_void_p(self._variance.data_ptr()),
                ctypes.c_double(0.0),
            ),
            "cudnnBatchNormalizationForwardInference",
        )
        return self.output

    def __call__(self) -> torch.Tensor:
        return self.run()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in self._descriptors:
            if descriptor:
                self._library.cudnnDestroyTensorDescriptor(descriptor)
        self._descriptors = []
        if self._handle:
            self._library.cudnnDestroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class NvidiaBatchNormInferenceOperation:
    name = "batchnorm_inference"

    def __init__(self, context: NvidiaContext) -> None:
        self._context = context

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in CUDNN_COMPARE_DTYPES

    def prepare(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        inv_variance: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        **_: Any,
    ) -> PreparedCudnnBatchNormInference:
        for tensor in (x, mean, inv_variance, scale, bias):
            self._context.validate_tensor(self.name, tensor)
        if any(
            tensor.device != x.device
            for tensor in (mean, inv_variance, scale, bias)
        ):
            raise ValueError(
                "cuDNN batchnorm_inference tensors must share a device"
            )
        with torch.cuda.device(x.device):
            prepared = PreparedCudnnBatchNormInference(
                x, mean, inv_variance, scale, bias
            )
        self._context.last_device = x.device
        return prepared

    def run(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        prepared = self.prepare(*args, **kwargs)
        try:
            output = prepared.run()
            self._context.synchronize()
            return output
        finally:
            prepared.close()
