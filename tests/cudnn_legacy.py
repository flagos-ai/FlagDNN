from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from typing import Iterable

import torch


class CudnnLegacyError(RuntimeError):
    pass


_CUDNN_DATA_TYPES = {
    torch.float16: 2,
    torch.float32: 0,
    torch.bfloat16: 9,
}
_CUDNN_BATCHNORM_SPATIAL = 1
_LIB = None


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
        yield found

    for name in ("libcudnn.so.9", "libcudnn.so"):
        if name not in seen:
            yield name


def _load_cudnn():
    global _LIB
    if _LIB is not None:
        return _LIB

    errors = []
    for path in _candidate_library_paths():
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        _configure_cudnn_api(lib)
        _LIB = lib
        return lib

    detail = "; ".join(errors) if errors else "no candidates found"
    raise CudnnLegacyError(f"failed to load libcudnn: {detail}")


def _configure_cudnn_api(lib) -> None:
    lib.cudnnGetErrorString.argtypes = [ctypes.c_int]
    lib.cudnnGetErrorString.restype = ctypes.c_char_p

    lib.cudnnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cudnnCreate.restype = ctypes.c_int
    lib.cudnnDestroy.argtypes = [ctypes.c_void_p]
    lib.cudnnDestroy.restype = ctypes.c_int
    lib.cudnnSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cudnnSetStream.restype = ctypes.c_int

    lib.cudnnCreateTensorDescriptor.argtypes = [
        ctypes.POINTER(ctypes.c_void_p)
    ]
    lib.cudnnCreateTensorDescriptor.restype = ctypes.c_int
    lib.cudnnDestroyTensorDescriptor.argtypes = [ctypes.c_void_p]
    lib.cudnnDestroyTensorDescriptor.restype = ctypes.c_int
    lib.cudnnSetTensor4dDescriptorEx.argtypes = [
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
    lib.cudnnSetTensor4dDescriptorEx.restype = ctypes.c_int

    lib.cudnnBatchNormalizationForwardInference.argtypes = [
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
    lib.cudnnBatchNormalizationForwardInference.restype = ctypes.c_int


def _check(status: int, name: str) -> None:
    if status == 0:
        return
    lib = _load_cudnn()
    message = lib.cudnnGetErrorString(status).decode("utf-8")
    raise CudnnLegacyError(f"{name} failed: {message} ({status})")


def _data_type(dtype: torch.dtype) -> int:
    try:
        return _CUDNN_DATA_TYPES[dtype]
    except KeyError as exc:
        raise CudnnLegacyError(f"unsupported cuDNN dtype: {dtype}") from exc


def _as_channel_tensor(
    tensor: torch.Tensor, channels: int, name: str
) -> torch.Tensor:
    if tensor.numel() != channels:
        raise CudnnLegacyError(
            f"{name} must contain {channels} values, got {tensor.numel()}"
        )
    if not tensor.is_cuda:
        raise CudnnLegacyError(f"{name} must be a CUDA tensor")
    return (
        tensor.to(dtype=torch.float32).contiguous().reshape(1, channels, 1, 1)
    )


class CudnnBatchNormInference:
    def __init__(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        inv_variance: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        if not x.is_cuda:
            raise CudnnLegacyError("x must be a CUDA tensor")
        if x.ndim != 4:
            raise CudnnLegacyError(
                "legacy cuDNN batchnorm_inference runner expects 4D NCHW"
            )

        self.lib = _load_cudnn()
        self.x = x
        self.y = torch.empty_strided(
            tuple(x.shape), tuple(x.stride()), device=x.device, dtype=x.dtype
        )
        channels = int(x.shape[1])
        self.mean = _as_channel_tensor(mean, channels, "mean")
        self.inv_variance = _as_channel_tensor(
            inv_variance, channels, "inv_variance"
        )
        self.scale = _as_channel_tensor(scale, channels, "scale")
        self.bias = _as_channel_tensor(bias, channels, "bias")
        # Legacy cuDNN accepts variance plus epsilon. Use epsilon=0 and convert
        # the cuDNN Frontend-style inv_variance input once at runner setup time.
        self.variance = self.inv_variance.reciprocal().square().contiguous()

        self.handle = ctypes.c_void_p()
        self.descs: list[ctypes.c_void_p] = []
        _check(self.lib.cudnnCreate(ctypes.byref(self.handle)), "cudnnCreate")
        self.x_desc = self._create_tensor_desc(self.x)
        self.y_desc = self._create_tensor_desc(self.y)
        self.param_desc = self._create_tensor_desc(self.scale)
        self.alpha = ctypes.c_float(1.0)
        self.beta = ctypes.c_float(0.0)

    def _create_tensor_desc(self, tensor: torch.Tensor) -> ctypes.c_void_p:
        desc = ctypes.c_void_p()
        _check(
            self.lib.cudnnCreateTensorDescriptor(ctypes.byref(desc)),
            "cudnnCreateTensorDescriptor",
        )
        self.descs.append(desc)
        n, c, h, w = (int(v) for v in tensor.shape)
        ns, cs, hs, ws = (int(v) for v in tensor.stride())
        _check(
            self.lib.cudnnSetTensor4dDescriptorEx(
                desc,
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
        return desc

    def __call__(self) -> torch.Tensor:
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check(
            self.lib.cudnnSetStream(self.handle, stream),
            "cudnnSetStream",
        )
        _check(
            self.lib.cudnnBatchNormalizationForwardInference(
                self.handle,
                _CUDNN_BATCHNORM_SPATIAL,
                ctypes.byref(self.alpha),
                ctypes.byref(self.beta),
                self.x_desc,
                ctypes.c_void_p(self.x.data_ptr()),
                self.y_desc,
                ctypes.c_void_p(self.y.data_ptr()),
                self.param_desc,
                ctypes.c_void_p(self.scale.data_ptr()),
                ctypes.c_void_p(self.bias.data_ptr()),
                ctypes.c_void_p(self.mean.data_ptr()),
                ctypes.c_void_p(self.variance.data_ptr()),
                ctypes.c_double(0.0),
            ),
            "cudnnBatchNormalizationForwardInference",
        )
        return self.y

    def close(self) -> None:
        if getattr(self, "lib", None) is None:
            return
        for desc in getattr(self, "descs", []):
            if desc:
                self.lib.cudnnDestroyTensorDescriptor(desc)
        self.descs = []
        handle = getattr(self, "handle", None)
        if handle:
            self.lib.cudnnDestroy(handle)
            self.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def cudnn_batchnorm_inference(
    x: torch.Tensor,
    mean: torch.Tensor,
    inv_variance: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    runner = CudnnBatchNormInference(x, mean, inv_variance, scale, bias)
    try:
        out = runner()
        torch.cuda.synchronize()
        return out
    finally:
        runner.close()
