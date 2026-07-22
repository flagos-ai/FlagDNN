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

from typing import Any, Callable

import torch

from devtools.dnn_reference.interfaces import (
    DnnProviderUnavailableError,
    DnnReferenceNotSupportedError,
)

try:
    import cudnn
except Exception as exc:
    raise DnnProviderUnavailableError(
        f"cuDNN frontend is unavailable for the NVIDIA reference: {exc}"
    ) from exc


CUDNN_COMPARE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def cudnn_data_type(dtype: torch.dtype) -> Any:
    if dtype == torch.float16:
        return cudnn.data_type.HALF
    if dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    if dtype == torch.float32:
        return cudnn.data_type.FLOAT
    if dtype == torch.bool:
        return cudnn.data_type.BOOLEAN
    raise TypeError(f"Unsupported dtype for cuDNN frontend: {dtype}")


def cudnn_graph(dtype: torch.dtype, handle: Any) -> Any:
    return cudnn.pygraph(
        io_data_type=cudnn_data_type(dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )


def _is_non_overlapping_and_dense(tensor: torch.Tensor) -> bool:
    if tensor.numel() == 0:
        return True
    dimensions = sorted(
        (
            (int(stride), int(size))
            for size, stride in zip(tensor.shape, tensor.stride())
            if int(size) > 1
        ),
        key=lambda item: item[0],
    )
    expected_stride = 1
    for stride, size in dimensions:
        if stride != expected_stride:
            return False
        expected_stride *= size
    return True


def _is_non_overlapping(tensor: torch.Tensor) -> bool:
    if tensor.numel() == 0:
        return True
    dimensions = sorted(
        (
            (int(stride), int(size))
            for size, stride in zip(tensor.shape, tensor.stride())
            if int(size) > 1
        ),
        key=lambda item: item[0],
    )
    expected_stride = 1
    for stride, size in dimensions:
        if stride < expected_stride:
            return False
        expected_stride = stride * size
    return True


def require_non_overlapping_layout(
    op_name: str, *tensors: torch.Tensor
) -> None:
    if any(not _is_non_overlapping(tensor) for tensor in tensors):
        raise DnnReferenceNotSupportedError(
            f"cuDNN {op_name} does not support internally overlapping inputs"
        )


def empty_output_like_layout(
    reference: torch.Tensor,
    output_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate a safe output while preserving a dense reference layout."""
    if tuple(reference.shape) == tuple(
        output_shape
    ) and _is_non_overlapping_and_dense(reference):
        return torch.empty_strided(
            output_shape,
            tuple(reference.stride()),
            device=reference.device,
            dtype=dtype,
        )
    return torch.empty(output_shape, device=reference.device, dtype=dtype)


def build_cudnn_graph(graph: Any, op_name: str) -> None:
    try:
        graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
        message = str(exc)
        if (
            isinstance(exc, cudnn.cudnnGraphNotSupportedError)
            or "CUDNN_STATUS_NOT_SUPPORTED" in message
            or "No valid engine configs" in message
        ):
            raise DnnReferenceNotSupportedError(
                f"cuDNN frontend does not support {op_name}: {exc}"
            ) from exc
        raise


def require_cudnn_sdpa_execution_supported(
    tensor: torch.Tensor, op_name: str
) -> None:
    """Reject a known cuDNN runtime failure before it poisons CUDA state."""
    if (
        tensor.dtype == torch.float32
        and torch.cuda.get_device_capability(tensor.device) == (9, 0)
        and int(cudnn.backend_version()) // 100 == 924
    ):
        raise DnnReferenceNotSupportedError(
            "cuDNN backend 9.24 cannot safely execute FP32 "
            f"{op_name} on SM90"
        )


class PreparedCudnnOperation:
    reference_name: str

    def __init__(
        self,
        graph: Any,
        exec_tensors: dict[Any, torch.Tensor],
        workspace: torch.Tensor,
        output: Any,
        handle: Any,
        result_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self._graph = graph
        self._exec_tensors = exec_tensors
        self._workspace = workspace
        self._handle = handle
        self._raw_output = output
        self._result_transform = result_transform
        self.output = output
        self._closed = False

    def run(self) -> Any:
        if self._closed:
            raise RuntimeError("prepared cuDNN operation is closed")
        self._graph.execute(
            self._exec_tensors,
            self._workspace,
            handle=self._handle,
        )
        if self._result_transform is None:
            self.output = self._raw_output
        else:
            self.output = self._result_transform(self._raw_output)
        return self.output

    def __call__(self) -> torch.Tensor:
        return self.run()

    def close(self) -> None:
        self._closed = True


class NvidiaContext:
    def __init__(self) -> None:
        self.handle: Any = None
        self.device = torch.device("cuda")
        self.last_device: Any = None
        try:
            cudnn.backend_version()
            self.device = torch.device("cuda", torch.cuda.current_device())
            self.handle = cudnn.create_handle()
            self.activate_stream(self.device)
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup_exc:
                exc.add_note(
                    "cuDNN handle cleanup also failed during NVIDIA reference "
                    f"initialization: {cleanup_exc}"
                )
            raise DnnProviderUnavailableError(
                "cuDNN runtime is unavailable for the NVIDIA reference: "
                f"{exc}"
            ) from exc

    def activate_stream(self, device: torch.device) -> None:
        cudnn.set_stream(
            handle=self.handle,
            stream=torch.cuda.current_stream(device=device).cuda_stream,
        )

    def validate_tensor(self, op_name: str, tensor: Any) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"cuDNN {op_name} reference expects torch.Tensor inputs"
            )
        if tensor.layout != torch.strided:
            raise ValueError(
                f"cuDNN {op_name} reference requires a strided tensor, "
                f"got {tensor.layout}"
            )
        if tensor.device.type != "cuda":
            raise ValueError(
                f"cuDNN {op_name} reference requires CUDA tensors, "
                f"got {tensor.device}"
            )
        if tensor.device != self.device:
            raise ValueError(
                f"cuDNN {op_name} handle and inputs must use the same GPU, "
                f"got handle={self.device}, input={tensor.device}"
            )
        if tensor.dtype not in (*CUDNN_COMPARE_DTYPES, torch.bool):
            raise TypeError(
                f"cuDNN {op_name} reference does not support {tensor.dtype}"
            )
        return tensor

    def synchronize(self) -> None:
        if self.last_device is not None:
            torch.cuda.synchronize(device=self.last_device)

    def close(self) -> None:
        handle = self.handle
        if handle is None:
            return
        self.handle = None
        with torch.cuda.device(self.device):
            cudnn.destroy_handle(handle)
