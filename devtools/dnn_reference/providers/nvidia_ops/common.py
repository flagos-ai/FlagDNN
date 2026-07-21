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

from typing import Any

import torch

try:
    import cudnn
except Exception as exc:
    raise RuntimeError(
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


class PreparedCudnnOperation:
    def __init__(
        self,
        graph: Any,
        exec_tensors: dict[Any, torch.Tensor],
        workspace: torch.Tensor,
        output: Any,
        handle: Any,
    ) -> None:
        self._graph = graph
        self._exec_tensors = exec_tensors
        self._workspace = workspace
        self._handle = handle
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
        return self.output

    def __call__(self) -> torch.Tensor:
        return self.run()

    def close(self) -> None:
        self._closed = True


class NvidiaContext:
    def __init__(self) -> None:
        try:
            cudnn.backend_version()
        except Exception as exc:
            raise RuntimeError(
                f"cuDNN backend is unavailable for the NVIDIA reference: {exc}"
            ) from exc

        self.handle: Any = None
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.last_device: Any = None
        try:
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
            raise

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
