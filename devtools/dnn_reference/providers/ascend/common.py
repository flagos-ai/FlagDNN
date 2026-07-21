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
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch

from .build import build_aclnn_oracle, find_cann_layout


DTYPE_CODES = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}
ERROR_BUFFER_SIZE = 4096
INT64_POINTER = ctypes.POINTER(ctypes.c_int64)


def load_cdll(path: Path, mode: int, description: str) -> Any:
    try:
        return ctypes.CDLL(str(path), mode=mode)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load {description} from {path}: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def load_library() -> Any:
    layout = find_cann_layout()
    mode = os.RTLD_NOW | os.RTLD_GLOBAL
    dependencies = (
        load_cdll(
            layout.lib_dir / "libascendcl.so",
            mode,
            "CANN libascendcl.so",
        ),
        load_cdll(
            layout.lib_dir / "libopapi.so",
            mode,
            "CANN libopapi.so",
        ),
    )
    wrapper_path = build_aclnn_oracle(layout=layout)
    library = load_cdll(wrapper_path, mode, "ACLNN test wrapper")
    library._flagdnn_dependency_handles = dependencies
    return library


def metadata_array(values: Iterable[int]) -> Any:
    items = tuple(int(value) for value in values)
    return (ctypes.c_int64 * len(items))(*items)


def npu_module() -> Any:
    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError(
            "torch_npu has not registered torch.npu for the Ascend reference"
        )
    return npu


class AscendContext:
    def __init__(
        self,
        library_loader: Callable[[], Any],
        npu_loader: Callable[[], Any],
    ) -> None:
        self._library_loader = library_loader
        self._npu_loader = npu_loader
        self._library: Any = None
        self._configured: set[Callable[[Any], None]] = set()
        self.last_device: Any = None

    def get_library(
        self,
        configure: Optional[Callable[[Any], None]] = None,
    ) -> Any:
        if self._library is None:
            self._library = self._library_loader()
        if configure is not None and configure not in self._configured:
            configure(self._library)
            self._configured.add(configure)
        return self._library

    def npu(self) -> Any:
        return self._npu_loader()

    def synchronize(self) -> None:
        if self.last_device is not None:
            self.npu().synchronize(device=self.last_device)


class PreparedAclnnOperation:
    def __init__(
        self,
        library: Any,
        handle: ctypes.c_void_p,
        keepalive: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        *,
        operation_name: str,
        symbol_name: Optional[str] = None,
    ) -> None:
        self._library = library
        self._handle = handle
        self._keepalive = keepalive
        self._operation_name = operation_name
        self._symbol_name = symbol_name or (
            "aclnn_" + operation_name[len("aclnn") :].lower()
        )
        self.output = output

    def run(self) -> torch.Tensor:
        handle = self._handle
        if handle.value is None:
            raise RuntimeError(
                f"prepared {self._operation_name} runner is closed"
            )
        error_buffer = ctypes.create_string_buffer(ERROR_BUFFER_SIZE)
        function = getattr(self._library, f"flagdnn_{self._symbol_name}_run")
        status = function(
            handle,
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
        if status != 0:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"prepared {self._operation_name} execution failed: "
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
        error_buffer = ctypes.create_string_buffer(ERROR_BUFFER_SIZE)
        function = getattr(
            self._library, f"flagdnn_{self._symbol_name}_destroy"
        )
        status = function(
            handle,
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
        if status != 0:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"prepared {self._operation_name} cleanup failed: "
                f"status={status}, detail={detail}"
            )
