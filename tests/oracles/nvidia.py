from __future__ import annotations

from typing import Any, Union

import torch

try:
    import cudnn
except Exception as exc:
    raise RuntimeError(
        f"cuDNN frontend is unavailable for the NVIDIA oracle: {exc}"
    ) from exc

from tests.base import (  # noqa: E402
    CUDNN_COMPARE_DTYPES,
    cudnn_graph,
    execute_cudnn_graph,
)


_Number = Union[int, float]


def _normalize_alpha(alpha: _Number) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(
            "cuDNN Add oracle alpha must be an int or float, "
            f"got {type(alpha).__name__}"
        )
    try:
        return float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "cuDNN Add oracle alpha cannot be represented as a float"
        ) from exc


class NvidiaDnnOracle:
    vendor_name = "nvidia"
    implementation = "cudnn"

    def __init__(self) -> None:
        try:
            cudnn.backend_version()
        except Exception as exc:
            raise RuntimeError(
                f"cuDNN backend is unavailable for the NVIDIA oracle: {exc}"
            ) from exc

        self._handle: Any = None
        try:
            self._device = torch.device("cuda", torch.cuda.current_device())
            self._handle = cudnn.create_handle()
            cudnn.set_stream(
                handle=self._handle,
                stream=torch.cuda.current_stream(
                    device=self._device
                ).cuda_stream,
            )
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup_exc:
                exc.add_note(
                    "cuDNN handle cleanup also failed during NVIDIA oracle "
                    f"initialization: {cleanup_exc}"
                )
            raise
        self._last_device: Any = None

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in CUDNN_COMPARE_DTYPES

    def _validate(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("cuDNN Add oracle expects two torch.Tensor inputs")
        if x.layout != torch.strided or y.layout != torch.strided:
            raise ValueError(
                "cuDNN Add oracle requires strided tensors, "
                f"got {x.layout} and {y.layout}"
            )
        if x.device.type != "cuda" or y.device.type != "cuda":
            raise ValueError(
                "cuDNN Add oracle requires CUDA tensors, "
                f"got {x.device} and {y.device}"
            )
        if x.device != y.device:
            raise ValueError(
                "cuDNN Add oracle inputs must be on the same GPU, "
                f"got {x.device} and {y.device}"
            )
        if x.device != self._device:
            raise ValueError(
                "cuDNN Add oracle handle and inputs must use the same GPU, "
                f"got handle={self._device}, inputs={x.device}"
            )
        if x.dtype != y.dtype:
            raise TypeError(
                "cuDNN Add oracle inputs must have the same dtype, "
                f"got {x.dtype} and {y.dtype}"
            )
        if not self.supports_dtype(x.dtype):
            raise TypeError(f"cuDNN Add oracle does not support {x.dtype}")

    def _validate_abs(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("cuDNN Abs oracle expects a torch.Tensor input")
        if x.layout != torch.strided:
            raise ValueError(
                "cuDNN Abs oracle requires a strided tensor, "
                f"got {x.layout}"
            )
        if x.device.type != "cuda":
            raise ValueError(
                "cuDNN Abs oracle requires a CUDA tensor, " f"got {x.device}"
            )
        if x.device != self._device:
            raise ValueError(
                "cuDNN Abs oracle handle and input must use the same GPU, "
                f"got handle={self._device}, input={x.device}"
            )
        if not self.supports_dtype(x.dtype):
            raise TypeError(f"cuDNN Abs oracle does not support {x.dtype}")

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
                "cuDNN Add oracle inputs are not broadcastable: "
                f"x_shape={tuple(x.shape)}, y_shape={tuple(y.shape)}"
            ) from exc

        with torch.cuda.device(x.device):
            cudnn.set_stream(
                handle=self._handle,
                stream=torch.cuda.current_stream(device=x.device).cuda_stream,
            )
            graph = cudnn_graph(x.dtype, self._handle)
            x_tensor = graph.tensor_like(x)
            y_tensor = graph.tensor_like(y)
            exec_tensors = {x_tensor: x, y_tensor: y}

            add_rhs = y_tensor
            if alpha_value != 1:
                alpha_tensor_value = torch.full_like(y, alpha_value)
                alpha_tensor = graph.tensor_like(alpha_tensor_value)
                exec_tensors[alpha_tensor] = alpha_tensor_value
                add_rhs = graph.mul(
                    a=y_tensor,
                    b=alpha_tensor,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="add_alpha_scale",
                )

            output_tensor = graph.add(
                a=x_tensor,
                b=add_rhs,
                compute_data_type=cudnn.data_type.FLOAT,
                name="add",
            )
            output_template = torch.empty(
                output_shape,
                device=x.device,
                dtype=x.dtype,
            )
            output = execute_cudnn_graph(
                graph,
                exec_tensors,
                output_tensor,
                output_template,
                self._handle,
                "add",
                skip_unsupported=False,
            )
        self._last_device = x.device
        return output

    def abs(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_abs(x)

        with torch.cuda.device(x.device):
            cudnn.set_stream(
                handle=self._handle,
                stream=torch.cuda.current_stream(device=x.device).cuda_stream,
            )
            graph = cudnn_graph(x.dtype, self._handle)
            x_tensor = graph.tensor_like(x)
            output_tensor = graph.abs(
                input=x_tensor,
                compute_data_type=cudnn.data_type.FLOAT,
                name="abs",
            )
            output_template = torch.empty_strided(
                x.shape,
                x.stride(),
                device=x.device,
                dtype=x.dtype,
            )
            output = execute_cudnn_graph(
                graph,
                {x_tensor: x},
                output_tensor,
                output_template,
                self._handle,
                "abs",
                skip_unsupported=False,
            )
        self._last_device = x.device
        return output

    def synchronize(self) -> None:
        if self._last_device is not None:
            torch.cuda.synchronize(device=self._last_device)

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return

        self._handle = None
        with torch.cuda.device(self._device):
            cudnn.destroy_handle(handle)


def create_oracle() -> NvidiaDnnOracle:
    return NvidiaDnnOracle()
