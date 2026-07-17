from __future__ import annotations

from typing import Any, Union

import torch

try:
    import cudnn
except Exception as exc:
    raise RuntimeError(
        f"cuDNN frontend is unavailable for the NVIDIA oracle: {exc}"
    ) from exc

_Number = Union[int, float]
CUDNN_COMPARE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def cudnn_data_type(dtype):
    if dtype == torch.float16:
        return cudnn.data_type.HALF
    if dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    if dtype == torch.float32:
        return cudnn.data_type.FLOAT
    raise TypeError(f"Unsupported dtype for cuDNN frontend: {dtype}")


def cudnn_graph(dtype, cudnn_handle):
    return cudnn.pygraph(
        io_data_type=cudnn_data_type(dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )


def execute_cudnn_graph(
    graph,
    exec_tensors,
    output_value,
    output_template,
    cudnn_handle,
    op_name,
    skip_unsupported=False,
):
    del op_name, skip_unsupported
    output_value.set_output(True).set_data_type(
        cudnn_data_type(output_template.dtype)
    )
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    output = torch.empty_strided(
        tuple(output_value.get_dim()),
        tuple(output_value.get_stride()),
        device=output_template.device,
        dtype=output_template.dtype,
    )
    workspace = torch.empty(
        graph.get_workspace_size(),
        device=output_template.device,
        dtype=torch.uint8,
    )
    exec_tensors[output_value] = output
    graph.execute(exec_tensors, workspace, handle=cudnn_handle)
    torch.cuda.synchronize(device=output.device)
    return output


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


class PreparedCudnnAdd:
    def __init__(
        self,
        graph: Any,
        exec_tensors: dict[Any, torch.Tensor],
        workspace: torch.Tensor,
        output: torch.Tensor,
        handle: Any,
    ) -> None:
        self._graph = graph
        self._exec_tensors = exec_tensors
        self._workspace = workspace
        self._handle = handle
        self.output = output
        self._closed = False

    def run(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("prepared cuDNN Add runner is closed")
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


class NvidiaDnnProvider:
    vendor_name = "nvidia"
    implementation = "cudnn"
    display_name = "cuDNN"

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
        prepared = self.prepare_add(x, y, alpha=alpha)
        try:
            output = prepared.run()
            torch.cuda.synchronize(device=x.device)
        finally:
            prepared.close()
        self._last_device = x.device
        return output

    def prepare_add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: _Number = 1,
    ) -> PreparedCudnnAdd:
        alpha_value = _normalize_alpha(alpha)
        self._validate(x, y)
        try:
            output_shape = tuple(torch.broadcast_shapes(x.shape, y.shape))
        except RuntimeError as exc:
            raise ValueError(
                "cuDNN Add inputs are not broadcastable: "
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
            output_tensor.set_output(True).set_data_type(
                cudnn_data_type(x.dtype)
            )
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            output = torch.empty(
                output_shape,
                device=x.device,
                dtype=x.dtype,
            )
            workspace = torch.empty(
                graph.get_workspace_size(),
                device=x.device,
                dtype=torch.uint8,
            )
            exec_tensors[output_tensor] = output
        self._last_device = x.device
        return PreparedCudnnAdd(
            graph,
            exec_tensors,
            workspace,
            output,
            self._handle,
        )

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


def create_provider() -> NvidiaDnnProvider:
    return NvidiaDnnProvider()


class NvidiaDnnOracle(NvidiaDnnProvider):
    """Compatibility name for existing correctness-test integrations."""


def create_oracle() -> NvidiaDnnOracle:
    return NvidiaDnnOracle()
