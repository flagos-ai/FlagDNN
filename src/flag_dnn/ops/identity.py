from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle


_COPY_BLOCK_SIZE = 1024


@triton.jit
def _identity_copy_kernel(
    input_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)


def _copy_dense_flat(input: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if input.numel() != out.numel():
        raise RuntimeError(
            f"identity copy numel mismatch: {input.numel()} vs {out.numel()}"
        )
    if not input.is_contiguous() or not out.is_contiguous():
        raise NotImplementedError(
            "flag_dnn identity materialization currently requires contiguous "
            "input and out"
        )
    n_elements = out.numel()
    if n_elements == 0:
        return out

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        _identity_copy_kernel[grid](
            input,
            out,
            n_elements,
            BLOCK_SIZE=_COPY_BLOCK_SIZE,
            num_warps=4,
        )
    return out


def identity(
    input: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type: Any = None,
    name: str = "",
) -> torch.Tensor:
    """Return ``input`` as a graph utility identity.

    Without ``out`` this is a view/no-op utility and does not launch a kernel.
    Supplying ``out`` requests materialization and uses a Triton copy kernel for
    the dense contiguous path.
    """
    del compute_data_type, name

    if out is None:
        return input

    if out.shape != input.shape:
        raise RuntimeError(
            f"identity out shape {tuple(out.shape)} does not match input "
            f"shape {tuple(input.shape)}"
        )
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"identity out dtype {out.dtype} does not match input dtype "
            f"{input.dtype}"
        )
    return _copy_dense_flat(input, out)
