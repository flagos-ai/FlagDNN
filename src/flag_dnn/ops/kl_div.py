import logging

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)

# ATen reduction codes
_REDUCTION_NONE = 0
_REDUCTION_MEAN = 1
_REDUCTION_SUM = 2


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("abs"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def kl_div_elementwise_kernel(
    input_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    LOG_TARGET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute element-wise KL divergence in the native dtype of the pointers.
    Caller is responsible for promoting fp16/bf16 to float32 before calling.
    If LOG_TARGET=False:
        loss_i = target_i * (log(target_i) - input_i),
        or 0 if target_i <= 0.
    If LOG_TARGET=True:  loss_i = exp(target_i) * (target_i - input_i)
    """
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(input_ptr + offsets, mask=mask)
    tgt = tl.load(target_ptr + offsets, mask=mask)

    if LOG_TARGET:
        res = tl.exp(tgt) * (tgt - inp)
    else:
        log_tgt = tl.log(tl.maximum(tgt, 1e-38))
        res = tgt * (log_tgt - inp)
        res = tl.where(tgt > 0.0, res, 0.0)

    tl.store(out_ptr + offsets, res, mask=mask)


def kl_div(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: int = _REDUCTION_MEAN,
    *,
    log_target: bool = False,
) -> torch.Tensor:
    """ATen-compatible kl_div. reduction: 0=none, 1=mean, 2=sum."""
    logger.debug(
        f"FLAG_DNN KL_DIV (reduction={reduction}, log_target={log_target})"
    )

    # Keep original tensors; the Triton kernel promotes loads
    # to fp32 internally.
    work_dtype = (
        torch.float32
        if input.dtype in (torch.float16, torch.bfloat16)
        else input.dtype
    )
    inp_fp = input.contiguous()
    tgt_fp = target.contiguous()

    n_elements = input.numel()
    out_fp = torch.empty(input.shape, dtype=work_dtype, device=input.device)

    if n_elements == 0:
        if reduction == _REDUCTION_NONE:
            return out_fp.to(input.dtype)
        from flag_dnn.ops.sum import sum as flag_sum

        return flag_sum(out_fp)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        kl_div_elementwise_kernel[grid](
            inp_fp,
            tgt_fp,
            out_fp,
            n_elements,
            LOG_TARGET=log_target,
        )

    if reduction == _REDUCTION_NONE:
        return out_fp.to(input.dtype)
    elif reduction == _REDUCTION_MEAN:
        from flag_dnn.ops.mean import mean as flag_mean

        return flag_mean(out_fp)
    elif reduction == _REDUCTION_SUM:
        from flag_dnn.ops.sum import sum as flag_sum

        return flag_sum(out_fp)
    else:
        from flag_dnn.ops.mean import mean as flag_mean

        return flag_mean(out_fp)
