from typing import Optional

import torch

from flag_dnn import runtime
from flag_dnn.ops.unary import unary


_PORTABLE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def tan(
    input: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    if input.dtype not in _PORTABLE_DTYPES:
        raise NotImplementedError(
            f"flag_dnn tan does not support dtype={input.dtype} "
            f"on device={runtime.device.name}"
        )
    if input.device.type != runtime.device.name:
        raise RuntimeError(
            f"flag_dnn tan expected a {runtime.device.name} tensor, "
            f"got device={input.device}"
        )
    return unary(input, out=out, op_type="tan")
