from typing import Optional

import torch

from flag_dnn.ops.bitwise_not import bitwise_not


def logical_not(
    input: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    if input.dtype != torch.bool:
        raise RuntimeError("logical_not expects bool input tensors")
    return bitwise_not(input, out=out)
