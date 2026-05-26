from typing import Optional
import torch
from flag_dnn.ops.binary import binary


def minimum(
    input: torch.Tensor,
    other: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return binary(input, other, out=out, op_type="minimum")