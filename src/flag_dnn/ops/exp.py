from typing import Optional, Union
import torch
from flag_dnn.ops.unary import unary

def exp(
    input: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return unary(input, out=out, op_type="exp")
