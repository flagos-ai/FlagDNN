from typing import Optional, Union
import torch
from flag_dnn.ops.binary import binary


def bitwise_xor(
    input: torch.Tensor,
    other: Union[torch.Tensor, int, bool],
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return binary(input, other, out=out, op_type="bitwise_xor")