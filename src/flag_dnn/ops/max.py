from typing import Optional, Union
import torch
from flag_dnn.ops.binary import binary


def max(
    input0: torch.Tensor,
    input1: Union[torch.Tensor, int, float],
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    return binary(input0, input1, out=out, op_type="max")
