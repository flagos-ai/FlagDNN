from typing import Optional, Union
import torch
from flag_dnn.ops.binary import binary


def add(
    input: torch.Tensor,
    other: Union[torch.Tensor, int, float],
    *,
    alpha: Union[int, float] = 1,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    return binary(input, other, out=out, op_type="add", alpha=alpha)
