from typing import Optional, Union
import torch
from flag_dnn.ops.binary import binary


def div(
    input: torch.Tensor,
    other: Union[torch.Tensor, int, float],
    *,
    rounding_mode: Optional[str] = None,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    return binary(
        input, other, out=out, op_type="div", rounding_mode=rounding_mode
    )
