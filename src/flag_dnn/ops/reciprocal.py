from typing import Optional

import torch

from flag_dnn.ops.unary import unary


def reciprocal(
    input: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    return unary(input, out=out, op_type="reciprocal")
