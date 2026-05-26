from typing import Optional

import torch

from flag_dnn.ops.add import add
from flag_dnn.ops.square import square


def add_square(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    return add(a, square(b), out=out)
