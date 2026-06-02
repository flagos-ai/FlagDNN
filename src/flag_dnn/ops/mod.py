from typing import Optional, Union

import torch

from flag_dnn.ops.binary import binary


def mod(
    input: torch.Tensor | None = None,
    other: Union[torch.Tensor, int, float, None] = None,
    *,
    input0: torch.Tensor | None = None,
    input1: Union[torch.Tensor, int, float, None] = None,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    left = input if input is not None else input0
    right = other if other is not None else input1
    if left is None:
        raise TypeError("mod missing input tensor")
    if right is None:
        raise TypeError("mod missing other operand")
    return binary(left, right, out=out, op_type="mod")
