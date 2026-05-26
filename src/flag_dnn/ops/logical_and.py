from typing import Optional, Union

import torch

from flag_dnn.ops.bitwise_and import bitwise_and


def _check_bool_pair(
    input: torch.Tensor, other: Union[torch.Tensor, bool], op_name: str
) -> None:
    if input.dtype != torch.bool:
        raise RuntimeError(f"{op_name} expects bool input tensors")
    if isinstance(other, torch.Tensor) and other.dtype != torch.bool:
        raise RuntimeError(f"{op_name} expects bool input tensors")


def logical_and(
    input: torch.Tensor | None = None,
    other: Union[torch.Tensor, bool, None] = None,
    *,
    a: torch.Tensor | None = None,
    b: Union[torch.Tensor, bool, None] = None,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    left = input if input is not None else a
    right = other if other is not None else b
    if left is None or right is None:
        raise TypeError("logical_and expects input/other or a/b")
    _check_bool_pair(left, right, "logical_and")
    return bitwise_and(left, right, out=out)
