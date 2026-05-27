import torch
from flag_dnn.ops.unary import unary


def positive(input: torch.Tensor) -> torch.Tensor:
    return unary(input, op_type="positive")
