import torch

from flag_dnn.ops.elu import elu as elu_op


def elu_(input: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return elu_op(input, alpha, inplace=True)
