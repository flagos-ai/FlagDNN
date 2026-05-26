from typing import Optional, Union
import torch
from flag_dnn.ops.unary import unary

def isnan(
    input: torch.Tensor
) -> torch.Tensor:
    return unary(input, op_type="isnan")
