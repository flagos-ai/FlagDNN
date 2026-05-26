import torch

from flag_dnn.ops.mul import mul


def scale(
    input: torch.Tensor,
    scale: torch.Tensor,
    *,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    return mul(
        input,
        scale,
        compute_data_type=compute_data_type,
        name=name,
    )
