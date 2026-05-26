import torch

from flag_dnn.ops.batch_norm import batchnorm_inference_forward


def batchnorm_inference(
    input: torch.Tensor,
    mean: torch.Tensor,
    inv_variance: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    return batchnorm_inference_forward(input, mean, inv_variance, scale, bias)
