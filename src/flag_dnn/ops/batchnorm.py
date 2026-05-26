from __future__ import annotations

import torch

from flag_dnn.ops.batch_norm import batchnorm_forward


def batchnorm(
    input: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    in_running_mean: torch.Tensor,
    in_running_var: torch.Tensor,
    epsilon,
    momentum,
    peer_stats=None,
    *,
    compute_data_type=None,
    name: str = "",
):
    del compute_data_type, name
    return batchnorm_forward(
        input,
        scale,
        bias,
        in_running_mean,
        in_running_var,
        epsilon,
        momentum,
        peer_stats=peer_stats,
    )
