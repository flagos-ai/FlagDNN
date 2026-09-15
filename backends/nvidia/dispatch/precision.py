"""Preserve explicit IEEE/TF32 choices across NVIDIA algorithm selection."""

from .common import _require_integer
from .conv_fprop import _convolution_kernel_configuration
from .conv_backward import _convolution_backward_kernel_configuration
from .matmul import _matmul_kernel_configuration


def _explicit_precision_configuration(
    operation, parameters, tensors, architecture
):
    precision = _require_integer(
        parameters, "input_precision", minimum=1, maximum=2
    )
    if any(tensor["data_type"] != "float32" for tensor in tensors):
        raise ValueError("explicit IEEE/TF32 precision requires FP32 storage")
    if operation == "matmul":
        # The generic GEMM plans honor the dot-mode selectors and do
        # not pack or
        # round inputs before the selected operation executes.
        function, signature, constants, grid, arguments = (
            _matmul_kernel_configuration(parameters, tensors, 0)
        )
        constants["USE_TF32"] = precision == 2
        # This existing selector enables the FP32 tensor-core paths. Disabling
        # it with FP32 pointers selects IEEE FP32 dot without changing storage.
        constants["INPUT_IS_FLOAT32"] = precision == 2
        constants["NATIVE_TF32_RNE"] = architecture >= 90
        # SIMT FP32 accumulators need smaller tiles than tensor-core GEMM.
        # More independent tiles also keep short contractions occupied.
        m, n, k = (constants[name] for name in ("M", "N", "K"))
        block_m = 16 if m * n <= 8192 else 32
        block_n = 32 if m * n <= 8192 else 64
        constants.update(BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=32)
        grid = (
            ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n),
            grid[1],
            1,
        )
    elif operation == "convolution_fprop":
        function, signature, constants, grid, arguments = (
            _convolution_kernel_configuration(parameters, tensors)
        )
        constants["INPUT_PRECISION"] = 0 if precision == 1 else 2
        if (
            precision == 1
            and function == "conv2d_spatial_nchw_kernel"
            and constants["CIN_PER_GROUP"] * constants["KH"] * constants["KW"]
            <= 32
            and constants["COUT_PER_GROUP"] * constants["OH"] * constants["OW"]
            <= 256
        ):
            function = "conv_fprop_direct_kernel"
            constants["BLOCK_HW"] = 32
            grid = (
                (
                    constants["COUT_PER_GROUP"]
                    * constants["OH"]
                    * constants["OW"]
                    + 31
                )
                // 32,
                grid[1],
                1,
            )
    else:
        function, signature, constants, grid, arguments = (
            _convolution_backward_kernel_configuration(
                operation, parameters, tensors
            )
        )
        constants["INPUT_PRECISION"] = 0 if precision == 1 else 2
    if (
        precision == 1
        and function == "conv_dgrad_nd_kernel"
        and constants["COUT_PER_GROUP"]
        * constants["KD"]
        * constants["KH"]
        * constants["KW"]
        <= 192
    ):
        function = "conv_dgrad_direct_kernel"
        constants["BLOCK_M"] = 128
        grid = (
            (constants["M"] + 127) // 128,
            grid[1] * constants["CIN_PER_GROUP"],
            1,
        )
    elif (
        precision == 1
        and function == "conv_wgrad_nd_kernel"
        and constants["M"] <= 4096
    ):
        function = "conv_wgrad_direct_kernel"
        constants["BLOCK_M"] = 1 << (constants["M"] - 1).bit_length()
        grid = (
            constants["COUT_PER_GROUP"] * constants["CIN_PER_GROUP"] * grid[1],
            grid[2],
            1,
        )
    return function, signature, constants, grid, arguments
