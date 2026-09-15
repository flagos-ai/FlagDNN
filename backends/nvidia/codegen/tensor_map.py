"""Host TensorMap ABI: FP32 storage with hardware TF32-RNE input loads.

Only contiguous, external rank-two views are supported. Descriptor tile sizes
belong to the compiled variant, never to shared mutable tuning state.
"""

from __future__ import annotations

from typing import Any
import copy


def resolve_tensor_maps(
    signature: dict[str, str],
    arguments: list[dict[str, Any]],
    constants: dict[str, Any],
    *,
    gluon: bool = False,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    result = dict(signature)
    abi = copy.deepcopy(arguments)
    for name, argument in zip(signature, abi[:-2], strict=True):
        if argument["kind"] != "tensor_map":
            continue
        shape = argument["shape"]
        block = [constants.get(key) for key in argument["block_shape"]]
        if (
            result[name] != "tensordesc"
            or argument["data_type"] != "tf32_rne"
            or len(shape) != 2
            or any(
                isinstance(x, bool)
                or not isinstance(x, int)
                or x <= 0
                or x > 2**31 - 1
                for x in shape
            )
            or len(block) != 2
            or any(
                isinstance(x, bool)
                or not isinstance(x, int)
                or x < 32
                or x > 256
                or x & (x - 1)
                for x in block
            )
            or any(s % b for s, b in zip(shape, block))
            or argument["strides"] != [shape[1], 1]
            or argument["size"] != shape[0] * shape[1] * 4
            or argument["alignment"] < 16
        ):
            raise ValueError("unsupported host TensorMap metadata or tile")
        argument["block_shape"] = block
        layout = (
            ",NVMMASharedLayout(swizzle_byte_width=128,element_bitwidth=32,rank=2)"
            if gluon
            else ""
        )
        result[name] = f"tensordesc<fp32[{block[0]},{block[1]}]{layout}>"
    return result, abi
