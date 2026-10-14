# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Specialize a single concatenation entry with one pointer per graph input."""

import ast
import math


def materialize_concatenate_source(source_bytes, tensors, axis):
    tree = ast.parse(source_bytes)
    entry = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "concatenate_kernel"
    )
    inputs, output = tensors[:-1], tensors[-1]
    storage = {
        "float32": "uint32", "int32": "uint32",
        "float16": "uint16", "bfloat16": "uint16",
        "boolean": "uint8", "fp8_e4m3": "uint8",
        "fp8_e5m2": "uint8", "fp8_e8m0": "uint8",
    }[output["data_type"]]
    input_names = {
        uid: f"input_{index}"
        for index, uid in enumerate(dict.fromkeys(t["uid"] for t in inputs))
    }
    names = [*input_names.values(), "output_ptr"]
    body = [
        "def concatenate_kernel(" + ", ".join(names)
        + ", BLOCK_SIZE: tl.constexpr):",
        "    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE "
        "+ tl.arange(0, BLOCK_SIZE)",
    ]
    begin = 0
    for source in inputs:
        end = begin + source["dimensions"][axis]
        body.append(
            f"    _concatenate_region({input_names[source['uid']]}, "
            f"output_ptr, logical, "
            f"{math.prod(output['dimensions'])}, "
            f"{tuple(output['dimensions'])!r}, {tuple(source['strides'])!r}, "
            f"{tuple(output['strides'])!r}, {axis}, {begin}, {end}, "
            f"tl.{storage}, BLOCK_SIZE)"
        )
        begin = end
    specialized = ast.parse("\n".join(body)).body[0]
    entry.args = specialized.args
    entry.body = specialized.body
    ast.fix_missing_locations(tree)
    return (
        "# Copyright 2026 FlagOS Contributors\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(tree) + "\n"
    ).encode("utf-8")
