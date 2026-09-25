#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the Hygon-private libtriton_jit compiler helper.

HCU Triton specializes tensor pointers whose reachable storage is at most
INT32_MAX bytes with ``tt.pointer_range=32``. libtriton_jit receives static
signatures rather than Tensor objects, so its upstream standalone helper
cannot infer that property. FlagDNN encodes the proven property as an ``S``
suffix and this deterministic build step teaches the private helper to restore
the matching TTIR attribute.

Some DTK Triton builds remove LLVM ``memory(...)`` attributes unconditionally
even though their bundled clang is new enough to consume them. For an OCML
``pow`` this leaves an illegal empty attribute group. The prepared helper patches that
HCU compiler method in process: clang 17 and newer retain the LLVM IR
unchanged, while an older clang receives the vendor compatibility rewrite plus
removal of any groups made empty by that rewrite. Builds that write the LLVM
IR directly use the same version-aware handling.

The supported source patterns are intentionally exact so an unknown upstream
change fails during helper preparation or import instead of silently dropping
an optimization or corrupting a compiler helper.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path


_UPSTREAM = """    # specialization: divisibility by 16 or equal to 1
    hints = {
        i: constexpr(s.rsplit(\":\", 1)[1])
        for i, s in enumerate(signature)
        if i not in constants and \":\" in s
    }
    hints = {k: v for k, v in hints.items() if v is not None}
    for h in hints.values():
        assert h in [1, 16], f\"Only 1 and 16 are valid hints, got {h}\"
    divisible_by_16 = tuple(i for i, h in hints.items() if h == 16)
    equal_to_1 = tuple(i for i, h in hints.items() if h == 1)

    if triton_version.major == 3 and triton_version.minor == 1:
        attrs = triton.compiler.AttrsDescriptor(
            divisible_by_16=divisible_by_16, equal_to_1=equal_to_1
        )
    elif triton_version.major == 3 and triton_version.minor == 2:
        attrs = triton.backends.compiler.AttrsDescriptor.from_dict(
            {
                \"arg_properties\": {
                    \"tt.divisibility\": divisible_by_16,
                    \"tt.equal_to\": equal_to_1,
                },
                \"cls\": \"AttrsDescriptor\",
            }
        )
    elif triton_version.major == 3 and triton_version.minor == 3:
        attrs = {(k,): [[\"tt.divisibility\", 16]] for k, v in hints.items() if v == 16}
    elif triton_version.major == 3 and triton_version.minor == 4:
        attrs = {(k,): [[\"tt.divisibility\", 16]] for k, v in hints.items() if v == 16}
    elif triton_version.major == 3 and triton_version.minor == 5:
        attrs = {(k,): [[\"tt.divisibility\", 16]] for k, v in hints.items() if v == 16}
    elif triton_version.major == 3 and triton_version.minor == 6:
        attrs = {(k,): [[\"tt.divisibility\", 16]] for k, v in hints.items() if v == 16}
    else:
        raise RuntimeError(
            \"Triton may change APIs, we cannot ensure compatibility here now. \"
            \"You can goto https://github.com/flagos-ai/libtriton_jit to raise an issue \"
            \"about supporting your triton version. Triton 3.1/3.2/3.3/3.4/3.5/3.6 are supported now.\"
        )
"""

_HYGON = """    # specialization: divisibility, equality, and HCU pointer range
    hints = {
        i: s.rsplit(\":\", 1)[1]
        for i, s in enumerate(signature)
        if i not in constants and \":\" in s
    }
    valid_hints = {\"1\", \"16\", \"S\", \"16S\"}
    for hint in hints.values():
        assert hint in valid_hints, f\"Invalid static signature hint: {hint}\"
    divisible_by_16 = tuple(
        i for i, hint in hints.items() if hint in (\"16\", \"16S\")
    )
    equal_to_1 = tuple(i for i, hint in hints.items() if hint == \"1\")
    pointer_range_32 = tuple(
        i for i, hint in hints.items() if hint in (\"S\", \"16S\")
    )

    if (
        pointer_range_32
        and triton_version.major == 3
        and triton_version.minor in (1, 2)
    ):
        raise RuntimeError(
            \"HCU pointer-range specialization requires Triton 3.3 or newer\"
        )

    if triton_version.major == 3 and triton_version.minor == 1:
        attrs = triton.compiler.AttrsDescriptor(
            divisible_by_16=divisible_by_16, equal_to_1=equal_to_1
        )
    elif triton_version.major == 3 and triton_version.minor == 2:
        attrs = triton.backends.compiler.AttrsDescriptor.from_dict(
            {
                \"arg_properties\": {
                    \"tt.divisibility\": divisible_by_16,
                    \"tt.equal_to\": equal_to_1,
                },
                \"cls\": \"AttrsDescriptor\",
            }
        )
    elif triton_version.major == 3 and triton_version.minor in (3, 4, 5, 6):
        attrs = {}
        for index in signature_without_spec:
            properties = []
            if index in divisible_by_16:
                properties.append([\"tt.divisibility\", 16])
            if index in pointer_range_32:
                properties.append([\"tt.pointer_range\", 32])
            if properties:
                attrs[(index,)] = properties
    else:
        raise RuntimeError(
            \"Triton may change APIs, we cannot ensure compatibility here now. \"
            \"You can goto https://github.com/flagos-ai/libtriton_jit to raise an issue \"
            \"about supporting your triton version. Triton 3.1/3.2/3.3/3.4/3.5/3.6 are supported now.\"
        )
"""


def _flagdnn_hygon_compatible_llir(source: str, clang_major: int) -> str:
    """Return LLVM IR accepted by the selected HCU clang generation."""

    if not isinstance(source, str):
        raise TypeError("HCU LLVM IR must be text")
    if not isinstance(clang_major, int) or isinstance(clang_major, bool):
        raise TypeError("HCU clang major version must be an integer")
    if clang_major < 0:
        raise ValueError("HCU clang major version must not be negative")
    if clang_major >= 17:
        return source

    import re as _flagdnn_re

    compatible = _flagdnn_re.sub(
        r"\s*memory\([^)]*\)", "", source
    )
    empty_groups = tuple(
        _flagdnn_re.findall(
            r"(?m)^attributes #([0-9]+) = \{\s*\}\s*$", compatible
        )
    )
    for group in empty_groups:
        compatible = _flagdnn_re.sub(
            rf"(?m)^attributes #{group} = \{{\s*\}}\s*(?:\n|$)",
            "",
            compatible,
        )
        # LLVM attribute-group references are whitespace-prefixed. The
        # numeric lookahead keeps #5 from matching #50.
        compatible = _flagdnn_re.sub(
            rf"[ \t]+#{group}(?![0-9])", "", compatible
        )
    return compatible


_HCU_PATCH_ANCHOR = "triton_version = Version(triton.__version__)\n"
_HCU_PATCH = inspect.getsource(_flagdnn_hygon_compatible_llir) + r'''

def _flagdnn_hygon_patch_hcu_compiler():
    import inspect as _flagdnn_inspect
    import re as _flagdnn_re
    import subprocess as _flagdnn_subprocess
    import textwrap as _flagdnn_textwrap
    from triton.backends.hcu import compiler_hcu as _flagdnn_compiler_hcu

    marker_name = "_flagdnn_hygon_memory_attribute_compatibility"
    marker_value = "clang-memory-attributes-v2"
    existing_marker = getattr(
        _flagdnn_compiler_hcu.HIPBackend, marker_name, None
    )
    if existing_marker is not None:
        if existing_marker != marker_value:
            raise RuntimeError(
                "another Hygon LLVM compatibility patch is already active"
            )
        return

    clang_path = _flagdnn_compiler_hcu.HIPBackend.path_to_rocm_clang()
    try:
        version_result = _flagdnn_subprocess.run(
            [clang_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, _flagdnn_subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"cannot identify HCU clang at {clang_path}"
        ) from error
    version_text = version_result.stdout + "\n" + version_result.stderr
    version_match = _flagdnn_re.search(
        r"\bclang version\s+([0-9]+)(?:\.|\b)", version_text
    )
    if version_match is None:
        raise RuntimeError(
            f"cannot parse HCU clang version from {version_text.strip()!r}"
        )
    clang_major = int(version_match.group(1))

    method_source = _flagdnn_textwrap.dedent(
        _flagdnn_inspect.getsource(
            _flagdnn_compiler_hcu.HIPBackend.make_amdgcn
        )
    )
    vendor_block = """        llir_str = str(src)
        # Remove memory(...) from function attributes
        llir_str = re.sub(r'\\s*memory\\([^)]*\\)', '', llir_str)
"""
    replacement = """        llir_str = _flagdnn_hygon_compatible_llir(
            str(src), _flagdnn_hygon_clang_major
        )
"""
    # Other HCU builds already preserve the LLVM IR. Match the .ll write
    # explicitly so this adaptation cannot affect an assembly/binary output.
    direct_write_block = """        with tempfile.NamedTemporaryFile(mode='w', suffix=".ll", delete=False) as f:
            llir_file = f.name
            f.write(str(src))
"""
    direct_write_replacement = """        with tempfile.NamedTemporaryFile(mode='w', suffix=".ll", delete=False) as f:
            llir_file = f.name
            f.write(_flagdnn_hygon_compatible_llir(
                str(src), _flagdnn_hygon_clang_major
            ))
"""
    vendor_occurrences = method_source.count(vendor_block)
    direct_occurrences = method_source.count(direct_write_block)
    if (vendor_occurrences, direct_occurrences) == (1, 0):
        method_source = method_source.replace(vendor_block, replacement)
    elif (vendor_occurrences, direct_occurrences) == (0, 1):
        method_source = method_source.replace(
            direct_write_block, direct_write_replacement
        )
    else:
        raise RuntimeError(
            "unsupported DTK Triton HCU memory-attribute compatibility "
            "block: expected exactly one recognized implementation, "
            f"found legacy={vendor_occurrences}, direct={direct_occurrences}"
        )

    compiler_globals = _flagdnn_compiler_hcu.__dict__
    helper_name = "_flagdnn_hygon_compatible_llir"
    if helper_name in compiler_globals:
        raise RuntimeError(
            "DTK Triton HCU compiler already defines FlagDNN LLVM helper"
        )
    compiler_globals[helper_name] = _flagdnn_hygon_compatible_llir
    compiler_globals["_flagdnn_hygon_clang_major"] = clang_major
    patched_namespace = {}
    exec(
        compile(
            method_source,
            "<flagdnn-hygon-hcu-compiler-compatibility>",
            "exec",
        ),
        compiler_globals,
        patched_namespace,
    )
    patched_method = patched_namespace.get("make_amdgcn")
    if not isinstance(patched_method, staticmethod):
        raise RuntimeError("cannot construct patched HCU make_amdgcn method")
    _flagdnn_compiler_hcu.HIPBackend.make_amdgcn = patched_method
    setattr(
        _flagdnn_compiler_hcu.HIPBackend, marker_name, marker_value
    )


if get_backend() == "HCU":
    _flagdnn_hygon_patch_hcu_compiler()
'''


def _prepare_source(source: str) -> str:
    specialization_occurrences = source.count(_UPSTREAM)
    if specialization_occurrences != 1:
        raise RuntimeError(
            "unsupported libtriton_jit standalone specialization block: "
            "expected exactly one match, found "
            f"{specialization_occurrences}"
        )
    anchor_occurrences = source.count(_HCU_PATCH_ANCHOR)
    if anchor_occurrences != 1:
        raise RuntimeError(
            "unsupported libtriton_jit Triton-version anchor: expected "
            f"exactly one match, found {anchor_occurrences}"
        )
    prepared = source.replace(_UPSTREAM, _HYGON)
    return prepared.replace(
        _HCU_PATCH_ANCHOR,
        _HCU_PATCH_ANCHOR + "\n" + _HCU_PATCH + "\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    source = arguments.input.read_text(encoding="utf-8")
    prepared = _prepare_source(source)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(prepared, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
