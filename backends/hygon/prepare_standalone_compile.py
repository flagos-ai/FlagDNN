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
``pow`` this leaves an illegal empty attribute group. The prepared helper
patches that
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


_UPSTREAM = (
    "    # specialization: divisibility by 16 or equal to 1\n    hints"
    ' = {\n        i: constexpr(s.rsplit(":", 1)[1])\n        for i, s '
    'in enumerate(signature)\n        if i not in constants and ":" in'
    " s\n    }\n    hints = {k: v for k, v in hints.items() if v is not"
    " None}\n    for h in hints.values():\n        assert h in [1, 16],"
    ' f"Only 1 and 16 are valid hints, got {h}"\n    divisible_by_16 ='
    " tuple(i for i, h in hints.items() if h == 16)\n    equal_to_1 = "
    "tuple(i for i, h in hints.items() if h == 1)\n\n    if triton_vers"
    "ion.major == 3 and triton_version.minor == 1:\n        attrs = tr"
    "iton.compiler.AttrsDescriptor(\n            divisible_by_16=divis"
    "ible_by_16, equal_to_1=equal_to_1\n        )\n    elif triton_vers"
    "ion.major == 3 and triton_version.minor == 2:\n        attrs = tr"
    "iton.backends.compiler.AttrsDescriptor.from_dict(\n            {\n"
    '                "arg_properties": {\n                    "tt.divi'
    'sibility": divisible_by_16,\n                    "tt.equal_to": e'
    'qual_to_1,\n                },\n                "cls": "AttrsDescr'
    'iptor",\n            }\n        )\n    elif triton_version.major =='
    ' 3 and triton_version.minor == 3:\n        attrs = {(k,): [["tt.d'
    'ivisibility", 16]] for k, v in hints.items() if v == 16}\n    eli'
    "f triton_version.major == 3 and triton_version.minor == 4:\n     "
    '   attrs = {(k,): [["tt.divisibility", 16]] for k, v in hints.it'
    "ems() if v == 16}\n    elif triton_version.major == 3 and triton_"
    'version.minor == 5:\n        attrs = {(k,): [["tt.divisibility", '
    "16]] for k, v in hints.items() if v == 16}\n    elif triton_versi"
    "on.major == 3 and triton_version.minor == 6:\n        attrs = {(k"
    ',): [["tt.divisibility", 16]] for k, v in hints.items() if v == '
    '16}\n    else:\n        raise RuntimeError(\n            "Triton ma'
    'y change APIs, we cannot ensure compatibility here now. "\n      '
    '      "You can goto https://github.com/flagos-ai/libtriton_jit t'
    'o raise an issue "\n            "about supporting your triton ver'
    'sion. Triton 3.1/3.2/3.3/3.4/3.5/3.6 are supported now."\n       '
    " )\n"
)

_HYGON = (
    "    # specialization: divisibility, equality, and HCU pointer ra"
    'nge\n    hints = {\n        i: s.rsplit(":", 1)[1]\n        for i, '
    's in enumerate(signature)\n        if i not in constants and ":" '
    'in s\n    }\n    valid_hints = {"1", "16", "S", "16S"}\n    for hin'
    't in hints.values():\n        assert hint in valid_hints, f"Inval'
    'id static signature hint: {hint}"\n    divisible_by_16 = tuple(\n '
    '       i for i, hint in hints.items() if hint in ("16", "16S")\n '
    "   )\n    equal_to_1 = tuple(i for i, hint in hints.items() if hi"
    'nt == "1")\n    pointer_range_32 = tuple(\n        i for i, hint i'
    'n hints.items() if hint in ("S", "16S")\n    )\n\n    if (\n        '
    "pointer_range_32\n        and triton_version.major == 3\n        a"
    "nd triton_version.minor in (1, 2)\n    ):\n        raise RuntimeEr"
    'ror(\n            "HCU pointer-range specialization requires Trit'
    'on 3.3 or newer"\n        )\n\n    if triton_version.major == 3 and'
    " triton_version.minor == 1:\n        attrs = triton.compiler.Attr"
    "sDescriptor(\n            divisible_by_16=divisible_by_16, equal_"
    "to_1=equal_to_1\n        )\n    elif triton_version.major == 3 and"
    " triton_version.minor == 2:\n        attrs = triton.backends.comp"
    'iler.AttrsDescriptor.from_dict(\n            {\n                "a'
    'rg_properties": {\n                    "tt.divisibility": divisib'
    'le_by_16,\n                    "tt.equal_to": equal_to_1,\n       '
    '         },\n                "cls": "AttrsDescriptor",\n          '
    "  }\n        )\n    elif triton_version.major == 3 and triton_vers"
    "ion.minor in (3, 4, 5, 6):\n        attrs = {}\n        for index "
    "in signature_without_spec:\n            properties = []\n         "
    "   if index in divisible_by_16:\n                properties.appen"
    'd(["tt.divisibility", 16])\n            if index in pointer_range'
    '_32:\n                properties.append(["tt.pointer_range", 32])'
    "\n            if properties:\n                attrs[(index,)] = pr"
    'operties\n    else:\n        raise RuntimeError(\n            "Trit'
    'on may change APIs, we cannot ensure compatibility here now. "\n '
    '           "You can goto https://github.com/flagos-ai/libtriton_'
    'jit to raise an issue "\n            "about supporting your trito'
    'n version. Triton 3.1/3.2/3.3/3.4/3.5/3.6 are supported now."\n  '
    "      )\n"
)


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

    compatible = _flagdnn_re.sub(r"\s*memory\([^)]*\)", "", source)
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
_HCU_PATCH = inspect.getsource(_flagdnn_hygon_compatible_llir) + (
    "\n\ndef _flagdnn_hygon_patch_hcu_compiler():\n    import inspec"
    "t as _flagdnn_inspect\n    import re as _flagdnn_re\n    impor"
    "t subprocess as _flagdnn_subprocess\n    import textwrap as _"
    "flagdnn_textwrap\n    from triton.backends.hcu import compile"
    'r_hcu as _flagdnn_compiler_hcu\n\n    marker_name = "_flagdnn_'
    'hygon_memory_attribute_compatibility"\n    marker_value = "cl'
    'ang-memory-attributes-v2"\n    existing_marker = getattr(\n   '
    "     _flagdnn_compiler_hcu.HIPBackend, marker_name, None\n   "
    " )\n    if existing_marker is not None:\n        if existing_m"
    "arker != marker_value:\n            raise RuntimeError(\n     "
    '           "another Hygon LLVM compatibility patch is alread'
    'y active"\n            )\n        return\n\n    clang_path = _fl'
    "agdnn_compiler_hcu.HIPBackend.path_to_rocm_clang()\n    try:\n"
    "        version_result = _flagdnn_subprocess.run(\n          "
    '  [clang_path, "--version"],\n            check=True,\n       '
    "     capture_output=True,\n            text=True,\n        )\n "
    "   except (OSError, _flagdnn_subprocess.SubprocessError) as "
    'error:\n        raise RuntimeError(\n            f"cannot iden'
    'tify HCU clang at {clang_path}"\n        ) from error\n    ver'
    'sion_text = version_result.stdout + "\\n" + version_result.st'
    'derr\n    version_match = _flagdnn_re.search(\n        r"\\bcla'
    'ng version\\s+([0-9]+)(?:\\.|\\b)", version_text\n    )\n    if v'
    "ersion_match is None:\n        raise RuntimeError(\n          "
    '  f"cannot parse HCU clang version from {version_text.strip('
    ')!r}"\n        )\n    clang_major = int(version_match.group(1)'
    ")\n\n    method_source = _flagdnn_textwrap.dedent(\n        _fl"
    "agdnn_inspect.getsource(\n            _flagdnn_compiler_hcu.H"
    'IPBackend.make_amdgcn\n        )\n    )\n    vendor_block = """'
    "        llir_str = str(src)\n        # Remove memory(...) fro"
    "m function attributes\n        llir_str = re.sub(r'\\\\s*memory"
    '\\\\([^)]*\\\\)\', \'\', llir_str)\n"""\n    replacement = """       '
    " llir_str = _flagdnn_hygon_compatible_llir(\n            str("
    'src), _flagdnn_hygon_clang_major\n        )\n"""\n    # Other H'
    "CU builds already preserve the LLVM IR. Match the .ll write\n"
    "    # explicitly so this adaptation cannot affect an assembl"
    'y/binary output.\n    direct_write_block = """        with te'
    "mpfile.NamedTemporaryFile(mode='w', suffix=\".ll\", delete=Fal"
    "se) as f:\n            llir_file = f.name\n            f.write"
    '(str(src))\n"""\n    direct_write_replacement = """        wit'
    "h tempfile.NamedTemporaryFile(mode='w', suffix=\".ll\", delete"
    "=False) as f:\n            llir_file = f.name\n            f.w"
    "rite(_flagdnn_hygon_compatible_llir(\n                str(src"
    '), _flagdnn_hygon_clang_major\n            ))\n"""\n    vendor_'
    "occurrences = method_source.count(vendor_block)\n    direct_o"
    "ccurrences = method_source.count(direct_write_block)\n    if "
    "(vendor_occurrences, direct_occurrences) == (1, 0):\n        "
    "method_source = method_source.replace(vendor_block, replacem"
    "ent)\n    elif (vendor_occurrences, direct_occurrences) == (0"
    ", 1):\n        method_source = method_source.replace(\n       "
    "     direct_write_block, direct_write_replacement\n        )\n"
    '    else:\n        raise RuntimeError(\n            "unsupport'
    'ed DTK Triton HCU memory-attribute compatibility "\n         '
    '   "block: expected exactly one recognized implementation, "'
    '\n            f"found legacy={vendor_occurrences}, direct={di'
    'rect_occurrences}"\n        )\n\n    compiler_globals = _flagdn'
    'n_compiler_hcu.__dict__\n    helper_name = "_flagdnn_hygon_co'
    'mpatible_llir"\n    if helper_name in compiler_globals:\n     '
    '   raise RuntimeError(\n            "DTK Triton HCU compiler '
    'already defines FlagDNN LLVM helper"\n        )\n    compiler_'
    "globals[helper_name] = _flagdnn_hygon_compatible_llir\n    co"
    'mpiler_globals["_flagdnn_hygon_clang_major"] = clang_major\n '
    "   patched_namespace = {}\n    exec(\n        compile(\n       "
    '     method_source,\n            "<flagdnn-hygon-hcu-compiler'
    '-compatibility>",\n            "exec",\n        ),\n        com'
    "piler_globals,\n        patched_namespace,\n    )\n    patched_"
    'method = patched_namespace.get("make_amdgcn")\n    if not isi'
    "nstance(patched_method, staticmethod):\n        raise Runtime"
    'Error("cannot construct patched HCU make_amdgcn method")\n   '
    " _flagdnn_compiler_hcu.HIPBackend.make_amdgcn = patched_meth"
    "od\n    setattr(\n        _flagdnn_compiler_hcu.HIPBackend, ma"
    'rker_name, marker_value\n    )\n\n\nif get_backend() == "HCU":\n '
    "   _flagdnn_hygon_patch_hcu_compiler()\n"
)


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
