# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Adapt the private TritonJIT helper to MTGPU signatures and compiler APIs."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile


_REPLACEMENT = '''def _bracket_aware_split(sig: str) -> List[str]:
    """Split top-level commas while preserving grouped type syntax."""
    parts = []
    current = []
    opening = {"(": ")", "[": "]", "<": ">"}
    closing = {value: key for key, value in opening.items()}
    stack = []

    for ch in sig:
        if ch in opening:
            stack.append(ch)
        elif ch in closing:
            if not stack or stack[-1] != closing[ch]:
                raise ValueError(f"unmatched {ch!r} in signature: {sig}")
            stack.pop()

        if ch == "," and not stack:
            part = "".join(current).strip()
            if not part:
                raise ValueError(f"empty token in signature: {sig}")
            parts.append(part)
            current = []
        else:
            current.append(ch)

    if stack:
        raise ValueError(
            f"unmatched {opening[stack[-1]]!r} in signature: {sig}"
        )

    final = "".join(current).strip()
    if final:
        parts.append(final)
    elif parts:
        raise ValueError(f"empty token in signature: {sig}")
    return parts
'''


_MTGPU_IMPORT = """    elif backend == "MTGPU":
        import shutil

        try:
            from triton._C.libtriton import mtgpu
        except ImportError:
            from triton._C.libtriton import mthreads as mtgpu
"""

_MTGPU_COMPILED_ARTIFACT = """
        if not callable(getattr(mtgpu, "translate_llvmir_to_mubin", None)):
            # Some backends compile without the legacy translation binding.
            # Match MusaBackend's load order and verify the selected cache file
            # belongs to the CompiledKernel just returned by triton.compile().
            # Reject unrelated files that would mask a valid artifact.
            artifact_error = "no loadable kernel artifact was produced"
            for extension in ("mubin", "o", "so", "llir"):
                artifact_path = Path(cache_dir) / f"{fn.__name__}.{extension}"
                if not artifact_path.exists():
                    continue
                compiled_artifact = ccinfo.asm.get(extension)
                if isinstance(compiled_artifact, str):
                    compiled_artifact = compiled_artifact.encode("utf-8")
                if (
                    isinstance(compiled_artifact, (bytes, bytearray))
                    and compiled_artifact
                    and artifact_path.is_file()
                    and artifact_path.read_bytes() == compiled_artifact
                ):
                    return cache_dir
                artifact_error = (
                    f"{artifact_path} is empty or does not match "
                    "the artifact returned by triton.compile()"
                )
                break
            raise RuntimeError(
                "MTGPU: translate_llvmir_to_mubin is unavailable and "
                + artifact_error
            )
"""


def patch_source(source: str) -> str:
    start_marker = "def _bracket_aware_split(sig: str) -> List[str]:"
    end_marker = "\ndef _parse_type_token(token: str):"
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0 or source.find(start_marker, start + 1) >= 0:
        raise RuntimeError(
            "TritonJIT standalone signature splitter is unrecognized"
        )
    patched = source[:start] + _REPLACEMENT + source[end:]
    if patched.count(_MTGPU_IMPORT) != 1:
        raise RuntimeError(
            "TritonJIT standalone MTGPU compiler layout is unrecognized"
        )
    patched = patched.replace(
        _MTGPU_IMPORT, _MTGPU_IMPORT + _MTGPU_COMPILED_ARTIFACT, 1
    )
    compile(patched, "<FlagDNN private MTGPU helper>", "exec")
    return patched


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch_standalone_compile.py INPUT OUTPUT")
    input_path = Path(sys.argv[1])
    output = Path(sys.argv[2])
    if input_path.resolve() == output.resolve():
        raise RuntimeError("TritonJIT source and private output must differ")
    patched = patch_source(input_path.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(patched)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
