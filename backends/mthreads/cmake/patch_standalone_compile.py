# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Patch the private TritonJIT helper for MTGPU descriptor signatures."""

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


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: patch_standalone_compile.py INPUT OUTPUT"
        )
    source = Path(sys.argv[1]).read_text(encoding="utf-8")
    start_marker = "def _bracket_aware_split(sig: str) -> List[str]:"
    end_marker = "\ndef _parse_type_token(token: str):"
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0 or source.find(start_marker, start + 1) >= 0:
        raise RuntimeError(
            "TritonJIT standalone signature splitter is unrecognized"
        )
    patched = source[:start] + _REPLACEMENT + source[end:]
    output = Path(sys.argv[2])
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
