#!/usr/bin/env python3
"""Dispatch a versioned FlagDNN build request to a platform compiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# The compiler process is ephemeral and artifact caching is handled explicitly
# by FlagDNN. Avoid leaving Python implementation caches in source or installed
# resource directories, which may also be read-only in deployed SDKs.
sys.dont_write_bytecode = True

if __package__:
    from .provider_loader import get_provider
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from flagdnn_codegen.provider_loader import get_provider


def _request_backend(request_path: Path) -> str:
    request: Any = json.loads(request_path.read_bytes())
    if not isinstance(request, dict):
        raise ValueError("compiler request must be a JSON object")
    backend = request.get("backend")
    if not isinstance(backend, str) or not backend:
        raise ValueError("compiler request backend is invalid")
    return backend


def _write_identity(output: Path, identity: dict[str, Any]) -> None:
    digest = identity.get("identity_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("compiler provider returned an invalid identity")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(digest + "\n", encoding="ascii")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--identify", action="store_true")
    parser.add_argument("--backend")
    parser.add_argument("--target")
    parser.add_argument(
        "--execution-engine",
        choices=("external_artifact", "libtriton_jit"),
        default="external_artifact",
    )
    parser.add_argument("--identity-output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()

    if arguments.identify:
        if (
            arguments.request is not None
            or arguments.output_dir is not None
            or not arguments.backend
            or not arguments.target
            or arguments.identity_output is None
        ):
            parser.error(
                "--identify requires --backend, --target and "
                "--identity-output only"
            )
        provider = get_provider(arguments.backend)
        identity = provider.compiler_identity(
            arguments.target, arguments.execution_engine
        )
        _write_identity(arguments.identity_output.resolve(), identity)
        result = {
            "backend": arguments.backend,
            "target": arguments.target,
            **identity,
        }
    else:
        if (
            arguments.request is None
            or arguments.output_dir is None
            or arguments.backend is not None
            or arguments.target is not None
            or arguments.identity_output is not None
        ):
            parser.error("compile mode requires --request and --output-dir")
        request_path = arguments.request.resolve()
        provider = get_provider(_request_backend(request_path))
        result = provider.compile_request(
            request_path,
            arguments.output_dir.resolve(),
            arguments.execution_engine,
        )

    if not arguments.quiet:
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
