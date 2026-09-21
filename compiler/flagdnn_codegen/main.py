#!/usr/bin/env python3
"""Dispatch a versioned FlagDNN build request to a platform compiler."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any

# The compiler process is ephemeral and artifact caching is handled explicitly
# by FlagDNN. Avoid leaving Python implementation caches in source or installed
# resource directories, which may also be read-only in deployed SDKs.
sys.dont_write_bytecode = True

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_provider_loader = importlib.import_module(
    ".provider_loader",
    package=__package__ or "flagdnn_codegen",
)

_IDENTITY_TEMPORARY_FAILURE = 75


class _IdentitySnapshotChanged(RuntimeError):
    """Signal that identity discovery raced a dependency change."""


def get_provider(backend: str) -> Any:
    return _provider_loader.get_provider(backend)  # type: ignore[attr-defined]


def _request_backend(request_path: Path) -> str:
    request: Any = json.loads(request_path.read_bytes())
    if not isinstance(request, dict):
        raise ValueError("compiler request must be a JSON object")
    backend = request.get("backend")
    if not isinstance(backend, str) or not backend:
        raise ValueError("compiler request backend is invalid")
    return backend


def _request_identity_and_target(request_path: Path) -> tuple[str, str, str]:
    request: Any = json.loads(request_path.read_bytes())
    if not isinstance(request, dict):
        raise ValueError("compiler request must be a JSON object")
    backend = request.get("backend")
    target = request.get("target")
    identity = request.get("compiler_identity")
    if not isinstance(backend, str) or not backend:
        raise ValueError("compiler request backend is invalid")
    if not isinstance(target, str) or not target:
        raise ValueError("compiler request target is invalid")
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ValueError("compiler request identity is invalid")
    return backend, target, identity


def _identity_dependencies(
    provider: Any, target: str, execution_engine: str
) -> tuple[list[str], bool]:
    values: list[object] = [
        Path(__file__),
        Path(_provider_loader.__file__),
        Path(provider.__file__),
        Path(sys.executable),
    ]
    provider_dependencies = getattr(
        provider, "compiler_identity_dependencies", None
    )
    dependencies_complete = callable(provider_dependencies)
    if dependencies_complete:
        supplied = provider_dependencies(target, execution_engine)
        if isinstance(supplied, (str, bytes)):
            raise ValueError(
                "compiler_identity_dependencies() must return paths"
            )
        values.extend(supplied)

    result: set[str] = set()
    for value in values:
        try:
            text = os.path.abspath(os.fspath(Path(value).expanduser()))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "compiler identity dependency is not path-like"
            ) from error
        if not text or "\x00" in text:
            raise ValueError("compiler identity dependency path is invalid")
        result.add(text)
    if len(result) > 65536:
        raise ValueError("compiler identity has too many dependencies")
    return sorted(result), dependencies_complete


def _stat_record(value: os.stat_result) -> bytes:
    return (
        f"{value.st_dev}:{value.st_ino}:{value.st_mode}:{value.st_size}:"
        f"{value.st_mtime_ns // 1_000_000_000}:"
        f"{value.st_mtime_ns % 1_000_000_000}:"
        f"{value.st_ctime_ns // 1_000_000_000}:"
        f"{value.st_ctime_ns % 1_000_000_000}"
    ).encode("ascii")


def _error_number(error: OSError) -> int:
    return error.errno if error.errno is not None else -1


def _dependency_fingerprint(dependency: str) -> str:
    """Hash lstat, link text and followed stat without resolving the path."""

    digest = hashlib.sha256(b"flagdnn-dependency-state-v1\0")
    try:
        link_status = os.lstat(dependency)
    except OSError as error:
        digest.update(f"lstat-error:{_error_number(error)}\0".encode("ascii"))
        return digest.hexdigest()

    digest.update(b"lstat:")
    digest.update(_stat_record(link_status))
    digest.update(b"\0")
    if stat.S_ISLNK(link_status.st_mode):
        try:
            link_value = os.readlink(dependency)
        except OSError as error:
            digest.update(
                f"link-error:{_error_number(error)}\0".encode("ascii")
            )
        else:
            digest.update(b"link:")
            digest.update(os.fsencode(link_value))
            digest.update(b"\0")
    else:
        digest.update(b"link:\0")

    try:
        target_status = os.stat(dependency)
    except OSError as error:
        digest.update(f"stat-error:{_error_number(error)}\0".encode("ascii"))
    else:
        digest.update(b"stat:")
        digest.update(_stat_record(target_status))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_dependency_digest(dependency: str, fingerprint: str) -> str:
    content = hashlib.sha256()
    with open(dependency, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            content.update(chunk)
    if _dependency_fingerprint(dependency) != fingerprint:
        raise _IdentitySnapshotChanged
    return content.hexdigest()


# A compiler subprocess is short-lived. Reuse content only within this process,
# with all lstat/link/stat fields in the key, and never cache a racing read.
_cached_dependency_digest = functools.lru_cache(maxsize=4096)(
    _read_dependency_digest
)


def _dependency_snapshots(
    dependencies: list[str],
) -> list[dict[str, str]]:
    snapshots: list[dict[str, str]] = []
    for dependency in dependencies:
        snapshot = {
            "path": dependency,
            "fingerprint": _dependency_fingerprint(dependency),
        }
        try:
            status = os.stat(dependency)
        except OSError:
            pass
        else:
            if stat.S_ISREG(status.st_mode):
                # Coarse filesystem timestamps can alias consecutive writes.
                # Recent/future files must be read again even if their metadata
                # looks unchanged; installed toolchains normally predate this.
                stable = time.time_ns() - max(
                    status.st_mtime_ns, status.st_ctime_ns
                ) >= 2_000_000_000
                digest = (_cached_dependency_digest if stable
                          else _read_dependency_digest)
                snapshot["content_sha256"] = digest(
                    dependency, snapshot["fingerprint"]
                )
        snapshots.append(snapshot)
    return snapshots


def _stable_compiler_identity(
    provider: Any, target: str, execution_engine: str
) -> tuple[dict[str, Any], list[str], list[dict[str, str]], bool]:
    dependencies_before, complete_before = _identity_dependencies(
        provider, target, execution_engine
    )
    snapshots_before = _dependency_snapshots(dependencies_before)
    identity = provider.compiler_identity(target, execution_engine)
    dependencies_after, complete_after = _identity_dependencies(
        provider, target, execution_engine
    )
    snapshots_after = _dependency_snapshots(dependencies_after)
    if (
        complete_before != complete_after
        or dependencies_before != dependencies_after
        or snapshots_before != snapshots_after
    ):
        raise _IdentitySnapshotChanged
    return identity, dependencies_after, snapshots_after, complete_after


def _write_identity(
    output: Path,
    identity: dict[str, Any],
    dependencies: list[str],
    snapshots: list[dict[str, str]],
    dependencies_complete: bool,
) -> None:
    digest = identity.get("identity_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("compiler provider returned an invalid identity")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    metadata = json.dumps(
        {
            "dependencies_complete": dependencies_complete,
            "files": dependencies,
            "schema_version": 1,
            "snapshot_schema_version": 1,
            "snapshots": snapshots,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    temporary.write_text(digest + "\n" + metadata + "\n", encoding="utf-8")
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
        try:
            (
                identity,
                dependencies,
                snapshots,
                dependencies_complete,
            ) = _stable_compiler_identity(
                provider, arguments.target, arguments.execution_engine
            )
        except _IdentitySnapshotChanged:
            return _IDENTITY_TEMPORARY_FAILURE
        _write_identity(
            arguments.identity_output.resolve(),
            identity,
            dependencies,
            snapshots,
            dependencies_complete,
        )
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
        backend, target, requested_identity = _request_identity_and_target(
            request_path
        )
        provider = get_provider(backend)
        try:
            identity_before, dependencies, snapshots_before, complete_before = (
                _stable_compiler_identity(
                    provider, target, arguments.execution_engine
                )
            )
            if identity_before.get("identity_sha256") != requested_identity:
                raise ValueError(
                    "request compiler identity does not match provider"
                )
            result = provider.compile_request(
                request_path,
                arguments.output_dir.resolve(),
                arguments.execution_engine,
            )
            dependencies_after, complete_after = _identity_dependencies(
                provider, target, arguments.execution_engine
            )
            snapshots_after = _dependency_snapshots(dependencies_after)
            if (
                complete_before != complete_after
                or dependencies != dependencies_after
                or snapshots_before != snapshots_after
            ):
                raise _IdentitySnapshotChanged
        except _IdentitySnapshotChanged:
            return _IDENTITY_TEMPORARY_FAILURE

    if not arguments.quiet:
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
