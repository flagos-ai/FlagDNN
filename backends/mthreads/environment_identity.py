#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
PLATFORM = "mthreads"
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "platform",
    "resources",
    "python",
    "errors",
    "identity_sha256",
}
REQUIRED_RESOURCES = (
    "musa_root",
    "musa_runtime_header",
    "musa_driver_header",
    "mudnn_header",
    "musa_runtime",
    "musa_driver",
    "mudnn",
    "mcc",
    "patchelf",
    "triton_jit_prefix",
    "triton_jit",
    "triton_jit_config",
    "triton_jit_header",
    "triton_jit_gen_ssig",
    "triton_jit_standalone_compile",
)
MUSA_ROOT_RESOURCES = (
    "musa_runtime_header",
    "musa_driver_header",
    "mudnn_header",
    "musa_runtime",
    "mudnn",
    "mcc",
)
TRITON_JIT_PREFIX_RESOURCES = (
    "triton_jit",
    "triton_jit_config",
    "triton_jit_header",
    "triton_jit_gen_ssig",
    "triton_jit_standalone_compile",
)
REQUIRED_PYTHON_MODULES = ("torch", "torch_musa", "triton", "yaml")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CollectionError(RuntimeError):
    pass


def canonical_json(document: dict[str, object]) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: Path, **extra: object) -> dict[str, object]:
    requested = path.absolute()
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise CollectionError(f"required file is not regular: {requested}")
    return {
        "requested_path": str(requested),
        "realpath": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        **extra,
    }


def _directory_record(path: Path) -> dict[str, object]:
    requested = path.absolute()
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise CollectionError(f"required directory is not present: {requested}")
    return {
        "requested_path": str(requested),
        "realpath": str(resolved),
    }


def _run(
    argv: list[str],
    *,
    timeout: int = 120,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=environment,
    )


def _require_command(name: str) -> Path:
    resolved = shutil.which(name)
    if resolved is None:
        raise CollectionError(f"required command is not on PATH: {name}")
    return Path(resolved)


def _resolve_library(name: str, search_directories: list[Path]) -> Path:
    candidates = (name, f"{name}.1", f"{name}.so", f"lib{name}.so")
    for directory in search_directories:
        for candidate in candidates:
            path = directory / candidate
            if path.exists():
                return path

    ldconfig = shutil.which("ldconfig")
    if ldconfig is not None:
        result = _run([ldconfig, "-p"], timeout=30)
        if result.returncode == 0:
            requested_names = {name, f"{name}.1", f"{name}.so", f"lib{name}.so"}
            for line in result.stdout.splitlines():
                match = re.match(r"\s*(\S+)\s+\([^)]*\)\s+=>\s+(\S+)", line)
                if match and match.group(1) in requested_names:
                    path = Path(match.group(2))
                    if path.exists():
                        return path
    raise CollectionError(f"required shared library was not found: {name}")


def _parse_cmake_backend(config: Path) -> str:
    text = config.read_text(encoding="utf-8")
    match = re.search(
        r"set\s*\(\s*TritonJIT_BACKEND\s+[\"']?([A-Za-z0-9_]+)[\"']?\s*\)",
        text,
    )
    if match is None:
        raise CollectionError(
            f"TritonJITConfig.cmake does not declare TritonJIT_BACKEND: {config}"
        )
    return match.group(1)


def _parse_elf_dynamic_section(text: str) -> tuple[str, list[str]]:
    soname_match = re.search(
        r"\(SONAME\).*?(?:Library soname|Shared library): \[([^\]]+)\]",
        text,
    )
    if soname_match is None:
        raise CollectionError("ELF SONAME is missing")
    runpath_match = re.search(
        r"\((?:RUNPATH|RPATH)\).*?Library (?:runpath|rpath): \[([^\]]*)\]",
        text,
    )
    runpath = [] if runpath_match is None else runpath_match.group(1).split(":")
    return soname_match.group(1), runpath


def _elf_metadata(library: Path) -> tuple[str, list[str]]:
    readelf = _require_command("readelf")
    result = _run([str(readelf), "-d", str(library)], timeout=60)
    if result.returncode != 0:
        raise CollectionError(
            f"readelf failed for {library}: {result.stderr.strip()}"
        )
    try:
        return _parse_elf_dynamic_section(result.stdout)
    except CollectionError as error:
        raise CollectionError(f"{error}: {library}") from error


def _ldd_dependencies(library: Path) -> list[dict[str, object]]:
    ldd = _require_command("ldd")
    result = _run([str(ldd), str(library)], timeout=120)
    if result.returncode != 0:
        raise CollectionError(f"ldd failed for {library}: {result.stderr.strip()}")
    dependencies: list[dict[str, object]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if "=>" not in line:
            continue
        name, remainder = (part.strip() for part in line.split("=>", 1))
        if remainder.startswith("not found"):
            resolved: str | None = None
        else:
            resolved = remainder.split(" ", 1)[0]
            if resolved:
                resolved = str(Path(resolved).resolve(strict=False))
        dependencies.append({"name": name, "resolved": resolved})
    return dependencies


def _module_records(
    names: tuple[str, ...],
) -> tuple[dict[str, object], list[str]]:
    command = """
import importlib
import json
import sys

records = {}
for name in sys.argv[1:]:
    try:
        module = importlib.import_module(name)
        records[name] = {
            "file": module.__file__,
            "version": str(getattr(module, "__version__", "unknown")),
        }
    except BaseException as error:
        records[name] = {
            "error": f"{type(error).__name__}: {error}",
        }
print("FLAGDNN_MODULE_IDENTITIES=" + json.dumps(records, sort_keys=True))
"""
    result = _run(
        [sys.executable, "-c", command, *names], timeout=360
    )
    if result.returncode != 0:
        diagnostic = result.stderr.strip().splitlines()
        suffix = diagnostic[-1] if diagnostic else f"exit {result.returncode}"
        raise CollectionError(
            f"cannot collect codegen Python modules: {suffix}"
        )
    prefix = "FLAGDNN_MODULE_IDENTITIES="
    payload = next(
        (
            line.removeprefix(prefix)
            for line in reversed(result.stdout.splitlines())
            if line.startswith(prefix)
        ),
        "",
    )
    try:
        metadata = json.loads(payload)
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise CollectionError(
            f"invalid Python module identity output: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise CollectionError("Python module identity output is not an object")
    records: dict[str, object] = {}
    errors: list[str] = []
    for name in names:
        value = metadata.get(name)
        if not isinstance(value, dict):
            errors.append(f"codegen Python cannot import {name}: no record")
        elif isinstance(value.get("error"), str):
            errors.append(
                f"codegen Python cannot import {name}: {value['error']}"
            )
        else:
            try:
                records[name] = _file_record(
                    Path(value["file"]), version=str(value["version"])
                )
            except (KeyError, TypeError, OSError) as error:
                errors.append(
                    f"invalid Python module identity for {name}: {error}"
                )
    return records, errors


def _patchelf_record() -> dict[str, object]:
    executable = _require_command("patchelf")
    result = _run([str(executable), "--version"], timeout=30)
    if result.returncode != 0:
        raise CollectionError(
            f"patchelf --version failed: {result.stderr.strip()}"
        )
    match = re.search(r"\bpatchelf\s+([^\s]+)", result.stdout)
    if match is None:
        raise CollectionError(
            f"cannot parse patchelf version: {result.stdout.strip()}"
        )
    try:
        distribution_version = importlib.metadata.version("patchelf")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = "not-installed"
    return _file_record(
        executable,
        version=match.group(1),
        distribution_version=distribution_version,
    )


def identity_sha256(document: dict[str, object]) -> str:
    def canonical_identity_value(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: canonical_identity_value(item)
                for key, item in value.items()
                if key != "requested_path"
            }
        if isinstance(value, list):
            return [canonical_identity_value(item) for item in value]
        return value

    payload = canonical_identity_value(document)
    assert isinstance(payload, dict)
    payload.pop("identity_sha256", None)
    return _sha256_bytes(canonical_json(payload))


def _collect_required(
    resources: dict[str, object],
    errors: list[str],
    name: str,
    collector: Any,
) -> None:
    try:
        resources[name] = collector()
    except (CollectionError, OSError, subprocess.SubprocessError) as error:
        errors.append(str(error))


def collect_environment() -> dict[str, object]:
    errors: list[str] = []
    resources: dict[str, object] = {}

    musa_home_value = os.environ.get(
        "FLAGDNN_MTHREADS_MUSA_ROOT"
    ) or os.environ.get("MUSA_HOME")
    if not musa_home_value:
        errors.append("MUSA_HOME is required")
        musa_root = Path("/usr/local/musa")
    else:
        musa_root = Path(musa_home_value)

    virtual_environment = Path(os.environ.get("VIRTUAL_ENV", sys.prefix))
    jit_prefix = Path(
        os.environ.get("MTHREADS_TRITON_JIT_PREFIX")
        or os.environ.get("FLAGDNN_MTHREADS_TRITON_JIT_ROOT")
        or "/usr/local"
    )
    configured_jit_dir = os.environ.get(
        "FLAGDNN_MTHREADS_TRITON_JIT_DIR", ""
    )
    jit_config = Path(
        os.environ.get(
            "MTHREADS_TRITON_JIT_CONFIG",
            str(
                Path(configured_jit_dir) / "TritonJITConfig.cmake"
                if configured_jit_dir
                else jit_prefix
                / "lib"
                / "cmake"
                / "TritonJIT"
                / "TritonJITConfig.cmake"
            ),
        )
    )
    jit_build_root = (
        jit_config.parent if (jit_config.parent / "CMakeCache.txt").is_file() else None
    )
    jit_default_library = (
        jit_build_root / "src" / "libtriton_jit.so"
        if jit_build_root is not None
        else jit_prefix / "lib" / "libtriton_jit.so"
    )
    jit_script_dir = (
        jit_prefix / "scripts"
        if jit_build_root is not None
        else jit_prefix / "share" / "triton_jit" / "scripts"
    )
    jit_library = Path(
        os.environ.get(
            "MTHREADS_TRITON_JIT_LIBRARY",
            os.environ.get("FLAGDNN_MTHREADS_TRITON_JIT_LIBRARY")
            or str(jit_default_library),
        )
    )

    _collect_required(
        resources, errors, "musa_root", lambda: _directory_record(musa_root)
    )
    _collect_required(
        resources,
        errors,
        "musa_runtime_header",
        lambda: _file_record(musa_root / "include" / "musa_runtime_api.h"),
    )
    _collect_required(
        resources,
        errors,
        "musa_driver_header",
        lambda: _file_record(musa_root / "include" / "musa.h"),
    )
    _collect_required(
        resources,
        errors,
        "mudnn_header",
        lambda: _file_record(musa_root / "include" / "mudnn.h"),
    )
    _collect_required(
        resources,
        errors,
        "musa_runtime",
        lambda: _file_record(musa_root / "lib" / "libmusart.so"),
    )
    _collect_required(
        resources,
        errors,
        "mudnn",
        lambda: _file_record(musa_root / "lib" / "libmudnn.so"),
    )
    _collect_required(
        resources,
        errors,
        "mcc",
        lambda: _file_record(musa_root / "bin" / "mcc"),
    )
    _collect_required(
        resources,
        errors,
        "patchelf",
        _patchelf_record,
    )

    driver_search = [
        *(Path(item) for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/lib/x86_64-linux-gnu"),
        musa_root / "lib",
    ]
    _collect_required(
        resources,
        errors,
        "musa_driver",
        lambda: _file_record(_resolve_library("libmusa.so", driver_search)),
    )

    _collect_required(
        resources,
        errors,
        "triton_jit_prefix",
        lambda: _directory_record(jit_prefix),
    )

    if jit_build_root is not None:
        _collect_required(
            resources,
            errors,
            "triton_jit_build_root",
            lambda: _directory_record(jit_build_root),
        )

    def collect_jit() -> dict[str, object]:
        backend = _parse_cmake_backend(jit_config)
        soname, runpath = _elf_metadata(jit_library)
        dependencies = _ldd_dependencies(jit_library)
        return _file_record(
            jit_library,
            backend=backend,
            soname=soname,
            runpath=runpath,
            dependencies=dependencies,
        )

    _collect_required(resources, errors, "triton_jit", collect_jit)
    _collect_required(
        resources,
        errors,
        "triton_jit_config",
        lambda: _file_record(jit_config),
    )
    _collect_required(
        resources,
        errors,
        "triton_jit_header",
        lambda: _file_record(
            jit_prefix / "include" / "triton_jit" / "triton_jit_function.h"
        ),
    )
    _collect_required(
        resources,
        errors,
        "triton_jit_gen_ssig",
        lambda: _file_record(jit_script_dir / "gen_ssig.py"),
    )
    _collect_required(
        resources,
        errors,
        "triton_jit_standalone_compile",
        lambda: _file_record(jit_script_dir / "standalone_compile.py"),
    )

    modules: dict[str, object] = {}
    try:
        modules, module_errors = _module_records(REQUIRED_PYTHON_MODULES)
        errors.extend(module_errors)
    except (CollectionError, OSError, subprocess.SubprocessError) as error:
        errors.append(str(error))

    python: dict[str, object] = {
        "executable": _file_record(Path(sys.executable)),
        "prefix": str(virtual_environment.resolve(strict=False)),
        "version": ".".join(str(part) for part in sys.version_info[:3]),
        "modules": modules,
    }

    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "platform": PLATFORM,
        "resources": resources,
        "python": python,
        "errors": sorted(set(errors)),
        "identity_sha256": "",
    }
    document["identity_sha256"] = identity_sha256(document)
    return document


def _mapping(
    value: object, name: str, errors: list[str]
) -> dict[str, object] | None:
    if not isinstance(value, dict):
        errors.append(f"{name} must be an object")
        return None
    return value


def _validate_path_record(
    name: str,
    record: dict[str, object],
    errors: list[str],
) -> Path | None:
    requested_value = record.get("requested_path")
    realpath_value = record.get("realpath")
    if not isinstance(requested_value, str) or not Path(requested_value).is_absolute():
        errors.append(f"resource {name} requested_path must be absolute")
        return None
    if not isinstance(realpath_value, str) or not Path(realpath_value).is_absolute():
        errors.append(f"resource {name} realpath must be absolute")
        return None
    requested = Path(requested_value)
    recorded = Path(realpath_value)
    try:
        actual = requested.resolve(strict=True)
    except OSError:
        errors.append(f"resource {name} requested path does not exist")
        return None
    try:
        canonical_recorded = recorded.resolve(strict=True)
    except OSError:
        errors.append(f"resource {name} realpath does not exist")
        return None
    if actual != recorded or canonical_recorded != recorded:
        errors.append(f"resource {name} realpath is not canonical")
        return None

    has_file_fields = "size" in record or "sha256" in record
    if has_file_fields:
        if not actual.is_file():
            errors.append(f"resource {name} is not a regular file")
            return actual
        size = record.get("size")
        if not isinstance(size, int) or isinstance(size, bool):
            errors.append(f"resource {name} size must be an integer")
        elif actual.stat().st_size != size:
            errors.append(f"resource {name} size does not match current bytes")
        digest = record.get("sha256")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            errors.append(f"resource {name} SHA-256 is invalid")
        elif _sha256_file(actual) != digest:
            errors.append(
                f"resource {name} SHA-256 does not match current bytes"
            )
    elif not actual.is_dir():
        errors.append(f"resource {name} is not a directory")
    return actual


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_environment(document: dict[str, object]) -> list[str]:
    errors: list[str] = []
    actual_keys = set(document)
    missing_keys = sorted(EXPECTED_TOP_LEVEL_KEYS - actual_keys)
    unexpected_keys = sorted(actual_keys - EXPECTED_TOP_LEVEL_KEYS)
    if missing_keys:
        errors.append(f"missing top-level keys: {','.join(missing_keys)}")
    if unexpected_keys:
        errors.append(f"unexpected top-level keys: {','.join(unexpected_keys)}")
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if document.get("platform") != PLATFORM:
        errors.append(f"platform must be {PLATFORM}")
    if not isinstance(document.get("errors"), list):
        errors.append("errors must be an array")

    resources = _mapping(document.get("resources"), "resources", errors)
    resolved_resources: dict[str, Path] = {}
    if resources is not None:
        for name in REQUIRED_RESOURCES:
            record = _mapping(
                resources.get(name), f"resource {name}", errors
            )
            if record is None:
                continue
            resolved = _validate_path_record(name, record, errors)
            if resolved is not None:
                resolved_resources[name] = resolved

        musa_root = resolved_resources.get("musa_root")
        if musa_root is not None:
            for name in MUSA_ROOT_RESOURCES:
                path = resolved_resources.get(name)
                if path is not None and not _under(path, musa_root):
                    errors.append(
                        "MUSA and muDNN resources do not share the selected root"
                    )

        jit_build_root = None
        if "triton_jit_build_root" in resources:
            build_record = _mapping(
                resources["triton_jit_build_root"], "triton_jit_build_root", errors
            )
            if build_record is not None:
                jit_build_root = _validate_path_record(
                    "triton_jit_build_root", build_record, errors
                )
        jit_prefix = resolved_resources.get("triton_jit_prefix")
        if jit_prefix is not None:
            for name in TRITON_JIT_PREFIX_RESOURCES:
                path = resolved_resources.get(name)
                root = (
                    jit_build_root
                    if jit_build_root is not None
                    and name in ("triton_jit", "triton_jit_config")
                    else jit_prefix
                )
                if path is not None and not _under(path, root):
                    errors.append(
                        "TritonJIT resources do not share the selected prefix"
                    )

        jit = resources.get("triton_jit")
        if isinstance(jit, dict):
            if jit.get("backend") != "MUSA":
                errors.append("TritonJIT backend must be MUSA")
            soname = jit.get("soname")
            if not isinstance(soname, str) or not soname:
                errors.append("TritonJIT SONAME is missing")
            runpath = jit.get("runpath")
            if not isinstance(runpath, list) or not all(
                isinstance(item, str) for item in runpath
            ):
                errors.append("TritonJIT RUNPATH must be an array of strings")
            dependencies = jit.get("dependencies")
            if not isinstance(dependencies, list):
                errors.append("TritonJIT dependencies must be an array")
            else:
                dependency_names: set[str] = set()
                for dependency in dependencies:
                    if not isinstance(dependency, dict):
                        errors.append(
                            "TritonJIT dependency record must be an object"
                        )
                        continue
                    name = dependency.get("name")
                    resolved = dependency.get("resolved")
                    if not isinstance(name, str) or not name:
                        errors.append("TritonJIT dependency name is missing")
                        continue
                    dependency_names.add(name)
                    if resolved is None:
                        errors.append(
                            f"TritonJIT dependency not found: {name}"
                        )
                    elif not isinstance(resolved, str) or not Path(
                        resolved
                    ).is_absolute():
                        errors.append(
                            f"TritonJIT dependency path is invalid: {name}"
                        )
                required_dependency_groups = {
                    "libpython": lambda name: name.startswith("libpython"),
                    "Torch": lambda name: name.startswith("libtorch")
                    or name.startswith("libc10"),
                    "MUSA": lambda name: name.startswith("libmusa"),
                    "C++ runtime": lambda name: name.startswith("libstdc++"),
                }
                for label, predicate in required_dependency_groups.items():
                    if not any(predicate(name) for name in dependency_names):
                        errors.append(
                            f"TritonJIT dependency set is missing {label}"
                        )

        patchelf = resources.get("patchelf")
        if isinstance(patchelf, dict):
            if not isinstance(patchelf.get("version"), str) or not patchelf.get(
                "version"
            ):
                errors.append("patchelf version evidence is missing")
            if not isinstance(
                patchelf.get("distribution_version"), str
            ) or not patchelf.get("distribution_version"):
                errors.append(
                    "patchelf distribution version evidence is missing"
                )

    python = _mapping(document.get("python"), "python", errors)
    if python is not None:
        executable = _mapping(
            python.get("executable"), "python executable", errors
        )
        if executable is not None:
            _validate_path_record("python_executable", executable, errors)
        modules = _mapping(python.get("modules"), "python modules", errors)
        if modules is not None:
            for name in REQUIRED_PYTHON_MODULES:
                record = modules.get(name)
                if not isinstance(record, dict):
                    errors.append(f"codegen Python cannot import {name}")
                    continue
                _validate_path_record(f"python_{name}", record, errors)
    declared_identity = document.get("identity_sha256")
    if (
        not isinstance(declared_identity, str)
        or SHA256_PATTERN.fullmatch(declared_identity) is None
        or identity_sha256(document) != declared_identity
    ):
        errors.append("environment identity SHA-256 mismatch")

    return sorted(set(errors))


def _finalize(document: dict[str, object]) -> dict[str, object]:
    collection_errors = document.get("errors")
    if not isinstance(collection_errors, list):
        collection_errors = ["errors must be an array"]
    document["identity_sha256"] = identity_sha256(document)
    validation_errors = [
        error
        for error in validate_environment(document)
        if error != "environment identity SHA-256 mismatch"
    ]
    document["errors"] = sorted(
        set(str(error) for error in collection_errors) | set(validation_errors)
    )
    document["identity_sha256"] = identity_sha256(document)
    return document


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect the active mthreads compiler environment identity"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        document = _finalize(collect_environment())
    except (CollectionError, OSError, subprocess.SubprocessError) as error:
        document = {
            "schema_version": SCHEMA_VERSION,
            "platform": PLATFORM,
            "resources": {},
            "python": {},
            "errors": [str(error)],
            "identity_sha256": "",
        }
        document["identity_sha256"] = identity_sha256(document)
    payload = canonical_json(document) + b"\n"
    write_atomic(arguments.output, payload)
    for error in document["errors"]:
        print(error, file=sys.stderr)
    return 0 if not document["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
