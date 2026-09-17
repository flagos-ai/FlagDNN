# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hygon compiler identity and artifact-cache invalidation inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

from ..dispatch import nn as compiler_nn
from ..dispatch import tensor as compiler_tensor
from ..dispatch.graph import SUPPORTED_OPERATIONS as _SUPPORTED_OPERATIONS
from .. import python_environment_identity

from flagdnn_codegen.kernel_registry import (
    iter_kernel_registry_sources,
    resolve_kernel_source,
    resolve_tuning_source,
    select_kernel_candidate,
)


_SUPPORTED_TARGET = "gfx936"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TRITON_JIT_SONAME_PATTERN = re.compile(r"libtriton_jit\.so(?:\.[0-9]+)*")
_COMPILER_ENVIRONMENT_FILE = "flagdnn_hygon_compiler_environment.json"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _runtime_python_environment() -> dict[str, Any]:
    """Describe the Python ABI and packages used by HCU libtriton_jit."""

    return python_environment_identity.runtime_python_environment()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _script_bundle_sha256(
    standalone: str, gen_ssig: str, environment_helper: str
) -> str:
    return hashlib.sha256(
        f"{standalone}:{gen_ssig}:{environment_helper}".encode("ascii")
    ).hexdigest()


def _jit_script_directory_candidates(source: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    configured = os.environ.get("FLAGDNN_HYGON_TRITON_JIT_SCRIPT_DIR", "")
    if configured:
        candidates.append(Path(configured).expanduser())
    root = os.environ.get("FLAGDNN_HYGON_TRITON_JIT_ROOT", "")
    if root:
        root_path = Path(root).expanduser()
        candidates.extend(
            (root_path / "scripts", root_path / "share/triton_jit/scripts")
        )
    triton_jit_dir = os.environ.get("FLAGDNN_HYGON_TRITON_JIT_DIR", "")
    if triton_jit_dir:
        directory = Path(triton_jit_dir).expanduser()
        candidates.extend(
            (
                directory / "../scripts",
                directory / "../../../share/triton_jit/scripts",
            )
        )
    for entry in os.environ.get("FLAGDNN_BACKEND_PATH", "").split(os.pathsep):
        if entry:
            candidates.append(
                Path(entry).expanduser() / "flagdnn/share/triton_jit/scripts"
            )
    candidates.append(source.parent / "flagdnn/share/triton_jit/scripts")
    for parent in (source.parent, *tuple(source.parents)[:6]):
        candidates.append(parent / "share/triton_jit/scripts")

    return tuple(candidates)


def _jit_script_directories(source: Path) -> tuple[Path, ...]:
    candidates = _jit_script_directory_candidates(source)

    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _discover_jit_script_hashes(
    source: Path,
) -> tuple[str, str, str, Path | None, Path | None]:
    local_helper = source.with_name("python_environment_identity.py")
    for directory in _jit_script_directories(source):
        standalone = directory / "standalone_compile.py"
        gen_ssig = directory / "gen_ssig.py"
        installed_helper = directory / "flagdnn_python_environment_identity.py"
        helper = (
            installed_helper if installed_helper.is_file() else local_helper
        )
        if standalone.is_file() and gen_ssig.is_file() and helper.is_file():
            return (
                _sha256_file(standalone),
                _sha256_file(gen_ssig),
                _sha256_file(helper),
                directory,
                helper,
            )
    missing = hashlib.sha256(b"unconfigured-libtriton-jit-scripts").hexdigest()
    return missing, missing, missing, None, None


def _compiler_environment_candidate_paths(
    provider_path: Path,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    configured = os.environ.get("FLAGDNN_HYGON_COMPILER_ENVIRONMENT", "")
    if configured:
        candidates.append(Path(configured).expanduser())
    for entry in os.environ.get("FLAGDNN_BACKEND_PATH", "").split(os.pathsep):
        if entry:
            candidates.append(
                Path(entry).expanduser() / _COMPILER_ENVIRONMENT_FILE
            )
    candidates.append(provider_path.with_name(_COMPILER_ENVIRONMENT_FILE))

    return tuple(candidates)


def _compiler_environment_candidates(provider_path: Path) -> tuple[Path, ...]:
    candidates = _compiler_environment_candidate_paths(provider_path)

    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _install_relative_path(
    value: object,
    *,
    source: Path,
    name: str,
    suffix: tuple[str, ...],
) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError(f"{source}: {name} must be a safe relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or tuple(parsed.parts[-len(suffix) :]) != suffix:
        raise ValueError(f"{source}: {name} has an invalid SDK layout")
    return Path(*parsed.parts)


def _compiler_environment_resource_layout(
    value: object, source: Path
) -> tuple[str, Path, Path]:
    if not isinstance(value, dict):
        raise ValueError(f"{source}: compiler environment must be an object")
    soname = value.get("libtriton_jit_soname")
    if (
        not isinstance(soname, str)
        or _TRITON_JIT_SONAME_PATTERN.fullmatch(soname) is None
    ):
        raise ValueError(f"{source}: libtriton_jit_soname is invalid")
    library = _install_relative_path(
        value.get("libtriton_jit_install_relative_path"),
        source=source,
        name="libtriton_jit_install_relative_path",
        suffix=("flagdnn", "hygon", soname),
    )
    scripts = _install_relative_path(
        value.get("jit_script_install_relative_path"),
        source=source,
        name="jit_script_install_relative_path",
        suffix=("flagdnn", "share", "triton_jit", "scripts"),
    )
    if library.parts[:-3] != scripts.parts[:-4]:
        raise ValueError(
            f"{source}: private JIT library and scripts use different libdirs"
        )
    return soname, library, scripts


def _configured_resource_layout(
    provider_path: Path,
) -> tuple[str, Path, Path]:
    for candidate in _compiler_environment_candidates(provider_path):
        if not candidate.is_file():
            continue
        if candidate.stat().st_size > 64 * 1024:
            raise ValueError(f"{candidate}: compiler environment is too large")
        configured = json.loads(candidate.read_text(encoding="utf-8"))
        return _compiler_environment_resource_layout(configured, candidate)
    soname = "libtriton_jit.so"
    return (
        soname,
        Path("lib/flagdnn/hygon") / soname,
        Path("lib/flagdnn/share/triton_jit/scripts"),
    )


def _validate_compiler_environment(
    value: object, source: Path, runtime_environment_sha256: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != 3:
        raise ValueError(
            f"{source}: unsupported Hygon compiler environment schema"
        )
    soname, library_relative, private_scripts_relative = (
        _compiler_environment_resource_layout(value, source)
    )
    required_hashes = (
        "libtriton_jit_sha256",
        "triton_jit_provenance_sha256",
        "python_environment_sha256",
        "python_environment_helper_sha256",
        "standalone_compile_sha256",
        "gen_ssig_sha256",
        "jit_scripts_sha256",
    )
    for name in required_hashes:
        digest = value.get(name)
        if (
            not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError(f"{source}: {name} must be a SHA-256 digest")
    build_identity = value.get("build_identity")
    if (
        not isinstance(build_identity, str)
        or re.fullmatch(
            r"[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}", build_identity
        )
        is None
    ):
        raise ValueError(f"{source}: build_identity is invalid")
    expected_build_identity = (
        f"{value['triton_jit_provenance_sha256'][:16]}-"
        f"{value['python_environment_sha256'][:16]}-"
        f"{value['jit_scripts_sha256'][:16]}"
    )
    if build_identity != expected_build_identity:
        raise ValueError(
            f"{source}: build_identity does not match its "
            "JIT/environment hashes"
        )
    if value["python_environment_sha256"] != runtime_environment_sha256:
        raise RuntimeError(
            "the active Python/Torch/Triton environment does not match the "
            f"Hygon backend selected at CMake configure time ({source})"
        )
    expected_scripts = _script_bundle_sha256(
        value["standalone_compile_sha256"],
        value["gen_ssig_sha256"],
        value["python_environment_helper_sha256"],
    )
    if value["jit_scripts_sha256"] != expected_scripts:
        raise ValueError(
            f"{source}: jit_scripts_sha256 does not match its script hashes"
        )

    result = {
        "schema_version": 3,
        "origin": "cmake",
        "libtriton_jit_soname": soname,
        "libtriton_jit_install_relative_path": library_relative.as_posix(),
        "jit_script_install_relative_path": (
            private_scripts_relative.as_posix()
        ),
        "libtriton_jit_sha256": value["libtriton_jit_sha256"],
        "triton_jit_provenance_sha256": value["triton_jit_provenance_sha256"],
        "python_environment_sha256": value["python_environment_sha256"],
        "python_environment_helper_sha256": value[
            "python_environment_helper_sha256"
        ],
        "standalone_compile_sha256": value["standalone_compile_sha256"],
        "gen_ssig_sha256": value["gen_ssig_sha256"],
        "jit_scripts_sha256": value["jit_scripts_sha256"],
        "build_identity": build_identity,
    }

    # The build tree mirrors the installed Hygon-private layout below the
    # plugin directory. Installed layouts carry the exact exported SONAME in
    # the private libdir recorded by CMake, so another backend's root-level
    # SONAME cannot satisfy this identity check.
    library_candidates = [source.parent / "flagdnn/hygon" / soname]
    for parent in tuple(source.parents)[:6]:
        library_candidates.append(parent / library_relative)
    for library in library_candidates:
        if not library.is_file():
            continue
        actual = _sha256_file(library)
        if actual != value["libtriton_jit_sha256"]:
            raise RuntimeError(
                f"{library}: libtriton_jit does not match the Hygon CMake "
                "selection"
            )
        result["verified_library_sha256"] = actual
        break
    else:
        raise RuntimeError(
            f"{source}: cannot locate the CMake-selected libtriton_jit image"
        )

    # Verify the path libtriton_jit itself will resolve. Both the build tree
    # and an installed SDK use the same private layout relative to the plugin.
    # Rebase the CMake-recorded install-relative directory from the environment
    # file instead of relying on generic source-tree discovery: an installed
    # provider lives below share/ while its private JIT scripts live below
    # lib/, so those trees have no fixed source-relative relationship.
    private_script_candidates = [
        source.parent / "flagdnn/share/triton_jit/scripts"
    ]
    for parent in tuple(source.parents)[:6]:
        private_script_candidates.append(parent / private_scripts_relative)
    for directory in private_script_candidates:
        private_standalone = directory / "standalone_compile.py"
        private_gen_ssig = directory / "gen_ssig.py"
        private_helper = directory / "flagdnn_python_environment_identity.py"
        if not (
            private_standalone.is_file()
            and private_gen_ssig.is_file()
            and private_helper.is_file()
        ):
            continue
        if (
            _sha256_file(private_standalone)
            != value["standalone_compile_sha256"]
            or _sha256_file(private_gen_ssig) != value["gen_ssig_sha256"]
            or _sha256_file(private_helper)
            != value["python_environment_helper_sha256"]
        ):
            raise RuntimeError(
                f"{directory}: private libtriton_jit scripts do not match "
                "the Hygon CMake selection"
            )
        result["verified_private_jit_script_directory"] = str(
            directory.resolve()
        )
        result["verified_jit_scripts_sha256"] = _script_bundle_sha256(
            value["standalone_compile_sha256"],
            value["gen_ssig_sha256"],
            value["python_environment_helper_sha256"],
        )
        break
    else:
        raise RuntimeError(
            f"{source}: cannot locate the private libtriton_jit scripts"
        )
    return result


def _libtriton_jit_candidate_paths(provider_path: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    soname, library_relative, _ = _configured_resource_layout(provider_path)
    explicit = os.environ.get("FLAGDNN_HYGON_TRITON_JIT_LIBRARY", "")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    root = os.environ.get("FLAGDNN_HYGON_TRITON_JIT_ROOT", "")
    if root:
        root_path = Path(root).expanduser()
        candidates.extend(
            (
                root_path / "build/src/libtriton_jit.so",
                root_path / "lib/libtriton_jit.so",
                root_path / "lib64/libtriton_jit.so",
                root_path / "lib/flagdnn/hygon" / soname,
                root_path / "lib64/flagdnn/hygon" / soname,
            )
        )
    triton_jit_dir = os.environ.get("FLAGDNN_HYGON_TRITON_JIT_DIR", "")
    if triton_jit_dir:
        directory = Path(triton_jit_dir).expanduser()
        candidates.extend(
            (
                directory / "src/libtriton_jit.so",
                directory / "../../../lib/libtriton_jit.so",
                directory / "../../../lib64/libtriton_jit.so",
            )
        )
    for environment_file in _compiler_environment_candidate_paths(
        provider_path
    ):
        candidates.append(environment_file.parent / "flagdnn/hygon" / soname)
        for parent in tuple(environment_file.parents)[:6]:
            candidates.append(parent / library_relative)
    candidates.append(
        provider_path.resolve().parents[3]
        / "libtriton_jit/build/src/libtriton_jit.so"
    )
    return tuple(candidates)


def _libtriton_jit_candidates(provider_path: Path) -> tuple[Path, ...]:
    candidates = _libtriton_jit_candidate_paths(provider_path)
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _discover_libtriton_jit(provider_path: Path) -> str:
    """Best-effort identity for direct, non-CMake provider invocations."""

    for resolved in _libtriton_jit_candidates(provider_path):
        if resolved.is_file():
            return _sha256_file(resolved)
    return hashlib.sha256(b"unconfigured-libtriton-jit").hexdigest()


def _compiler_environment(
    provider_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_environment = _runtime_python_environment()
    runtime_environment_sha256 = hashlib.sha256(
        _canonical(runtime_environment)
    ).hexdigest()
    for candidate in _compiler_environment_candidates(provider_path):
        if not candidate.is_file():
            continue
        if candidate.stat().st_size > 64 * 1024:
            raise ValueError(f"{candidate}: compiler environment is too large")
        configured = json.loads(candidate.read_text(encoding="utf-8"))
        return runtime_environment, _validate_compiler_environment(
            configured, candidate, runtime_environment_sha256
        )
    jit_sha256 = _discover_libtriton_jit(provider_path)
    (
        standalone_sha256,
        gen_ssig_sha256,
        environment_helper_sha256,
        _,
        _,
    ) = _discover_jit_script_hashes(provider_path)
    scripts_sha256 = _script_bundle_sha256(
        standalone_sha256, gen_ssig_sha256, environment_helper_sha256
    )
    provenance_sha256 = hashlib.sha256(
        f"runtime-discovery:{jit_sha256}:{scripts_sha256}".encode("ascii")
    ).hexdigest()
    return runtime_environment, {
        "schema_version": 3,
        "origin": "runtime-discovery",
        "libtriton_jit_soname": "libtriton_jit.so",
        "libtriton_jit_install_relative_path": (
            "lib/flagdnn/hygon/libtriton_jit.so"
        ),
        "jit_script_install_relative_path": (
            "lib/flagdnn/share/triton_jit/scripts"
        ),
        "libtriton_jit_sha256": jit_sha256,
        "triton_jit_provenance_sha256": provenance_sha256,
        "python_environment_sha256": runtime_environment_sha256,
        "python_environment_helper_sha256": environment_helper_sha256,
        "standalone_compile_sha256": standalone_sha256,
        "gen_ssig_sha256": gen_ssig_sha256,
        "jit_scripts_sha256": scripts_sha256,
        "build_identity": (
            f"{provenance_sha256[:16]}-{runtime_environment_sha256[:16]}-"
            f"{scripts_sha256[:16]}"
        ),
    }


def _identity_inputs(
    provider_path: Path, compiler_entry: Path
) -> dict[str, Path]:
    provider_path = provider_path.resolve()
    compiler_entry = compiler_entry.resolve()
    identity_inputs: dict[str, Path] = {
        "provider:hygon": provider_path,
        "provider_identity:hygon": Path(__file__).resolve(),
        "provider_tensor:hygon": Path(compiler_tensor.__file__).resolve(),
        "provider_nn:hygon": Path(compiler_nn.__file__).resolve(),
        "provider_environment:hygon": Path(
            python_environment_identity.__file__
        ).resolve(),
        "driver": compiler_entry,
    }
    for name in ("kernel_registry.py", "provider_loader.py"):
        source = compiler_entry.with_name(name)
        if source.is_file():
            identity_inputs[f"codegen:{name}"] = source

    for package in ("codegen", "dispatch"):
        for source in sorted((provider_path.parent / package).rglob("*.py")):
            label = source.relative_to(provider_path.parent).as_posix()
            identity_inputs[f"provider_module:hygon:{label}"] = (
                source.resolve()
            )

    resource_root = compiler_entry.parents[2]
    for registry_path in iter_kernel_registry_sources("hygon"):
        label = registry_path.relative_to(resource_root).as_posix()
        identity_inputs[f"registry:{label}"] = registry_path.resolve()

    for operation in _SUPPORTED_OPERATIONS:
        candidate = select_kernel_candidate("hygon", operation)
        kernel_source = resolve_kernel_source(
            compiler_entry, candidate
        ).resolve()
        kernel_label = ":".join(
            (
                "kernel",
                candidate.ownership,
                candidate.source_layout,
                candidate.provider,
                candidate.source,
            )
        )
        previous_kernel = identity_inputs.setdefault(
            kernel_label, kernel_source
        )
        if previous_kernel != kernel_source:
            raise RuntimeError(
                f"compiler identity kernel label collision: {kernel_label}"
            )
        if candidate.tuning is not None:
            tuning_source = resolve_tuning_source(
                compiler_entry, candidate
            ).resolve()
            tuning_label = ":".join(
                (
                    "tuning",
                    candidate.ownership,
                    candidate.source_layout,
                    candidate.provider,
                    candidate.tuning.source,
                )
            )
            previous_tuning = identity_inputs.setdefault(
                tuning_label, tuning_source
            )
            if previous_tuning != tuning_source:
                raise RuntimeError(
                    f"compiler identity tuning label collision: "
                    f"{tuning_label}"
                )
    return identity_inputs


def _configured_roots(environment_name: str) -> tuple[Path, ...]:
    return tuple(
        Path(value).expanduser()
        for value in os.environ.get(environment_name, "").split(os.pathsep)
        if value
    )


def _kernel_dependency_candidates(
    compiler_entry: Path,
) -> tuple[Path, ...]:
    """Return logical roots/files involved in kernel and tuning selection."""

    resource_root = compiler_entry.resolve().parents[2]
    dependencies: list[Path] = []
    seen_candidates: set[tuple[str, str, str | None]] = set()
    for operation in _SUPPORTED_OPERATIONS:
        candidate = select_kernel_candidate("hygon", operation)
        candidate_key = (
            candidate.source_layout,
            candidate.source,
            candidate.tuning.source if candidate.tuning is not None else None,
        )
        if candidate_key in seen_candidates:
            continue
        seen_candidates.add(candidate_key)

        if candidate.source_layout == "kernels":
            source_roots = [
                *_configured_roots("FLAGDNN_KERNEL_SOURCE_ROOT"),
                resource_root / "kernels/common",
            ]
        elif candidate.source_layout == "platform":
            source_roots = [
                *_configured_roots("FLAGDNN_BACKEND_ROOT"),
                resource_root / "backends/hygon/kernels",
            ]
        else:
            raise ValueError(
                f"unknown kernel source layout: {candidate.source_layout!r}"
            )
        for root in source_roots:
            dependencies.extend((root, root / candidate.source))

        if candidate.tuning is not None:
            tuning_roots = [
                *source_roots,
                *_configured_roots("FLAGDNN_TUNING_ROOT"),
                resource_root / "backends/hygon/tuning",
            ]
            for root in tuning_roots:
                dependencies.extend((root, root / candidate.tuning.source))
    return tuple(dependencies)


def compiler_identity_dependency_paths(
    *, provider_path: Path, compiler_entry: Path
) -> tuple[Path, ...]:
    """Return resources whose metadata safely guards an identity memo."""

    dependencies: set[Path] = set()

    def add_dependency(path: Path) -> None:
        logical = Path(os.path.abspath(os.fspath(path.expanduser())))
        dependencies.add(logical)
        dependencies.add(path.expanduser().resolve())

    for path in (
        provider_path,
        compiler_entry,
        Path(__file__),
        Path(compiler_tensor.__file__),
        Path(compiler_nn.__file__),
        Path(python_environment_identity.__file__),
    ):
        add_dependency(path)
    # Track package directories so additions/removals invalidate the memo too.
    for package in ("codegen", "dispatch"):
        directory = provider_path.resolve().parent / package
        add_dependency(directory)
        for child in directory.rglob("*"):
            if child.is_dir() and "__pycache__" not in child.parts:
                add_dependency(child)
    for path in _identity_inputs(provider_path, compiler_entry).values():
        add_dependency(path)
    for path in iter_kernel_registry_sources("hygon"):
        add_dependency(path)
    for path in _kernel_dependency_candidates(compiler_entry):
        add_dependency(path)
    environment_helper = python_environment_identity
    environment_paths = (
        environment_helper.runtime_python_environment_dependency_paths()
    )
    for path in environment_paths:
        add_dependency(path)
    for path in _compiler_environment_candidate_paths(provider_path):
        add_dependency(path)
    for path in _libtriton_jit_candidate_paths(provider_path):
        add_dependency(path)
    for directory in _jit_script_directory_candidates(provider_path):
        add_dependency(directory)
        add_dependency(directory / "standalone_compile.py")
        add_dependency(directory / "gen_ssig.py")
        add_dependency(directory / "flagdnn_python_environment_identity.py")

    # The configured environment searches these deterministic locations for
    # its exact JIT image. Keep missing candidates too: creation of a
    # higher-priority file must invalidate the memo just like a file change.
    soname, library_relative, private_scripts_relative = (
        _configured_resource_layout(provider_path)
    )
    environment_files = (
        *_compiler_environment_candidate_paths(provider_path),
        *_compiler_environment_candidates(provider_path),
    )
    for environment_file in environment_files:
        add_dependency(environment_file.parent / "flagdnn/hygon" / soname)
        add_dependency(
            environment_file.parent / "flagdnn/share/triton_jit/scripts"
        )
        for parent in tuple(environment_file.parents)[:6]:
            add_dependency(parent / library_relative)
            private_scripts = parent / private_scripts_relative
            add_dependency(private_scripts)
            add_dependency(private_scripts / "standalone_compile.py")
            add_dependency(private_scripts / "gen_ssig.py")
            add_dependency(
                private_scripts / "flagdnn_python_environment_identity.py"
            )
    return tuple(sorted(dependencies))


def build_compiler_identity(
    target_name: str,
    execution_engine: str,
    *,
    provider_path: Path,
    compiler_entry: Path,
    provider_name: str,
    provider_version: str,
    graph_schema_version: int,
    artifact_schema_version: int,
    execution_program_version: int,
) -> dict[str, Any]:
    """Hash all inputs that can change a supported Hygon JIT artifact."""

    if target_name != _SUPPORTED_TARGET:
        raise ValueError(
            f"Hygon compiler identity supports only target "
            f"{_SUPPORTED_TARGET!r}"
        )
    if execution_engine != "libtriton_jit":
        raise ValueError("Hygon supports only the libtriton_jit engine")

    provider_path = provider_path.resolve()
    compiler_entry = compiler_entry.resolve()
    runtime_environment, compiler_environment = _compiler_environment(
        provider_path
    )
    identity_inputs = _identity_inputs(provider_path, compiler_entry)

    source_files = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in sorted(identity_inputs.items())
    }
    payload: dict[str, Any] = {
        "provider": provider_name,
        "provider_version": provider_version,
        "graph_schema_version": graph_schema_version,
        "artifact_schema_version": artifact_schema_version,
        "execution_program_version": execution_program_version,
        "execution_engine": execution_engine,
        "target": target_name,
        "target_backend": "hip",
        "target_warp_size": 64,
        "python_implementation": runtime_environment["python_implementation"],
        "python_version": runtime_environment["python_version"],
        "python_cache_tag": runtime_environment["python_cache_tag"],
        "torch_version": runtime_environment["torch_version"],
        "triton_version": runtime_environment["triton_version"],
        "yaml_version": runtime_environment["yaml_version"],
        "runtime_environment": runtime_environment,
        "cmake_environment": compiler_environment,
        "source_files": source_files,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload
