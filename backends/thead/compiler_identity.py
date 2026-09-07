# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed compiler identity for the FlagDNN THead Triton provider."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Iterable

from flagdnn_codegen import kernel_registry

from .python_environment_identity import (
    collect_environment_identity,
    module_dependency_paths,
    sha256_file,
)


PROVIDER_NAME = "thead_triton"
PROVIDER_VERSION = "1"
GRAPH_IR_SCHEMA_VERSION = 3
ARTIFACT_SCHEMA_VERSION = 1
EXECUTION_PROGRAM_VERSION = 1

_TARGET = re.compile(r"^ppu_[a-z0-9][a-z0-9_]{0,110}_cc([1-9][0-9]{1,2})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JIT_SONAME = re.compile(r"^libtriton_jit\.so(?:\.[0-9]+)*$")
_COMPILER_ENVIRONMENT_FILE = "flagdnn_thead_compiler_environment.json"
_JIT_HEADERS = (
    "triton_jit/backend_config.h",
    "triton_jit/backend_policy.h",
    "triton_jit/backends/cuda_backend.h",
    "triton_jit/jit_function_arg.h",
    "triton_jit/jit_utils.h",
    "triton_jit/kernel_metadata.h",
    "triton_jit/triton_jit_function.h",
    "triton_jit/triton_kernel.h",
)


def _required_directory(path: Path, description: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"cannot resolve {description}: {error}") from error
    if not resolved.is_dir():
        raise RuntimeError(f"{description} is not a directory: {resolved}")
    return resolved


def _required_file(
    path: Path,
    description: str,
    *,
    root: Path | None = None,
    executable: bool = False,
) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"cannot resolve {description}: {error}") from error
    if not resolved.is_file():
        raise RuntimeError(f"{description} is not a file: {resolved}")
    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"{description} escaped configured root {root}: {resolved}"
            ) from error
    if executable and not os.access(resolved, os.X_OK):
        raise RuntimeError(f"{description} is not executable: {resolved}")
    return resolved


def _provider_directory() -> Path:
    return Path(__file__).resolve().parent


def _resource_root() -> Path:
    configured = os.environ.get("FLAGDNN_THEAD_RESOURCE_ROOT", "")
    root = (
        Path(configured)
        if configured
        else Path(kernel_registry.__file__).resolve().parents[2]
    )
    result = _required_directory(root, "FlagDNN resource root")
    _required_file(
        result / "kernels/registry.json",
        "FlagDNN common kernel registry",
        root=result,
    )
    return result


def _validate_target(target: str, execution_engine: str) -> int:
    if not isinstance(target, str):
        raise ValueError("THead compiler target must be a string")
    match = _TARGET.fullmatch(target)
    if match is None:
        raise ValueError(
            "THead compiler target must use ppu_<model>_cc<capability>"
        )
    if execution_engine != "libtriton_jit":
        raise ValueError("THead supports only the libtriton_jit engine")
    capability = int(match.group(1))
    override = os.environ.get("TRITON_OVERRIDE_ARCH", "")
    if override and override != f"sm{capability}":
        raise ValueError(
            "TRITON_OVERRIDE_ARCH conflicts with the THead target capability"
        )
    _normalize_jit_environment(capability)
    return capability


def _normalize_jit_environment(capability: int) -> None:
    """Make implicit and explicit THead JIT defaults identity-equivalent."""

    expected_backend = "CUDA"
    backend = os.environ.get("TRITON_JIT_BACKEND", "")
    if backend and backend != expected_backend:
        raise ValueError(
            "TRITON_JIT_BACKEND conflicts with the THead CUDA JIT backend"
        )
    os.environ["TRITON_JIT_BACKEND"] = expected_backend

    expected_architecture = f"sm{capability}"
    architecture = os.environ.get("TRITON_OVERRIDE_ARCH", "")
    if architecture and architecture != expected_architecture:
        raise ValueError(
            "TRITON_OVERRIDE_ARCH conflicts with the THead target capability"
        )
    os.environ["TRITON_OVERRIDE_ARCH"] = expected_architecture

    sdk_root = _selected_sdk_root()
    expected_directories = {
        "FLAGDNN_THEAD_PPU_SDK_ROOT": sdk_root,
        "PPU_SDK": sdk_root,
        "PPU_HOME": sdk_root,
        "CUDA_PATH": sdk_root / "CUDA_SDK",
    }
    for name, expected in expected_directories.items():
        configured = os.environ.get(name, "")
        if configured:
            actual = _required_directory(Path(configured), name)
            if actual != expected:
                raise ValueError(
                    f"{name} conflicts with the selected THead PPU SDK"
                )
        os.environ[name] = str(expected)

    expected_tools = {
        "TRITON_PTXAS_PATH": sdk_root / "CUDA_SDK/bin/ptxas",
        "TRITON_IR_FORMATTER_PATH": sdk_root / "bin/llvm-irformatter",
    }
    for name, expected in expected_tools.items():
        configured = os.environ.get(name, "")
        if configured:
            actual = _required_file(
                Path(configured), name, executable=True
            )
            if actual != expected.resolve(strict=True):
                raise ValueError(
                    f"{name} conflicts with the selected THead PPU SDK"
                )
        os.environ[name] = str(expected.resolve(strict=True))

    imported_triton_root = (
        Path(__import__("triton").__file__).resolve(strict=True).parent.parent
    )
    configured_triton_root = os.environ.get(
        "FLAGDNN_THEAD_TRITON_ROOT", ""
    )
    triton_root = _required_directory(
        Path(configured_triton_root)
        if configured_triton_root
        else imported_triton_root,
        "THead Triton root",
    )
    try:
        imported_triton_root.relative_to(triton_root)
    except ValueError as error:
        raise ValueError(
            "FLAGDNN_THEAD_TRITON_ROOT does not contain imported Triton"
        ) from error
    os.environ["FLAGDNN_THEAD_TRITON_ROOT"] = str(triton_root)


def _selected_sdk_root() -> Path:
    configured = os.environ.get("FLAGDNN_THEAD_PPU_SDK_ROOT", "")
    root = _required_directory(
        Path(configured or os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")),
        "PPU SDK root",
    )
    for name in ("PPU_SDK", "PPU_HOME", "PPU_PATH"):
        value = os.environ.get(name, "")
        if value and _required_directory(Path(value), name) != root:
            raise RuntimeError(f"{name} does not match selected PPU SDK root")
    cuda_root = root / "CUDA_SDK"
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path and _required_directory(Path(cuda_path), "CUDA_PATH") != cuda_root:
        raise RuntimeError("CUDA_PATH does not match the selected PPU CUDA SDK")
    return root


def _selected_tool(
    environment_name: str, fallback: Path, description: str
) -> Path:
    configured = os.environ.get(environment_name, "")
    return _required_file(
        Path(configured) if configured else fallback,
        description,
        executable=True,
    )


def _sdk_paths() -> dict[str, Path]:
    root = _selected_sdk_root()
    paths = {
        "release": _required_file(
            root / "release.yaml", "PPU SDK release", root=root
        ),
        "ptxas": _selected_tool(
            "TRITON_PTXAS_PATH",
            root / "CUDA_SDK/bin/ptxas",
            "PPU CUDA compatibility assembler",
        ),
        "ir_formatter": _selected_tool(
            "TRITON_IR_FORMATTER_PATH",
            root / "bin/llvm-irformatter",
            "PPU LLVM IR formatter",
        ),
        "compatibility_compiler": _required_file(
            root / "CUDA_SDK/bin/nvcc",
            "PPU CUDA compatibility compiler",
            root=root,
            executable=True,
        ),
        "cuda_header": _required_file(
            root / "CUDA_SDK/include/cuda.h",
            "PPU CUDA compatibility header",
            root=root,
        ),
        "cuda_driver": _required_file(
            root / "CUDA_SDK/lib64/libcuda.so.1",
            "PPU CUDA compatibility driver",
            root=root,
        ),
        "hggc_runtime": _required_file(
            root / "lib/libhggc.so", "PPU HGGC runtime", root=root
        ),
        "cuda_libdevice": _required_file(
            root / "CUDA_SDK/nvvm/libdevice/libdevice.10.bc",
            "PPU CUDA compatibility libdevice",
            root=root,
        ),
    }
    return paths


def _selected_jit_root() -> Path:
    configured = os.environ.get("FLAGDNN_THEAD_TRITON_JIT_ROOT", "")
    if not configured:
        installed = _installed_jit_layout()
        if installed is not None:
            return installed[0]
    provider_directory = _provider_directory()
    if (
        len(provider_directory.parents) < 3
        or provider_directory.name != "thead"
        or provider_directory.parent.name != "backends"
    ):
        raise RuntimeError(
            "cannot derive the source-tree libtriton_jit root from the "
            f"THead provider location: {provider_directory}"
        )
    source_tree_default = provider_directory.parents[2] / "libtriton_jit"
    return _required_directory(
        Path(configured) if configured else source_tree_default,
        "THead CUDA-backend libtriton_jit root",
    )


def _uses_installed_jit_layout() -> bool:
    return (
        not os.environ.get("FLAGDNN_THEAD_TRITON_JIT_ROOT", "")
        and (_provider_directory() / _COMPILER_ENVIRONMENT_FILE).is_file()
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key in compiler environment: {name}")
        result[name] = value
    return result


def _safe_install_relative_path(
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


def _installed_jit_layout() -> tuple[Path, dict[str, Path]] | None:
    environment_path = _provider_directory() / _COMPILER_ENVIRONMENT_FILE
    if not environment_path.is_file():
        return None
    if environment_path.stat().st_size > 64 * 1024:
        raise ValueError(
            f"{environment_path}: compiler environment is too large"
        )
    try:
        value = json.loads(
            environment_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read THead compiler environment {environment_path}: {error}"
        ) from error
    required_fields = {
        "schema_version",
        "backend",
        "libtriton_jit_soname",
        "libtriton_jit_install_relative_path",
        "jit_script_install_relative_path",
        "libtriton_jit_sha256",
        "triton_jit_provenance_sha256",
        "standalone_compile_sha256",
        "gen_ssig_sha256",
    }
    if not isinstance(value, dict) or set(value) != required_fields:
        actual_fields = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{environment_path}: compiler environment fields do not match; "
            f"missing={sorted(required_fields - actual_fields)}; "
            f"extra={sorted(actual_fields - required_fields)}"
        )
    if value["schema_version"] != 1 or value["backend"] != "CUDA":
        raise ValueError(
            f"{environment_path}: unsupported THead compiler environment"
        )
    soname = value["libtriton_jit_soname"]
    if not isinstance(soname, str) or _JIT_SONAME.fullmatch(soname) is None:
        raise ValueError(f"{environment_path}: invalid libtriton_jit SONAME")
    for name in (
        "libtriton_jit_sha256",
        "triton_jit_provenance_sha256",
        "standalone_compile_sha256",
        "gen_ssig_sha256",
    ):
        digest = value[name]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{environment_path}: {name} is not SHA-256")

    library_relative = _safe_install_relative_path(
        value["libtriton_jit_install_relative_path"],
        source=environment_path,
        name="libtriton_jit_install_relative_path",
        suffix=("flagdnn", "thead", soname),
    )
    scripts_relative = _safe_install_relative_path(
        value["jit_script_install_relative_path"],
        source=environment_path,
        name="jit_script_install_relative_path",
        suffix=("flagdnn", "share", "triton_jit", "scripts"),
    )
    if library_relative.parts[:-3] != scripts_relative.parts[:-4]:
        raise ValueError(
            f"{environment_path}: private JIT resources use different libdirs"
        )

    provider_directory = _provider_directory()
    if len(provider_directory.parents) < 4:
        raise ValueError(
            f"{environment_path}: provider is not in an installed SDK layout"
        )
    sdk_root = provider_directory.parents[3].resolve()
    expected_provider = (
        sdk_root / "share/flagdnn/backends/thead"
    ).resolve()
    if provider_directory.resolve() != expected_provider:
        raise ValueError(
            f"{environment_path}: provider is not in the canonical SDK layout"
        )
    library = _required_file(
        sdk_root / library_relative,
        "installed private libtriton_jit",
        root=sdk_root,
    )
    scripts = _required_directory(
        sdk_root / scripts_relative, "installed private JIT script directory"
    )
    standalone = _required_file(
        scripts / "standalone_compile.py",
        "installed private standalone compiler",
        root=sdk_root,
    )
    gen_ssig = _required_file(
        scripts / "gen_ssig.py",
        "installed private signature generator",
        root=sdk_root,
    )
    expected_hashes = {
        library: value["libtriton_jit_sha256"],
        standalone: value["standalone_compile_sha256"],
        gen_ssig: value["gen_ssig_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise RuntimeError(
                f"installed THead JIT resource hash mismatch: {path}"
            )
    return sdk_root, {
        "environment": environment_path.resolve(),
        "library": library,
        "standalone_compile": standalone,
        "gen_ssig": gen_ssig,
    }


def _jit_paths() -> dict[str, Path]:
    if not os.environ.get("FLAGDNN_THEAD_TRITON_JIT_ROOT", ""):
        installed = _installed_jit_layout()
        if installed is not None:
            return installed[1]
    root = _selected_jit_root()
    layouts = (
        (
            root / "build/TritonJITConfig.cmake",
            root / "build/src/libtriton_jit.so",
            root / "include",
            root / "scripts",
        ),
        (
            root / "lib/cmake/TritonJIT/TritonJITConfig.cmake",
            root / "lib/libtriton_jit.so",
            root / "include",
            root / "share/triton_jit/scripts",
        ),
        (
            root / "lib64/cmake/TritonJIT/TritonJITConfig.cmake",
            root / "lib64/libtriton_jit.so",
            root / "include",
            root / "share/triton_jit/scripts",
        ),
    )
    selected: tuple[Path, Path, Path, Path] | None = None
    for layout in layouts:
        config, library, include, scripts = layout
        if (
            config.is_file()
            and library.is_file()
            and include.is_dir()
            and scripts.is_dir()
            and all((include / header).is_file() for header in _JIT_HEADERS)
            and (scripts / "standalone_compile.py").is_file()
            and (scripts / "gen_ssig.py").is_file()
        ):
            selected = layout
            break
    if selected is None:
        raise RuntimeError(
            "selected libtriton_jit root has no coherent build/install layout"
        )
    config, library, include, scripts = selected
    paths: dict[str, Path] = {
        "config": _required_file(config, "TritonJITConfig.cmake", root=root),
        "library": _required_file(
            library, "libtriton_jit shared library", root=root
        ),
        "standalone_compile": _required_file(
            scripts / "standalone_compile.py",
            "libtriton_jit standalone compiler",
            root=root,
        ),
        "gen_ssig": _required_file(
            scripts / "gen_ssig.py", "libtriton_jit signature generator", root=root
        ),
    }
    for header in _JIT_HEADERS:
        paths[f"header::{header}"] = _required_file(
            include / header, f"libtriton_jit header {header}", root=root
        )
    for path in sorted(include.rglob("*.h")):
        paths[f"header::{path.relative_to(include).as_posix()}"] = (
            _required_file(path, "libtriton_jit public header", root=root)
        )
    for path in sorted(config.parent.glob("TritonJITTargets*.cmake")):
        paths[f"cmake::{path.name}"] = _required_file(
            path, "libtriton_jit exported target metadata", root=root
        )
    for path in sorted(scripts.rglob("*.py")):
        if "__pycache__" not in path.parts:
            paths[f"script::{path.relative_to(scripts).as_posix()}"] = (
                _required_file(path, "libtriton_jit script", root=root)
            )
    text = paths["config"].read_text(encoding="utf-8")
    backends = re.findall(
        r"^[ \t]*set\([ \t]*TritonJIT_BACKEND[ \t]+[\"']?"
        r"([A-Za-z0-9_+-]+)[\"']?[ \t]*\)[ \t]*$",
        text,
        flags=re.MULTILINE,
    )
    if backends != ["CUDA"]:
        raise RuntimeError(
            "THead requires libtriton_jit configured with backend CUDA"
        )
    return paths


def _kernel_paths() -> tuple[Path, ...]:
    root = _resource_root()
    resource_provider = root / "backends/thead"
    provider = (
        resource_provider
        if (resource_provider / "kernels/registry.json").is_file()
        and (resource_provider / "tuning/common.yaml").is_file()
        else _provider_directory()
    )
    paths: set[Path] = {
        _required_file(
            root / "kernels/registry.json", "common kernel registry", root=root
        ),
        _required_file(
            Path(kernel_registry.__file__), "kernel registry implementation"
        ),
    }
    common = root / "kernels/common"
    for path in sorted(common.glob("*.py")):
        paths.add(_required_file(path, "common Triton kernel", root=root))
    for relative in ("kernels", "tuning"):
        directory = provider / relative
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix not in {".pyc", ".pyo"}
                ):
                    paths.add(
                        _required_file(
                            path, f"THead {relative} dependency", root=provider
                        )
                    )
    return tuple(sorted(paths))


def _provider_paths() -> tuple[Path, ...]:
    directory = _provider_directory()
    paths = tuple(
        sorted(
            _required_file(path, "THead compiler module", root=directory)
            for path in directory.glob("*.py")
        )
    )
    if not paths:
        raise RuntimeError("THead provider contains no Python modules")
    return paths


def _git_metadata_paths(root: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--git-dir"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        return ()
    git_directory = Path(completed.stdout.strip())
    if not git_directory.is_absolute():
        git_directory = root / git_directory
    git_directory = git_directory.resolve()
    candidates = [
        git_directory / "HEAD",
        git_directory / "index",
        git_directory / "packed-refs",
    ]
    head = git_directory / "HEAD"
    if head.is_file():
        text = head.read_text(encoding="utf-8", errors="replace").strip()
        if text.startswith("ref: "):
            candidates.append(git_directory / text.removeprefix("ref: "))
    return tuple(sorted(path.resolve() for path in candidates if path.is_file()))


def compiler_identity_dependency_paths(
    target: str, execution_engine: str
) -> tuple[Path, ...]:
    _validate_target(target, execution_engine)
    sdk = _sdk_paths()
    jit = _jit_paths()
    environment_paths = module_dependency_paths()
    triton_root = Path(
        os.environ.get("FLAGDNN_THEAD_TRITON_ROOT", "")
        or Path(__import__("triton").__file__).resolve().parent.parent
    ).resolve()
    jit_git_metadata = (
        ()
        if _uses_installed_jit_layout()
        else _git_metadata_paths(_selected_jit_root())
    )
    paths: set[Path] = {
        *_provider_paths(),
        *_kernel_paths(),
        *sdk.values(),
        *jit.values(),
        *environment_paths,
        *_git_metadata_paths(triton_root),
        *jit_git_metadata,
    }
    result = tuple(sorted(path.resolve() for path in paths))
    for path in result:
        if not path.is_file():
            raise RuntimeError(f"compiler identity dependency is missing: {path}")
    return result


def _tool_version(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(path), "--version"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    output = completed.stdout.strip()
    if not output:
        raise RuntimeError(f"tool returned no version identity: {path}")
    return {"returncode": completed.returncode, "output": output}


def _driver_version(path: Path) -> dict[str, Any]:
    try:
        library = ctypes.CDLL(str(path))
        function = library.cuDriverGetVersion
        function.argtypes = [ctypes.POINTER(ctypes.c_int)]
        function.restype = ctypes.c_int
        value = ctypes.c_int()
        result = int(function(ctypes.byref(value)))
        if result == 0 and value.value > 0:
            return {"status": "available", "version": int(value.value)}
        return {"status": "unavailable", "result": result}
    except (AttributeError, OSError) as error:
        return {"status": "unavailable", "detail": str(error)}


def _repository_identity(root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if head.returncode != 0 or status.returncode != 0:
        return {"kind": "content_only"}
    return {
        "kind": "git",
        "head": head.stdout.strip(),
        "tracked_status_sha256": hashlib.sha256(
            status.stdout.encode("utf-8")
        ).hexdigest(),
    }


def _dependency_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    result = []
    for index, path in enumerate(paths):
        resolved = _required_file(path, "compiler identity dependency")
        result.append(
            {
                "index": index,
                "path": str(resolved),
                "size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return result


def _sdk_version(release: Path) -> str:
    matches = re.findall(
        r"^[ \t]*version[ \t]*:[ \t]*[\"']?([^\"'#\r\n]+)",
        release.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise RuntimeError("PPU SDK release has no unique version")
    return matches[0].strip()


def build_compiler_identity(
    target: str, execution_engine: str
) -> dict[str, Any]:
    capability = _validate_target(target, execution_engine)
    sdk = _sdk_paths()
    jit = _jit_paths()
    dependencies = compiler_identity_dependency_paths(target, execution_engine)
    triton_root = Path(
        os.environ.get("FLAGDNN_THEAD_TRITON_ROOT", "")
        or Path(__import__("triton").__file__).resolve().parent.parent
    ).resolve()
    jit_root = _selected_jit_root()
    jit_repository = (
        {"kind": "content_only"}
        if _uses_installed_jit_layout()
        else _repository_identity(jit_root)
    )
    document = {
        "schema_version": 1,
        "provider": PROVIDER_NAME,
        "provider_version": PROVIDER_VERSION,
        "backend": "thead",
        "target": target,
        "compatibility_capability": capability,
        "execution_engine": execution_engine,
        "graph_ir_schema_version": GRAPH_IR_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "execution_program_version": EXECUTION_PROGRAM_VERSION,
        "ppu_sdk": {
            "root": str(_selected_sdk_root()),
            "version": _sdk_version(sdk["release"]),
            "driver_version": _driver_version(sdk["cuda_driver"]),
            "ptxas": _tool_version(sdk["ptxas"]),
            "ir_formatter": _tool_version(sdk["ir_formatter"]),
        },
        "triton_repository": _repository_identity(triton_root),
        "libtriton_jit": {
            "root": str(jit_root),
            "backend": "CUDA",
            "repository": jit_repository,
        },
        "python_environment": collect_environment_identity(),
        "dependencies": _dependency_records(dependencies),
    }
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "provider": PROVIDER_NAME,
        "provider_version": PROVIDER_VERSION,
        "identity_sha256": hashlib.sha256(canonical).hexdigest(),
    }
