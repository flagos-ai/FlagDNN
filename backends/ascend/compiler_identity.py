"""Ascend compiler identity and code-generation environment policy."""

from __future__ import annotations

import hashlib
import importlib.util
from importlib import metadata
import json
import os
import platform
from pathlib import Path
import re
import shutil
import sys
from typing import Any

from flagdnn_codegen.kernel_registry import (
    iter_kernel_registry_sources,
    resolve_kernel_source,
    resolve_tuning_source,
    select_kernel_candidate,
)
from .tuning_decoder import canonical_json_bytes


_SAFE_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_VERSIONED_TARGET = re.compile(
    r"^ascend_(?P<arch>Ascend[A-Za-z0-9_-]*)_cann_"
    r"(?P<cann>[0-9][A-Za-z0-9_.-]*)_aic_(?P<aic>[1-9][0-9]{0,4})$"
)
_CODEGEN_ARCH = re.compile(r"^Ascend[A-Za-z0-9_-]*$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]+$")
MAXIMUM_AI_CORE_COUNT = 65535
_FALSE_VALUES = {"", "0", "false", "off", "no"}


def _load_target_policy() -> tuple[
    tuple[str, ...], tuple[str, ...], dict[str, str]
]:
    path = Path(__file__).with_name("capabilities.json")
    try:
        encoded = path.read_bytes()
        if not encoded or len(encoded) > (1 << 20):
            raise ValueError("capability resource has an invalid size")
        document = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot load the Ascend target policy") from error
    if not isinstance(document, dict):
        raise ValueError("Ascend capabilities must be a JSON object")
    arches = document.get("codegen_arches")
    policy = document.get("target_policy")
    if (
        not isinstance(arches, list)
        or not arches
        or any(
            not isinstance(arch, str)
            or _CODEGEN_ARCH.fullmatch(arch) is None
            for arch in arches
        )
        or len(set(arches)) != len(arches)
        or not isinstance(policy, dict)
        or set(policy)
        != {"runtime_detected_codegen_arches", "compatibility_alias"}
    ):
        raise ValueError("Ascend target policy fields are invalid")
    runtime_detected = policy.get("runtime_detected_codegen_arches")
    compatibility = policy.get("compatibility_alias")
    if (
        not isinstance(runtime_detected, list)
        or not runtime_detected
        or any(arch not in arches for arch in runtime_detected)
        or len(set(runtime_detected)) != len(runtime_detected)
        or not isinstance(compatibility, dict)
        or set(compatibility)
        != {"environment", "runtime_soc", "codegen_arch"}
        or not isinstance(compatibility.get("environment"), str)
        or _ENVIRONMENT_NAME.fullmatch(compatibility["environment"]) is None
        or compatibility.get("runtime_soc") not in arches
        or compatibility.get("codegen_arch") not in arches
        or compatibility["runtime_soc"] not in runtime_detected
        or compatibility["runtime_soc"] == compatibility["codegen_arch"]
    ):
        raise ValueError("Ascend target compatibility policy is invalid")
    return tuple(arches), tuple(runtime_detected), dict(compatibility)


(
    SUPPORTED_CODEGEN_ARCHES,
    RUNTIME_DETECTED_CODEGEN_ARCHES,
    COMPATIBILITY_ALIAS,
) = _load_target_policy()
SUPPORTED_TARGET_SOCS = SUPPORTED_CODEGEN_ARCHES
_COMPATIBILITY_ENVIRONMENT = (
    COMPATIBILITY_ALIAS["environment"]
)

_CODEGEN_ENVIRONMENT_KEYS = (
    "ASCEND_HOME_PATH",
    "TRITON_NPU_COMPILER_PATH",
    "MLIR_ROOT",
    "LLVM_ROOT",
    "CC",
    "TRITON_ENABLE_VF_FUSION",
    "TRITON_DISABLE_FFTS",
    "TRITON_ENABLE_LIBDEVICE_SIMT",
    "TRITON_ALL_BLOCKS_PARALLEL",
    "TRITON_DISABLE_LINE_INFO",
    "TRITON_ENABLE_SANITIZER",
    "ENABLE_UNPUBLISHED_FEATURE",
    "ENABLE_PRINT_UB_BITS",
    "TRITON_MEMORY_DISPLAY",
    "LLVM_EXTRACT_DI_LOCAL_VARIABLES",
    "TRITON_ENABLE_TASKQUEUE",
    "TRITON_DEVICE_PRINT",
    "TRITON_GRID_WARN_PRINT",
    "TRITON_DISABLE_PRECOMPILE",
    "TRITON_ALLOW_NON_CONSTEXPR_GLOBALS",
    "PYTHONHOME",
    "PYTHONPATH",
    "PATH",
    "LD_LIBRARY_PATH",
)

_FORBIDDEN_CACHE_KEYS = (
    "TRITON_COMPILE_ONLY",
    "TRITON_ALWAYS_COMPILE",
    "TRITON_KERNEL_OVERRIDE",
    "TRITON_STORE_BINARY_ONLY",
    "TRITON_CACHE_MANAGER",
    "TRITON_REMOTE_CACHE_BACKEND",
    "TRITON_OVERRIDE_DIR",
)

_KNOWN_TRITON_ENVIRONMENT_KEYS = set(_CODEGEN_ENVIRONMENT_KEYS).union(
    _FORBIDDEN_CACHE_KEYS,
    {
        "TRITON_JIT_BACKEND",
        "TRITON_ASCEND_ARCH",
        "TRITON_BACKEND",
        "TRITON_CACHE_DIR",
    },
)


def _canonical_ascend_home() -> Path | None:
    configured = [
        value
        for value in (
            os.environ.get("ASCEND_HOME_PATH", ""),
            os.environ.get("ASCEND_TOOLKIT_HOME", ""),
        )
        if value
    ]
    if not configured:
        return None
    resolved = [
        Path(value).expanduser().resolve(strict=True)
        for value in configured
    ]
    if any(not value.is_dir() for value in resolved):
        raise ValueError("configured CANN root is not a directory")
    if any(value != resolved[0] for value in resolved[1:]):
        raise ValueError(
            "ASCEND_HOME_PATH and ASCEND_TOOLKIT_HOME identify different "
            "CANN installations"
        )
    return resolved[0]


def validate_target_name(target_name: str) -> str:
    if (
        not isinstance(target_name, str)
        or _SAFE_TARGET.fullmatch(target_name) is None
        or _VERSIONED_TARGET.fullmatch(target_name) is None
    ):
        raise ValueError("Ascend target fingerprint is invalid")
    match = _VERSIONED_TARGET.fullmatch(target_name)
    assert match is not None
    if match.group("arch") not in SUPPORTED_TARGET_SOCS:
        raise ValueError("Ascend target SoC is not supported")
    if (
        "_aic_" in match.group("cann")
        or int(match.group("aic")) > MAXIMUM_AI_CORE_COUNT
    ):
        raise ValueError("Ascend target fingerprint is invalid")
    return target_name


def codegen_arch_from_target(target_name: str) -> str:
    validate_target_name(target_name)
    match = _VERSIONED_TARGET.fullmatch(target_name)
    assert match is not None
    target_soc = match.group("arch")
    compatibility = os.environ.get(
        _COMPATIBILITY_ENVIRONMENT
    )
    if compatibility not in {None, "", "0", "1"}:
        raise ValueError(
            f"{_COMPATIBILITY_ENVIRONMENT} must be unset, '0', or '1'"
        )
    if compatibility == "1":
        if target_soc != COMPATIBILITY_ALIAS["runtime_soc"]:
            raise ValueError(
                f"{_COMPATIBILITY_ENVIRONMENT}=1 is valid "
                f"only for {COMPATIBILITY_ALIAS['runtime_soc']}"
            )
        return COMPATIBILITY_ALIAS["codegen_arch"]
    return target_soc


def aicore_count_from_target(target_name: str) -> int:
    validate_target_name(target_name)
    match = _VERSIONED_TARGET.fullmatch(target_name)
    assert match is not None
    return int(match.group("aic"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(path: Path, resource_root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"compiler identity input is not a file: {path}")
    try:
        display_path = resolved.relative_to(resource_root).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "size": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _external_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"configured identity input is not a file: {path}")
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _effective_environment(target_name: str) -> dict[str, str]:
    backend = os.environ.get("TRITON_JIT_BACKEND", "NPU")
    if backend.upper() != "NPU":
        raise ValueError("TRITON_JIT_BACKEND must select NPU")
    triton_backend = os.environ.get("TRITON_BACKEND", "torch_npu")
    if triton_backend != "torch_npu":
        raise ValueError("TRITON_BACKEND must be torch_npu")
    autoload = os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    if autoload != "0":
        raise ValueError("TORCH_DEVICE_BACKEND_AUTOLOAD must be 0")
    all_blocks_parallel = os.environ.get("TRITON_ALL_BLOCKS_PARALLEL")
    if all_blocks_parallel not in {None, "false"}:
        raise ValueError("TRITON_ALL_BLOCKS_PARALLEL must be false")

    expected_arch = codegen_arch_from_target(target_name)
    match = _VERSIONED_TARGET.fullmatch(target_name)
    assert match is not None
    target_soc = match.group("arch")
    runtime_detected_arch = (
        target_soc in RUNTIME_DETECTED_CODEGEN_ARCHES
        and expected_arch == target_soc
    )
    configured_arch = os.environ.get("TRITON_ASCEND_ARCH")
    if runtime_detected_arch:
        if configured_arch is not None:
            raise ValueError(
                "TRITON_ASCEND_ARCH must be unset for exact "
                f"{target_soc} runtime detection"
            )
    elif configured_arch not in {None, expected_arch}:
        raise ValueError(
            "TRITON_ASCEND_ARCH differs from the target SoC codegen arch"
        )

    result = {
        "TRITON_JIT_BACKEND": "NPU",
        "TRITON_ASCEND_ARCH": (
            f"<runtime-detected:{expected_arch}>"
            if runtime_detected_arch
            else expected_arch
        ),
        _COMPATIBILITY_ENVIRONMENT: (
            "1" if expected_arch != target_soc else "<unset>"
        ),
        "TRITON_BACKEND": "torch_npu",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONSAFEPATH": "1",
        "PYTHONPYCACHEPREFIX": "<unset>",
        "PYTHONUSERBASE": "<unset>",
        "TRITON_CACHE_DIR": "<runtime-owned>",
        "TMPDIR": "<request-owned>",
    }
    ascend_home = _canonical_ascend_home()
    for name in _CODEGEN_ENVIRONMENT_KEYS:
        if name == "ASCEND_HOME_PATH" and ascend_home is not None:
            result[name] = str(ascend_home)
        elif name == "TRITON_ALL_BLOCKS_PARALLEL":
            result[name] = "false"
        else:
            result[name] = os.environ.get(name, "<unset>")
    for name in _FORBIDDEN_CACHE_KEYS:
        raw_value = os.environ.get(name, "")
        if raw_value.strip().lower() not in _FALSE_VALUES:
            raise ValueError(f"{name} is forbidden for the Ascend compiler")
        result[name] = "<unset>" if name not in os.environ else "false"

    unclassified = sorted(
        name
        for name, value in os.environ.items()
        if name.startswith("TRITON_")
        and name not in _KNOWN_TRITON_ENVIRONMENT_KEYS
        and value.strip().lower() not in _FALSE_VALUES
    )
    if unclassified:
        raise ValueError(
            "unclassified non-default Triton environment keys: "
            + ", ".join(unclassified)
        )
    return result


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for output_name, distribution_names in (
        ("torch", ("torch",)),
        ("triton", ("triton",)),
        ("pyyaml", ("PyYAML", "pyyaml")),
        ("torch_npu", ("torch-npu", "torch_npu")),
    ):
        version = "<unavailable>"
        for distribution_name in distribution_names:
            try:
                version = metadata.version(distribution_name)
                break
            except metadata.PackageNotFoundError:
                continue
        result[output_name] = version
    return result


def _package_source_files() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    required = os.environ.get(
        "FLAGDNN_ASCEND_REQUIRE_COMPILER_MODULES", "0"
    ) == "1"
    roots: dict[str, Path] = {}
    for package in ("torch", "triton", "yaml", "torch_npu"):
        specification = importlib.util.find_spec(package)
        if specification is None or specification.origin is None:
            if required:
                raise ValueError(
                    f"required Ascend compiler module is missing: {package}"
                )
            continue
        origin = Path(specification.origin).resolve(strict=True)
        roots[package] = origin.parent
        result[f"module:{package}"] = _external_file_identity(origin)

    triton_root = roots.get("triton")
    if triton_root is not None:
        candidates = {
            triton_root / "runtime/cache.py",
            triton_root / "runtime/driver.py",
            triton_root / "runtime/jit.py",
            triton_root / "compiler/compiler.py",
            triton_root / "spec/ascend/compiler/code_generator.py",
            triton_root / "spec/ascend/compiler/compiler.py",
            triton_root / "spec/ascend/runtime/driver.py",
            triton_root / "_C/libtriton.so",
        }
        ascend_backend = triton_root / "backends/ascend"
        if ascend_backend.is_dir():
            candidates.update(ascend_backend.rglob("*.py"))
        for path in sorted(candidates):
            if path.is_file():
                relative = path.relative_to(triton_root).as_posix()
                result[f"module:triton/{relative}"] = (
                    _external_file_identity(path)
                )
        if required and not (ascend_backend / "driver.py").is_file():
            raise ValueError("pinned Triton has no Ascend compiler backend")

    torch_npu_root = roots.get("torch_npu")
    if torch_npu_root is not None:
        for relative in ("version.py", "npu/utils.py"):
            path = torch_npu_root / relative
            if path.is_file():
                result[f"module:torch_npu/{relative}"] = (
                    _external_file_identity(path)
                )
    return result


def _configured_external_files() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    ascend_home = _canonical_ascend_home()
    if ascend_home is not None:
        architecture = platform.machine()
        architecture_names = [architecture]
        if architecture in {"aarch64", "arm64"}:
            architecture_names.extend(["aarch64-linux", "arm64-linux"])
        elif architecture in {"x86_64", "amd64"}:
            architecture_names.extend(["x86_64-linux"])
        candidates: dict[str, list[Path]] = {
            "cann_package_info": [
                ascend_home / name / "ascend_toolkit_install.info"
                for name in architecture_names
            ],
            "libascendcl": [
                ascend_home / name / "lib64" / "libascendcl.so"
                for name in architecture_names
            ],
            "libruntime": [
                ascend_home / name / "lib64" / "libruntime.so"
                for name in architecture_names
            ],
        }
        candidates["cann_package_info"].append(
            ascend_home / "ascend_toolkit_install.info"
        )
        candidates["libascendcl"].append(ascend_home / "lib64/libascendcl.so")
        candidates["libruntime"].append(ascend_home / "lib64/libruntime.so")
        for label, paths in candidates.items():
            for path in paths:
                if path.is_file():
                    result[label] = _external_file_identity(path)
                    break

    explicitly_configured = {
        "libtriton_jit": "FLAGDNN_LIBTRITON_JIT_LIBRARY",
        "standalone_compile": "FLAGDNN_TRITON_JIT_STANDALONE_COMPILER",
        "triton_jit_config": "FLAGDNN_TRITON_JIT_CONFIG",
    }
    configured_values = {
        label: os.environ.get(environment_name, "")
        for label, environment_name in explicitly_configured.items()
    }
    include_directory_value = os.environ.get(
        "FLAGDNN_TRITON_JIT_INCLUDE_DIRECTORY", ""
    )
    configured_count = sum(bool(value) for value in configured_values.values())
    configured_count += bool(include_directory_value)
    if configured_count not in {0, len(configured_values) + 1}:
        raise ValueError(
            "NPU TritonJIT identity paths must be configured as one set"
        )
    for label, value in configured_values.items():
        if value:
            result[label] = _external_file_identity(Path(value))
    if include_directory_value:
        include_directory = (
            Path(include_directory_value).expanduser().resolve(strict=True)
        )
        if not include_directory.is_dir():
            raise ValueError("NPU TritonJIT public include root is invalid")
        public_headers = sorted(
            path
            for path in (include_directory / "triton_jit").rglob("*")
            if path.is_file() and path.suffix in {".h", ".hpp"}
        )
        if not public_headers or not (
            include_directory / "triton_jit/kernel_metadata.h"
        ).is_file():
            raise ValueError("NPU TritonJIT public headers are incomplete")
        for path in public_headers:
            relative = path.relative_to(include_directory).as_posix()
            result[f"libtriton_jit_header:{relative}"] = (
                _external_file_identity(path)
            )

    compiler_value = os.environ.get("CC", "")
    if compiler_value and os.path.sep not in compiler_value:
        located = shutil.which(compiler_value)
        if located:
            result["cc"] = _external_file_identity(Path(located))
    elif compiler_value and Path(compiler_value).is_file():
        result["cc"] = _external_file_identity(Path(compiler_value))
    return result


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
    launch_abi: str,
) -> dict[str, Any]:
    """Hash deterministic inputs that can change an Ascend artifact."""

    validate_target_name(target_name)
    if execution_engine != "libtriton_jit":
        raise ValueError("Ascend supports only the libtriton_jit engine")
    provider_path = provider_path.resolve(strict=True)
    compiler_entry = compiler_entry.resolve(strict=True)
    resource_root = compiler_entry.parents[2]

    identity_inputs: dict[str, Path] = {
        "provider:ascend": provider_path,
        "provider_identity:ascend": Path(__file__),
        "compiler_driver": compiler_entry,
        "provider_loader": compiler_entry.with_name("provider_loader.py"),
        "kernel_registry_decoder": compiler_entry.with_name(
            "kernel_registry.py"
        ),
        "provider_add_plan:ascend": provider_path.with_name("add_plan.py"),
        "provider_tuning:ascend": provider_path.with_name(
            "tuning_decoder.py"
        ),
        "capabilities": provider_path.with_name("capabilities.json"),
    }
    for package in ("codegen", "dispatch", "kernels"):
        for source in sorted((provider_path.parent / package).rglob("*.py")):
            label = source.relative_to(provider_path.parent).as_posix()
            identity_inputs[f"provider_module:{label}"] = source
    for registry_path in iter_kernel_registry_sources("ascend"):
        label = registry_path.resolve().relative_to(resource_root).as_posix()
        identity_inputs[f"registry:{label}"] = registry_path
    from .add_plan import SUPPORTED_POINTWISE_OPERATIONS

    for operation in SUPPORTED_POINTWISE_OPERATIONS:
        candidate = select_kernel_candidate("ascend", operation)
        identity_inputs[f"kernel:{operation}"] = resolve_kernel_source(
            compiler_entry, candidate
        )
        identity_inputs[f"tuning:{operation}"] = resolve_tuning_source(
            compiler_entry, candidate
        )
    source_files = {
        label: _source_identity(path, resource_root)
        for label, path in sorted(identity_inputs.items())
    }

    codegen_environment = _effective_environment(target_name)
    environment_payload = {
        "schema_version": 1,
        "target": target_name,
        "values": codegen_environment,
    }
    environment_digest = hashlib.sha256(
        canonical_json_bytes(environment_payload, "Ascend compiler environment")
    ).hexdigest()

    payload: dict[str, Any] = {
        "provider": provider_name,
        "provider_version": provider_version,
        "graph_schema_version": graph_schema_version,
        "artifact_schema_version": artifact_schema_version,
        "execution_program_version": execution_program_version,
        "launch_abi": launch_abi,
        "execution_engine": execution_engine,
        "target": target_name,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "python_executable": str(Path(sys.executable).resolve()),
        "package_versions": _package_versions(),
        "package_source_files": _package_source_files(),
        "codegen_environment_schema_version": 1,
        "codegen_environment_digest": environment_digest,
        "codegen_environment": codegen_environment,
        "cache_runtime_policy": {
            "schema_version": 1,
            "cache_root": "<runtime-owned>",
            "tmp_root": "<request-owned>",
        },
        "external_files": _configured_external_files(),
        "source_files": source_files,
    }
    payload["identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload, "Ascend compiler identity")
    ).hexdigest()
    return payload
