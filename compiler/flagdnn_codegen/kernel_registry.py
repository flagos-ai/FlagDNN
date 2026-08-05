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

"""Select compiler-safe kernel sources without importing FlagDNN or Torch."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class KernelTuningSpec:
    source: str
    table: str
    key: str
    strategy: str
    warmup: int
    repetitions: int


@dataclass(frozen=True)
class KernelCandidate:
    backend: str
    operation: str
    provider: str
    source: str
    functions: tuple[str, ...]
    source_layout: str = "kernels"
    source_format: str = "module"
    ownership: str = "common"
    tuning: KernelTuningSpec | None = None

    @property
    def function(self) -> str:
        """Return the default entry point for single-function callers."""

        return self.functions[0]


UNARY_POINTWISE_OPERATIONS = (
    "abs",
    "ceil",
    "cos",
    "erf",
    "exp",
    "floor",
    "identity",
    "log",
    "neg",
    "reciprocal",
    "rsqrt",
    "sin",
    "sqrt",
    "tan",
    "logical_not",
    "sigmoid",
    "tanh",
    "elu",
    "gelu",
    "softplus",
    "swish",
    "gelu_approx_tanh",
)

BINARY_POINTWISE_OPERATIONS = (
    "sub",
    "sigmoid_backward",
    "mul",
    "div",
    "min",
    "max",
    "mod",
    "pow",
    "cmp_eq",
    "cmp_neq",
    "cmp_gt",
    "cmp_ge",
    "cmp_lt",
    "cmp_le",
    "logical_and",
    "logical_or",
)


TERNARY_POINTWISE_OPERATIONS = ("binary_select",)

_COMMON_CANDIDATES: dict[str, KernelCandidate] = {}
_PLATFORM_CANDIDATES: dict[tuple[str, str], KernelCandidate] = {}
_LOADED_PLATFORM_REGISTRIES: set[str] = set()


def _validate_backend_name(backend: str) -> None:
    if re.fullmatch(r"[a-z][a-z0-9_]{0,62}", backend) is None:
        raise ValueError("backend name must match [a-z][a-z0-9_]{0,62}")


def register_platform_candidate(candidate: KernelCandidate) -> None:
    """Register one explicit platform override.

    A registered platform operator always owns resolution for that operator.
    Compile-time capability failures must not trigger an implicit common
    fallback.
    """

    if candidate.ownership != "platform":
        raise ValueError(
            "platform kernel candidate ownership must be platform"
        )
    key = (candidate.backend, candidate.operation)
    if key in _PLATFORM_CANDIDATES:
        raise ValueError(
            "duplicate platform kernel candidate for "
            f"backend={candidate.backend!r}, "
            f"operation={candidate.operation!r}"
        )
    _PLATFORM_CANDIDATES[key] = candidate


def select_kernel_candidate(backend: str, operation: str) -> KernelCandidate:
    _ensure_platform_registry(backend)
    platform_candidate = _PLATFORM_CANDIDATES.get((backend, operation))
    if platform_candidate is not None:
        return platform_candidate

    common_candidate = _COMMON_CANDIDATES.get(operation)
    if common_candidate is not None:
        if common_candidate.backend == backend:
            return common_candidate
        return replace(common_candidate, backend=backend)

    raise ValueError(
        f"no kernel candidate for backend={backend!r}, "
        f"operation={operation!r}"
    )


def iter_kernel_candidates(backend: str) -> tuple[KernelCandidate, ...]:
    """Return the resolved platform view in deterministic operation order."""

    _ensure_platform_registry(backend)
    operations = set(_COMMON_CANDIDATES)
    operations.update(
        operation
        for candidate_backend, operation in _PLATFORM_CANDIDATES
        if candidate_backend == backend
    )
    return tuple(
        select_kernel_candidate(backend, operation)
        for operation in sorted(operations)
    )


def _required_string(value: object, field: str, registry_path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{registry_path}: {field} must be a nonempty string")
    return value


def _decode_tuning(
    value: object, registry_path: Path
) -> KernelTuningSpec | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{registry_path}: tuning must be an object")

    integer_fields = ("warmup", "repetitions")
    integers: dict[str, int] = {}
    for field in integer_fields:
        field_value = value.get(field)
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value <= 0
        ):
            raise ValueError(
                f"{registry_path}: tuning.{field} must be positive"
            )
        integers[field] = field_value

    return KernelTuningSpec(
        source=_required_string(
            value.get("source"), "tuning.source", registry_path
        ),
        table=_required_string(
            value.get("table"), "tuning.table", registry_path
        ),
        key=_required_string(value.get("key"), "tuning.key", registry_path),
        strategy=_required_string(
            value.get("strategy"), "tuning.strategy", registry_path
        ),
        warmup=integers["warmup"],
        repetitions=integers["repetitions"],
    )


def _decode_candidate(
    value: object,
    *,
    ownership: str,
    backend: str,
    registry_path: Path,
) -> KernelCandidate:
    if not isinstance(value, dict):
        raise ValueError(f"{registry_path}: kernel entry must be an object")

    functions_value = value.get("functions")
    if (
        not isinstance(functions_value, list)
        or not functions_value
        or any(
            not isinstance(function, str) or not function
            for function in functions_value
        )
    ):
        raise ValueError(
            f"{registry_path}: kernel functions must be nonempty strings"
        )

    expected_layout = "platform" if ownership == "platform" else "kernels"
    source_layout = _required_string(
        value.get("source_layout", expected_layout),
        "kernel.source_layout",
        registry_path,
    )
    if source_layout != expected_layout:
        raise ValueError(
            f"{registry_path}: {ownership} kernels must use "
            f"source_layout={expected_layout!r}"
        )

    source_format = _required_string(
        value.get("source_format", "module"),
        "kernel.source_format",
        registry_path,
    )
    if source_format != "module":
        raise ValueError(
            f"{registry_path}: kernels must be standalone modules"
        )

    return KernelCandidate(
        backend=backend,
        operation=_required_string(
            value.get("operation"), "kernel.operation", registry_path
        ),
        provider=_required_string(
            value.get("provider"), "kernel.provider", registry_path
        ),
        source=_required_string(
            value.get("source"), "kernel.source", registry_path
        ),
        functions=tuple(functions_value),
        source_layout=source_layout,
        source_format=source_format,
        ownership=ownership,
        tuning=_decode_tuning(value.get("tuning"), registry_path),
    )


def _decode_candidates(
    value: object,
    *,
    ownership: str,
    backend: str,
    registry_path: Path,
) -> tuple[KernelCandidate, ...]:
    if not isinstance(value, dict):
        raise ValueError(f"{registry_path}: kernel entry must be an object")

    operation = value.get("operation")
    operations = value.get("operations")
    if (operation is None) == (operations is None):
        raise ValueError(
            f"{registry_path}: kernel entry must define exactly one of "
            "operation or operations"
        )
    if operation is not None:
        operation_names = (
            _required_string(operation, "kernel.operation", registry_path),
        )
    else:
        if (
            not isinstance(operations, list)
            or not operations
            or any(
                not isinstance(item, str) or not item for item in operations
            )
            or len(set(operations)) != len(operations)
        ):
            raise ValueError(
                f"{registry_path}: kernel.operations must contain unique "
                "nonempty strings"
            )
        operation_names = tuple(operations)

    candidates: list[KernelCandidate] = []
    for operation_name in operation_names:
        expanded = dict(value)
        expanded.pop("operations", None)
        expanded["operation"] = operation_name
        candidates.append(
            _decode_candidate(
                expanded,
                ownership=ownership,
                backend=backend,
                registry_path=registry_path,
            )
        )
    return tuple(candidates)


def _load_registry(
    registry_path: Path,
    *,
    expected_ownership: str,
    expected_backend: str | None,
) -> None:
    if not registry_path.is_file():
        return
    document = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError(f"{registry_path}: unsupported registry schema")
    if document.get("ownership") != expected_ownership:
        raise ValueError(f"{registry_path}: registry ownership is invalid")

    if expected_ownership == "platform":
        backend = _required_string(
            document.get("backend"), "backend", registry_path
        )
        if backend != expected_backend:
            raise ValueError(f"{registry_path}: backend name is invalid")
    else:
        backend = "common"

    kernels = document.get("kernels")
    if not isinstance(kernels, list):
        raise ValueError(f"{registry_path}: kernels must be an array")

    seen: set[str] = set()
    for entry in kernels:
        for candidate in _decode_candidates(
            entry,
            ownership=expected_ownership,
            backend=backend,
            registry_path=registry_path,
        ):
            if candidate.operation in seen:
                raise ValueError(
                    f"{registry_path}: duplicate operation "
                    f"{candidate.operation!r}"
                )
            seen.add(candidate.operation)
            if expected_ownership == "common":
                _COMMON_CANDIDATES[candidate.operation] = candidate
            else:
                register_platform_candidate(candidate)


def _registry_resource_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_kernel_registry_sources(backend: str) -> tuple[Path, ...]:
    _validate_backend_name(backend)
    root = _registry_resource_root()
    candidates = (
        root / "kernels" / "registry.json",
        root / "backends" / backend / "kernels" / "registry.json",
    )
    return tuple(path for path in candidates if path.is_file())


def _load_common_registry() -> None:
    root = _registry_resource_root()
    _load_registry(
        root / "kernels" / "registry.json",
        expected_ownership="common",
        expected_backend=None,
    )


def _ensure_platform_registry(backend: str) -> None:
    _validate_backend_name(backend)
    if backend in _LOADED_PLATFORM_REGISTRIES:
        return
    root = _registry_resource_root()
    _load_registry(
        root / "backends" / backend / "kernels" / "registry.json",
        expected_ownership="platform",
        expected_backend=backend,
    )
    _LOADED_PLATFORM_REGISTRIES.add(backend)


_load_common_registry()


def _resource_root(compiler_path: Path) -> Path:
    resolved = compiler_path.resolve()
    if len(resolved.parents) < 3:
        raise ValueError("compiler path is outside a FlagDNN resource layout")
    return resolved.parents[2]


def _append_configured_roots(roots: list[Path], environment_name: str) -> None:
    configured = os.environ.get(environment_name, "")
    for value in configured.split(os.pathsep):
        if value:
            roots.append(Path(value).expanduser())


def _kernel_source_roots(
    compiler_path: Path, candidate: KernelCandidate
) -> tuple[Path, ...]:
    resource_root = _resource_root(compiler_path)
    roots: list[Path] = []

    if candidate.source_layout == "kernels":
        _append_configured_roots(roots, "FLAGDNN_KERNEL_SOURCE_ROOT")
        roots.append(resource_root / "kernels" / "common")
    elif candidate.source_layout == "platform":
        _append_configured_roots(roots, "FLAGDNN_BACKEND_ROOT")
        roots.append(
            resource_root / "backends" / candidate.backend / "kernels"
        )
    else:
        raise ValueError(
            f"unknown kernel source layout: {candidate.source_layout!r}"
        )

    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _resolve_relative_source(
    roots: tuple[Path, ...], source: str, description: str
) -> Path:
    relative = Path(source)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{description} path is unsafe")

    searched: list[str] = []
    for root in roots:
        path = root / relative
        searched.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{description} was not found; searched " + ", ".join(searched)
    )


def resolve_kernel_source(
    compiler_path: Path, candidate: KernelCandidate
) -> Path:
    """Resolve canonical kernel sources in source and installed layouts."""

    return _resolve_relative_source(
        _kernel_source_roots(compiler_path, candidate),
        candidate.source,
        "canonical kernel source",
    )


def resolve_tuning_source(
    compiler_path: Path, candidate: KernelCandidate
) -> Path:
    """Resolve tuning data without importing the Python compatibility API."""

    if candidate.tuning is None:
        raise ValueError("kernel candidate does not define a tuning source")

    roots = list(_kernel_source_roots(compiler_path, candidate))
    resource_root = _resource_root(compiler_path)
    _append_configured_roots(roots, "FLAGDNN_TUNING_ROOT")
    roots.append(resource_root / "backends" / candidate.backend / "tuning")
    return _resolve_relative_source(
        tuple(roots),
        candidate.tuning.source,
        "kernel tuning source",
    )


def materialize_kernel_source(
    source: Path, candidate: KernelCandidate
) -> bytes:
    """Return a standalone kernel module accepted by the compiler."""

    if candidate.source_format != "module":
        raise ValueError(
            "kernel sources must be standalone modules; got "
            f"{candidate.source_format!r}"
        )
    return source.read_bytes()
