#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed environment preflight for the FlagDNN THead backend."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tempfile
from typing import Any, Iterable
import uuid


sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triton_compat import configure_triton_path, ppu_codegen_backend, ppu_distribution
from TritonIdentityContract import IdentityError, _jit_backend


_TOP_LEVEL_FIELDS = (
    "schema_version",
    "ppu_sdk",
    "cuda_compat",
    "triton",
    "libtriton_jit",
    "python",
    "device",
)


class PreflightError(RuntimeError):
    """A stable, user-actionable preflight failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _required_directory(path: Path, name: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise PreflightError("static_provenance_error", f"{name}: {error}") from error
    if not resolved.is_dir():
        raise PreflightError(
            "static_provenance_error", f"{name} is not a directory: {resolved}"
        )
    return resolved


def _required_file(
    path: Path,
    name: str,
    *,
    root: Path | None = None,
    executable: bool = False,
) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise PreflightError("static_provenance_error", f"{name}: {error}") from error
    if not resolved.is_file():
        raise PreflightError(
            "static_provenance_error", f"{name} is not a file: {resolved}"
        )
    if root is not None and not resolved.is_relative_to(root):
        raise PreflightError(
            "static_provenance_error",
            f"{name} escaped configured root {root}: {resolved}",
        )
    if executable and not os.access(resolved, os.X_OK):
        raise PreflightError(
            "static_provenance_error", f"{name} is not executable: {resolved}"
        )
    return resolved


def _single_match(pattern: str, text: str, name: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise PreflightError(
            "static_provenance_error",
            f"expected one {name}, found {len(matches)}",
        )
    return matches[0].strip()


def _sdk_version(release_file: Path) -> str:
    version = _single_match(
        r"^[ \t]*version[ \t]*:[ \t]*['\"]?([^'\"#\r\n]+)['\"]?[ \t]*(?:#.*)?$",
        release_file.read_text(encoding="utf-8"),
        "PPU SDK version",
    )
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+_-]*", version):
        raise PreflightError(
            "static_provenance_error", f"malformed PPU SDK version: {version!r}"
        )
    return version


def _sha256_files(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            relative = resolved
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _find_repository(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate.resolve(strict=True)
    return None


def _source_identity(
    root: Path, identity_files: Iterable[Path]
) -> tuple[Path, dict[str, str]]:
    repository = _find_repository(root)
    if repository is not None:
        head = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        revision = head.stdout.strip()
        if (
            head.returncode == 0
            and status.returncode == 0
            and not status.stdout
            and re.fullmatch(r"[0-9a-f]{40}", revision)
        ):
            return repository, {"kind": "git_commit", "value": revision}
    identity_root = repository if repository is not None else root
    return identity_root, {
        "kind": "content_sha256",
        "value": _sha256_files(identity_files, identity_root),
    }


def _probe_sdk(sdk_root_argument: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sdk_root = _required_directory(sdk_root_argument, "PPU SDK root")
    release_file = _required_file(
        sdk_root / "release.yaml", "PPU SDK release file", root=sdk_root
    )
    hggc_library = _required_file(
        sdk_root / "lib" / "libhggc.so", "HGGC library", root=sdk_root
    )
    acdnn_header = _required_file(
        sdk_root / "include" / "acdnn.h", "acDNN header", root=sdk_root
    )
    acdnn_library = _required_file(
        sdk_root / "lib" / "libacdnn.so", "acDNN library", root=sdk_root
    )

    cuda_root = _required_directory(sdk_root / "CUDA_SDK", "CUDA compatibility root")
    cuda_header = _required_file(
        cuda_root / "include" / "cuda.h", "CUDA compatibility header", root=cuda_root
    )
    driver_library = _required_file(
        cuda_root / "lib64" / "libcuda.so.1",
        "CUDA compatibility driver",
        root=cuda_root,
    )
    compiler = _required_file(
        cuda_root / "bin" / "nvcc",
        "CUDA compatibility compiler",
        root=cuda_root,
        executable=True,
    )
    return (
        {
            "root": str(sdk_root),
            "version": _sdk_version(release_file),
            "release_file": str(release_file),
            "hggc_library": str(hggc_library),
            "acdnn_header": str(acdnn_header),
            "acdnn_library": str(acdnn_library),
        },
        {
            "root": str(cuda_root),
            "header": str(cuda_header),
            "driver_library": str(driver_library),
            "compiler": str(compiler),
            "ptxas": str(
                _required_file(
                    cuda_root / "bin" / "ptxas",
                    "PPU CUDA compatibility assembler",
                    root=cuda_root,
                    executable=True,
                )
            ),
            "ir_formatter": str(
                _required_file(
                    sdk_root / "bin" / "llvm-irformatter",
                    "PPU LLVM IR formatter",
                    root=sdk_root,
                    executable=True,
                )
            ),
        },
    )


def _probe_triton(triton_root_argument: Path) -> dict[str, Any]:
    triton_root = _required_directory(triton_root_argument, "Triton package root")
    configure_triton_path(triton_root)
    try:
        triton = importlib.import_module("triton")
        backend_module = importlib.import_module("triton.backends")
        compiler_api = importlib.import_module("triton.backends.compiler")
        compiler_frontend = importlib.import_module("triton.compiler.compiler")
    except Exception as error:
        raise PreflightError(
            "static_provenance_error", f"cannot import configured Triton: {error}"
        ) from error

    package_file_value = getattr(triton, "__file__", None)
    if not package_file_value:
        raise PreflightError("static_provenance_error", "triton has no package file")
    package_file = _required_file(
        Path(package_file_value), "Triton package file", root=triton_root
    )
    try:
        codegen_backend = ppu_codegen_backend(triton_root)
        distribution, _ = ppu_distribution(triton_root)
    except RuntimeError as error:
        raise PreflightError("static_provenance_error", str(error)) from error
    catalog = getattr(backend_module, "backends", None)
    if not isinstance(catalog, dict) or codegen_backend not in catalog:
        raise PreflightError(
            "static_provenance_error",
            f"configured Triton backend catalog has no CUDA codegen backend: {catalog!r}",
        )
    cuda_backend = catalog[codegen_backend]
    compiler_source = inspect.getsourcefile(cuda_backend.compiler)
    driver_source = inspect.getsourcefile(cuda_backend.driver)
    frontend_source = inspect.getsourcefile(compiler_frontend)
    if compiler_source is None or driver_source is None:
        raise PreflightError(
            "static_provenance_error", "Triton CUDA compiler/driver has no source file"
        )
    compiler_file = _required_file(
        Path(compiler_source), "Triton CUDA compiler", root=triton_root
    )
    driver_file = _required_file(
        Path(driver_source), "Triton CUDA driver", root=triton_root
    )
    frontend_file = _required_file(
        Path(frontend_source or ""), "Triton compiler frontend", root=triton_root
    )
    compiler_text = compiler_file.read_text(encoding="utf-8")
    if any(
        marker not in compiler_text
        for marker in ("PPU_SDK", "llvm-irformatter", "--ppu-backend-options")
    ):
        raise PreflightError(
            "static_provenance_error",
            "Triton CUDA compiler lacks the PPU compatibility path",
        )
    target_type = getattr(compiler_api, "GPUTarget", None)
    if target_type is None:
        raise PreflightError("static_provenance_error", "Triton has no GPUTarget API")
    backend = cuda_backend.compiler(target_type("cuda", 80, 32))
    expected_extension = "hgbin" if codegen_backend == "ppu" else "cubin"
    if getattr(backend, "binary_ext", None) != expected_extension:
        raise PreflightError(
            "static_provenance_error", "unexpected Triton PPU binary extension"
        )
    metadata_files = [
        Path(distribution.locate_file(entry)).resolve(strict=True)
        for entry in distribution.files or ()
        if entry.name == "METADATA" and entry.parent.name.endswith(".dist-info")
    ]
    if (
        len(metadata_files) != 1
        or not metadata_files[0].is_relative_to(triton_root)
        or "ppu" not in distribution.version.lower()
    ):
        raise PreflightError(
            "static_provenance_error",
            "configured Triton distribution is not the installed PPU-qualified build",
        )
    metadata_file = metadata_files[0]
    repository, source_identity = _source_identity(
        triton_root,
        (package_file, compiler_file, driver_file, frontend_file, metadata_file),
    )
    module_version = getattr(triton, "__version__", None)
    if not isinstance(module_version, str) or not module_version:
        raise PreflightError("static_provenance_error", "Triton version is missing")
    return {
        "root": str(triton_root),
        "repository_root": str(repository),
        "package_file": str(package_file),
        "module_version": module_version,
        "distribution_version": distribution.version,
        "distribution_metadata": str(metadata_file),
        "backend_catalog": sorted(catalog),
        "codegen_backend": codegen_backend,
        "target_backend": "cuda",
        "compiler_file": str(compiler_file),
        "driver_file": str(driver_file),
        "compiler_frontend_file": str(frontend_file),
        "binary_extension": backend.binary_ext,
        "ppu_compatibility": "cuda",
        "source_identity": source_identity,
    }


def _probe_triton_jit(jit_root_argument: Path) -> dict[str, Any]:
    jit_root = _required_directory(jit_root_argument, "libtriton_jit root")
    try:
        config_file, backend = _jit_backend(jit_root)
    except (IdentityError, OSError) as error:
        raise PreflightError("static_provenance_error", str(error)) from error
    if config_file.parent == jit_root / "build":
        library_path = jit_root / "build/src/libtriton_jit.so"
        scripts_path = jit_root / "scripts"
    elif config_file.parent in (
        jit_root / "lib/cmake/TritonJIT",
        jit_root / "lib64/cmake/TritonJIT",
    ):
        library_path = config_file.parents[2] / "libtriton_jit.so"
        scripts_path = jit_root / "share/triton_jit/scripts"
    else:
        raise PreflightError(
            "static_provenance_error", "unrecognized libtriton_jit configuration layout"
        )
    library = _required_file(
        library_path,
        "libtriton_jit library",
        root=jit_root,
    )
    script_directory = _required_directory(
        scripts_path, "libtriton_jit script directory"
    )
    if not script_directory.is_relative_to(jit_root):
        raise PreflightError(
            "static_provenance_error", "libtriton_jit script directory escaped root"
        )
    standalone_compile = _required_file(
        script_directory / "standalone_compile.py",
        "standalone_compile.py",
        root=jit_root,
    )
    gen_ssig = _required_file(
        script_directory / "gen_ssig.py", "gen_ssig.py", root=jit_root
    )
    identity_files = (config_file, library, standalone_compile, gen_ssig)
    if (jit_root / ".git").exists():
        repository, source_identity = _source_identity(jit_root, identity_files)
    else:
        # An installed SDK may live below an unrelated checkout. Its identity
        # must describe these installed files, not that checkout's Git revision.
        repository = jit_root
        source_identity = {
            "kind": "content_sha256",
            "value": _sha256_files(identity_files, jit_root),
        }
    return {
        "root": str(jit_root),
        "repository_root": str(repository),
        "config_file": str(config_file),
        "backend": backend,
        "library": str(library),
        "script_directory": str(script_directory),
        "standalone_compile": str(standalone_compile),
        "gen_ssig": str(gen_ssig),
        "source_identity": source_identity,
    }


def _probe_python() -> dict[str, Any]:
    executable = _required_file(
        Path(sys.executable), "Python executable", executable=True
    )
    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        raise PreflightError(
            "static_provenance_error", f"cannot import Torch: {error}"
        ) from error
    torch_file_value = getattr(torch, "__file__", None)
    torch_version = getattr(torch, "__version__", None)
    if not torch_file_value or not isinstance(torch_version, str) or not torch_version:
        raise PreflightError(
            "static_provenance_error", "Torch package identity is incomplete"
        )
    torch_file = _required_file(Path(torch_file_value), "Torch package file")
    abi = sysconfig.get_config_var("SOABI")
    if not isinstance(abi, str) or not abi:
        raise PreflightError("static_provenance_error", "Python SOABI is missing")
    return {
        "executable": str(executable),
        "version": sys.version.split()[0],
        "implementation": sys.implementation.name,
        "abi": abi,
        "cache_tag": sys.implementation.cache_tag,
        "torch_version": torch_version,
        "torch_file": str(torch_file),
    }


def _validate_schema_contract(schema_path_argument: Path) -> Path:
    schema_path = _required_file(schema_path_argument, "environment schema")
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PreflightError(
            "static_provenance_error", f"environment schema is invalid JSON: {error}"
        ) from error
    if (
        not isinstance(schema, dict)
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or schema.get("required") != list(_TOP_LEVEL_FIELDS)
    ):
        raise PreflightError(
            "static_provenance_error", "environment schema top-level contract drifted"
        )
    return schema_path


def _static_document(arguments: argparse.Namespace) -> dict[str, Any]:
    _validate_schema_contract(arguments.schema)
    ppu_sdk, cuda_compat = _probe_sdk(arguments.sdk_root)
    triton = _probe_triton(arguments.triton_root)
    triton_jit = _probe_triton_jit(arguments.triton_jit_root)
    python = _probe_python()
    return {
        "schema_version": 2,
        "ppu_sdk": ppu_sdk,
        "cuda_compat": cuda_compat,
        "triton": triton,
        "libtriton_jit": triton_jit,
        "python": python,
        "device": None,
    }


def _run_ppu_smi(executable_argument: Path) -> Path:
    executable = _required_file(executable_argument, "ppu-smi", executable=True)
    try:
        result = subprocess.run(
            [str(executable)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as error:
        raise PreflightError("driver_not_loaded", "ppu-smi timed out") from error
    combined = "\n".join((result.stdout, result.stderr)).strip()
    normalized = combined.lower()
    unavailable_markers = (
        "driver is not loaded",
        "init hgml error",
        "failed to initialize",
        "no devices were found",
    )
    if result.returncode != 0 or any(
        marker in normalized for marker in unavailable_markers
    ):
        detail = combined or f"ppu-smi exited with status {result.returncode}"
        raise PreflightError("driver_not_loaded", detail)
    return executable


def _active_triton_backend(triton_root: Path) -> dict[str, Any]:
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from triton_compat import configure_triton_path; "
        "configure_triton_path(sys.argv[2]); "
        "import json; from triton.backends import backends; "
        "from triton.runtime import driver; "
        "active=[name for name, value in backends.items() "
        "if value.driver.is_active()]; "
        "target=driver.active.get_current_target(); "
        "print(json.dumps({'active':active,'backend':target.backend,"
        "'arch':target.arch,'warp_size':target.warp_size},separators=(',', ':')))"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", probe,
         str(Path(__file__).resolve().parents[1]), str(triton_root)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    if result.returncode != 0:
        raise PreflightError(
            "driver_not_loaded",
            f"Triton active-driver probe failed: {result.stderr.strip()}",
        )
    try:
        active = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise PreflightError(
            "driver_not_loaded", "Triton active-driver probe returned invalid JSON"
        ) from error
    if (
        not isinstance(active, dict)
        or active.get("active") != [ppu_codegen_backend(triton_root)]
        or active.get("backend") != "cuda"
        or active.get("warp_size") != 32
        or not isinstance(active.get("arch"), int)
    ):
        raise PreflightError(
            "driver_not_loaded",
            f"expected one active Triton CUDA driver on PPU, got {active!r}",
        )
    return active


def _cuda_call(function: Any, name: str, *arguments: Any) -> None:
    result = int(function(*arguments))
    if result != 0:
        code = "driver_not_loaded" if name == "cuInit" else "driver_probe_failed"
        raise PreflightError(code, f"{name} failed with CUDA result {result}")


def _probe_device(
    ordinal: int,
    static_document: dict[str, Any],
    ppu_smi: Path,
) -> dict[str, Any]:
    if ordinal < 0:
        raise PreflightError("invalid_device", f"negative device ordinal: {ordinal}")
    smi_path = _run_ppu_smi(ppu_smi)
    active_backend = _active_triton_backend(
        Path(static_document["triton"]["root"])
    )
    driver_path = static_document["cuda_compat"]["driver_library"]
    try:
        driver = ctypes.CDLL(driver_path)
    except OSError as error:
        raise PreflightError(
            "driver_not_loaded", f"cannot load CUDA compatibility driver: {error}"
        ) from error

    _cuda_call(driver.cuInit, "cuInit", ctypes.c_uint(0))
    count = ctypes.c_int()
    _cuda_call(driver.cuDeviceGetCount, "cuDeviceGetCount", ctypes.byref(count))
    if count.value <= 0:
        raise PreflightError(
            "driver_not_loaded", "CUDA compatibility driver found no PPU"
        )
    if ordinal >= count.value:
        raise PreflightError(
            "invalid_device", f"device {ordinal} is outside [0, {count.value})"
        )

    device = ctypes.c_int()
    _cuda_call(driver.cuDeviceGet, "cuDeviceGet", ctypes.byref(device), ordinal)
    name_buffer = ctypes.create_string_buffer(256)
    _cuda_call(
        driver.cuDeviceGetName,
        "cuDeviceGetName",
        name_buffer,
        len(name_buffer),
        device,
    )
    major = ctypes.c_int()
    minor = ctypes.c_int()
    _cuda_call(
        driver.cuDeviceComputeCapability,
        "cuDeviceComputeCapability",
        ctypes.byref(major),
        ctypes.byref(minor),
        device,
    )
    pci_buffer = ctypes.create_string_buffer(64)
    _cuda_call(
        driver.cuDeviceGetPCIBusId,
        "cuDeviceGetPCIBusId",
        pci_buffer,
        len(pci_buffer),
        device,
    )

    class _CudaUuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    raw_uuid = _CudaUuid()
    _cuda_call(
        driver.cuDeviceGetUuid,
        "cuDeviceGetUuid",
        ctypes.byref(raw_uuid),
        device,
    )
    driver_version = ctypes.c_int()
    _cuda_call(
        driver.cuDriverGetVersion, "cuDriverGetVersion", ctypes.byref(driver_version)
    )
    name = name_buffer.value.decode("utf-8", errors="strict")
    capability_code = major.value * 10 + minor.value
    if "PPU" not in name.upper():
        raise PreflightError("wrong_device", f"selected device is not a PPU: {name}")
    if active_backend["arch"] != capability_code:
        raise PreflightError(
            "driver_probe_failed",
            "Triton target architecture differs from the CUDA compatibility device",
        )
    return {
        "ordinal": ordinal,
        "name": name,
        "uuid": str(uuid.UUID(bytes=bytes(raw_uuid.bytes))),
        "pci_bus_id": pci_buffer.value.decode("ascii", errors="strict"),
        "capability": f"{major.value}.{minor.value}",
        "capability_code": capability_code,
        "driver_version": driver_version.value,
        "triton_registry_backend": active_backend["active"][0],
        "triton_target_backend": active_backend["backend"],
        "triton_target_arch": active_backend["arch"],
        "triton_warp_size": active_backend["warp_size"],
        "ppu_smi": str(smi_path),
    }


def _write_output(path_argument: Path, serialized: str) -> None:
    path = path_argument.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            stream.write(serialized)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary_name = stream.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true", help="probe without a device")
    mode.add_argument("--device", type=int, help="probe the selected PPU device")
    parser.add_argument("--sdk-root", type=Path, default=Path("/usr/local/PPU_SDK"))
    parser.add_argument("--triton-root", type=Path, required=True)
    parser.add_argument("--triton-jit-root", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().with_name("environment.schema.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ppu-smi", type=Path)
    arguments = parser.parse_args()
    if arguments.ppu_smi is None:
        arguments.ppu_smi = arguments.sdk_root / "ppu-smi" / "bin" / "ppu-smi"
    return arguments


def main() -> int:
    arguments = _arguments()
    document = _static_document(arguments)
    if arguments.device is not None:
        document["device"] = _probe_device(
            arguments.device, document, arguments.ppu_smi
        )
    serialized = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    if arguments.output is not None:
        _write_output(arguments.output, serialized)
    print(serialized)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightError as error:
        print(
            json.dumps(
                {"error": error.code, "detail": error.detail},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
    except Exception as error:
        print(
            json.dumps(
                {"error": "preflight_internal_error", "detail": str(error)},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(3) from error
