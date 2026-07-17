from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True)
class CannLayout:
    root: Path
    include_dir: Path
    lib_dir: Path


_ROOT_ENV_VARS = (
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_TOOLKIT_LATEST_HOME",
)
_DEFAULT_ROOTS = (
    Path("/usr/local/Ascend/ascend-toolkit/latest"),
    Path("/usr/local/Ascend/latest"),
)
_PROCESS_OUTPUT_UNAVAILABLE = "unavailable (compiler process did not run)"
_NATIVE_ROOT = Path(__file__).parent / "csrc"


def _candidate_roots(environ: Mapping[str, str]) -> tuple[Path, ...]:
    candidates = [
        Path(environ[name]) for name in _ROOT_ENV_VARS if environ.get(name)
    ]
    candidates.extend(_DEFAULT_ROOTS)
    result = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return tuple(result)


def _layout_candidates(root: Path, machine: str) -> tuple[CannLayout, ...]:
    arch = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
    prefixes = (Path(), Path(f"{arch}-linux"))
    return tuple(
        CannLayout(
            root=root,
            include_dir=root / prefix / "include",
            lib_dir=root / prefix / "lib64",
        )
        for prefix in prefixes
    )


def _is_valid_layout(layout: CannLayout) -> bool:
    return all(
        path.is_file()
        for path in (
            layout.include_dir / "aclnnop" / "aclnn_add.h",
            layout.include_dir / "aclnnop" / "aclnn_abs.h",
            layout.lib_dir / "libascendcl.so",
            layout.lib_dir / "libopapi.so",
        )
    )


def find_cann_layout(
    environ: Optional[Mapping[str, str]] = None,
    machine: Optional[str] = None,
) -> CannLayout:
    env = os.environ if environ is None else environ
    host_machine = platform.machine() if machine is None else machine
    attempted = []
    for root in _candidate_roots(env):
        for layout in _layout_candidates(root, host_machine):
            attempted.append(
                f"include={layout.include_dir}, lib={layout.lib_dir}"
            )
            if _is_valid_layout(layout):
                return layout
    raise RuntimeError(
        "Could not find CANN ACLNN headers and libraries. Checked:\n"
        + "\n".join(attempted)
    )


def _compiler_command(environ: Mapping[str, str]) -> list[str]:
    command = shlex.split(environ.get("CXX", "c++"))
    if not command:
        raise RuntimeError("CXX resolved to an empty compiler command")
    executable = shutil.which(command[0])
    if executable is None:
        raise RuntimeError(f"C++ compiler was not found: {command[0]}")
    return [executable, *command[1:]]


def _display_command(command: list[str]) -> str:
    return " ".join(command)


def _build_failure(
    summary: str,
    *,
    command: list[str],
    compiler: str,
    layout: CannLayout,
    architecture: str,
    stdout: str,
    stderr: str,
) -> RuntimeError:
    return RuntimeError(
        f"{summary}:\n"
        f"command: {_display_command(command)}\n"
        f"compiler: {compiler}\n"
        f"architecture: {architecture}\n"
        f"CANN include: {layout.include_dir}\n"
        f"CANN lib: {layout.lib_dir}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def _compiler_identity(
    command: list[str], layout: CannLayout, architecture: str
) -> str:
    version_command = [*command, "--version"]
    try:
        completed = subprocess.run(
            version_command,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise _build_failure(
            f"Failed to query C++ compiler version ({exc})",
            command=version_command,
            compiler=_display_command(command),
            layout=layout,
            architecture=architecture,
            stdout=_PROCESS_OUTPUT_UNAVAILABLE,
            stderr=_PROCESS_OUTPUT_UNAVAILABLE,
        ) from exc
    if completed.returncode != 0:
        raise _build_failure(
            "Failed to query C++ compiler version",
            command=version_command,
            compiler=_display_command(command),
            layout=layout,
            architecture=architecture,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    return completed.stdout + completed.stderr


def _normalized_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(sorted({Path(path).resolve() for path in paths}, key=str))


def _default_source_paths() -> tuple[Path, ...]:
    return _normalized_paths(
        (
            *(_NATIVE_ROOT / "common").glob("*.cpp"),
            *(_NATIVE_ROOT / "ops").glob("*.cpp"),
        )
    )


def _default_header_paths() -> tuple[Path, ...]:
    return _normalized_paths((_NATIVE_ROOT / "common").glob("*.h"))


def _logical_inputs(
    sources: tuple[Path, ...], headers: tuple[Path, ...]
) -> tuple[tuple[str, bytes], ...]:
    paths = (*sources, *headers)
    if not sources:
        raise RuntimeError("ACLNN oracle source collection is empty")
    common_root = Path(
        os.path.commonpath([str(path.parent) for path in paths])
    )
    values = []
    for path in paths:
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read ACLNN oracle input {path}: {exc}"
            ) from exc
        values.append((path.relative_to(common_root).as_posix(), content))
    return tuple(values)


def _cache_key(
    inputs: tuple[tuple[str, bytes], ...],
    layout: CannLayout,
    compiler: list[str],
    compiler_identity: str,
) -> str:
    digest = hashlib.sha256()
    for logical_path, content in inputs:
        digest.update(logical_path.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    for value in (
        str(layout.include_dir.resolve()).encode(),
        str(layout.lib_dir.resolve()).encode(),
        "\0".join(compiler).encode(),
        compiler_identity.encode(),
        platform.machine().encode(),
    ):
        digest.update(value)
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def build_aclnn_oracle(
    *,
    layout: Optional[CannLayout] = None,
    source_paths: Optional[Iterable[Path]] = None,
    header_paths: Optional[Iterable[Path]] = None,
    build_root: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Path:
    env = os.environ if environ is None else environ
    selected_layout = find_cann_layout(env) if layout is None else layout
    architecture = platform.machine()
    sources = _normalized_paths(
        _default_source_paths() if source_paths is None else source_paths
    )
    headers = _normalized_paths(
        _default_header_paths() if header_paths is None else header_paths
    )
    root = (
        Path(__file__).resolve().parents[4] / "build" / "dnn_reference"
        if build_root is None
        else build_root
    )
    requested_compiler = env.get("CXX", "c++")
    attempted_compiler = [requested_compiler or "<empty CXX>"]
    try:
        parsed_compiler = shlex.split(requested_compiler)
        if parsed_compiler:
            attempted_compiler = parsed_compiler
        compiler = _compiler_command(env)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _build_failure(
            f"Failed to configure C++ compiler ({exc})",
            command=attempted_compiler,
            compiler=_display_command(attempted_compiler),
            layout=selected_layout,
            architecture=architecture,
            stdout=_PROCESS_OUTPUT_UNAVAILABLE,
            stderr=_PROCESS_OUTPUT_UNAVAILABLE,
        ) from exc
    identity = _compiler_identity(compiler, selected_layout, architecture)
    compiler_description = _display_command(compiler)
    if identity.strip():
        compiler_description += f" ({identity.strip()})"
    inputs = _logical_inputs(sources, headers)
    key = _cache_key(inputs, selected_layout, compiler, identity)
    output_dir = root / "ascend" / key
    output = output_dir / "libflagdnn_test_aclnn.so"
    if output.is_file():
        return output

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as temporary_file:
            temporary = Path(temporary_file.name)
    except OSError as exc:
        raise _build_failure(
            f"Failed to prepare ACLNN oracle output ({exc})",
            command=compiler,
            compiler=compiler_description,
            layout=selected_layout,
            architecture=architecture,
            stdout=_PROCESS_OUTPUT_UNAVAILABLE,
            stderr=_PROCESS_OUTPUT_UNAVAILABLE,
        ) from exc
    command = [
        *compiler,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wextra",
        *(str(source) for source in sources),
        "-o",
        str(temporary),
        "-I",
        str(_NATIVE_ROOT),
        "-I",
        str(selected_layout.include_dir),
        "-L",
        str(selected_layout.lib_dir),
        f"-Wl,-rpath,{selected_layout.lib_dir}",
        "-lascendcl",
        "-lopapi",
    ]
    try:
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            raise _build_failure(
                f"Failed to invoke the ACLNN test oracle compiler ({exc})",
                command=command,
                compiler=compiler_description,
                layout=selected_layout,
                architecture=architecture,
                stdout=_PROCESS_OUTPUT_UNAVAILABLE,
                stderr=_PROCESS_OUTPUT_UNAVAILABLE,
            ) from exc
        if completed.returncode != 0:
            raise _build_failure(
                "Failed to build the ACLNN test oracle",
                command=command,
                compiler=compiler_description,
                layout=selected_layout,
                architecture=architecture,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        try:
            os.replace(temporary, output)
        except OSError as exc:
            raise _build_failure(
                f"Failed to publish the ACLNN test oracle ({exc})",
                command=command,
                compiler=compiler_description,
                layout=selected_layout,
                architecture=architecture,
                stdout=completed.stdout,
                stderr=completed.stderr,
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return output
