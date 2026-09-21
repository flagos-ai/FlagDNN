# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Identify installed vendor Triton and FlagTree PPU distributions."""

from __future__ import annotations

from functools import lru_cache
import importlib.metadata
from pathlib import Path
import sys
import sysconfig
from time import time_ns


def configure_triton_path(root: str | Path) -> None:
    """Prefer the selected package root without shadowing Python's stdlib.

    A site-packages directory may contain backports such as dataclasses.
    Putting it before the stdlib breaks otherwise valid system installations.
    """
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"Triton package root is not a directory: {root}")
    standard_paths = {
        Path(sysconfig.get_path(name)).resolve()
        for name in ("stdlib", "platstdlib")
    }
    standard_paths.update(
        path.parent / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
        for path in tuple(standard_paths)
    )
    extension_path = sysconfig.get_config_var("DESTSHARED")
    if extension_path:
        standard_paths.add(Path(extension_path).resolve())
    paths = [
        entry for entry in sys.path
        if not isinstance(entry, str) or Path(entry).resolve() != root
    ]
    position = max(
        (index + 1 for index, entry in enumerate(paths)
         if isinstance(entry, str) and Path(entry).resolve() in standard_paths),
        default=0,
    )
    paths.insert(position, str(root))
    sys.path[:] = paths


def _metadata_state(path: Path) -> tuple[int, ...] | None:
    try:
        status = path.stat()
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return None
    return (
        status.st_dev, status.st_ino, status.st_mode, status.st_size,
        status.st_mtime_ns, status.st_ctime_ns,
    )


@lru_cache(maxsize=2048)
def _distribution_name(path: Path, _state: tuple) -> str:
    # Only cache the name used to select a distribution. The returned public
    # Distribution still reads version, RECORD, and file existence normally.
    return importlib.metadata.PathDistribution(path).metadata.get("Name", "").lower()


def _current_distribution_name(path: Path) -> str:
    # Include importlib's PKG-INFO fallback and directory identity. ctime/inode
    # also catch replacement or same-size writes with the old mtime restored.
    state = tuple(_metadata_state(item) for item in (
        path, path / "METADATA", path / "PKG-INFO",
    ))
    # Some filesystems update ctime only once per clock tick. Re-read recent
    # files so a same-size write with restored mtime cannot reuse that tick's
    # cached name. Installed packages normally have much older timestamps.
    cutoff = time_ns() - 2_000_000_000
    if any(item is not None and max(item[-2:]) > cutoff for item in state):
        return importlib.metadata.PathDistribution(path).metadata.get("Name", "").lower()
    return _distribution_name(path, state)


def ppu_distribution(
    root: Path,
) -> tuple[importlib.metadata.Distribution, Path]:
    """Select metadata from the same root as the imported Python package."""
    root = root.resolve(strict=True)
    candidates = [
        importlib.metadata.PathDistribution(metadata)
        for metadata in sorted(root.glob("*.dist-info"))
        if _current_distribution_name(metadata) in {"triton", "flagtree"}
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "configured root must contain exactly one Triton/FlagTree distribution"
        )
    distribution = candidates[0]
    if "ppu" not in distribution.version.lower():
        raise RuntimeError(
            "configured Triton distribution is not the PPU-qualified build"
        )
    metadata = [
        Path(distribution.locate_file(entry)).resolve(strict=True)
        for entry in distribution.files or ()
        if entry.name == "METADATA" and entry.parent.name.endswith(".dist-info")
    ]
    if len(metadata) != 1 or not metadata[0].is_relative_to(root):
        raise RuntimeError(
            "Triton distribution metadata is missing or outside configured root"
        )
    return distribution, metadata[0]


def ppu_codegen_backend(root: Path) -> str:
    distribution, _ = ppu_distribution(root)
    name = distribution.metadata.get("Name", "").lower()
    return "ppu" if name == "flagtree" else "nvidia"


def install_cuda_jit_bridge(standalone) -> None:
    """Expose PPU ELF bytes under the filename required by the CUDA JIT loader.

    FlagTree emits hgbin; PPU cuModuleLoad accepts those bytes directly.
    Only the libtriton_jit filename convention differs.
    """
    import functools
    import os
    import tempfile

    original = standalone.compile_a_kernel
    if getattr(original, "_flagdnn_ppu_bridge", False):
        return

    @functools.wraps(original)
    def compile_a_kernel(*args, **kwargs):
        cache = Path(original(*args, **kwargs)).resolve(strict=True)
        function = args[1] if len(args) > 1 else kwargs["fn_name"]
        if not isinstance(function, str) or not function.isidentifier():
            raise RuntimeError("invalid PPU kernel function name")
        binary = cache / (function + ".hgbin")
        if not binary.is_file() or binary.is_symlink():
            raise RuntimeError("PPU compiler did not emit a regular hgbin")
        content = binary.read_bytes()
        if not content.startswith(b"\x7fELF"):
            raise RuntimeError("PPU compiler binary is not ELF")
        destination = cache / (function + ".cubin")
        is_alias = destination.is_symlink()
        if is_alias and destination.resolve() != binary:
            raise RuntimeError("PPU CUDA loader symlink escaped its hgbin")
        if (
            is_alias
            or not destination.is_file()
            or destination.read_bytes() != content
        ):
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=cache, delete=False
                ) as output:
                    temporary = Path(output.name)
                    output.write(content)
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return str(cache)

    compile_a_kernel._flagdnn_ppu_bridge = True
    standalone.compile_a_kernel = compile_a_kernel
