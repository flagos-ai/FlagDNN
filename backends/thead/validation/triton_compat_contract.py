#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Regression contracts for vendor/FlagTree identity and PPU JIT loading."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triton_compat import install_cuda_jit_bridge, ppu_codegen_backend
from TritonIdentityContract import IdentityError, _jit_backend, identify
from preflight_environment import _probe_triton_jit


class CompatibilityContract(unittest.TestCase):
    def metadata(self, root, name, version):
        info = root / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
        (info / "RECORD").write_text(f"{info.name}/METADATA,,\n")

    def test_vendor_and_flagtree_identity(self):
        for name, backend, version in (
            ("triton", "nvidia", "3.5.0+ppu"),
            ("flagtree", "ppu", "0.6.0+ppu"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.metadata(root, name, version)
                self.assertEqual(ppu_codegen_backend(root), backend)
                other = "flagtree" if name == "triton" else "triton"
                self.metadata(root, other, version)
                with self.assertRaisesRegex(RuntimeError, "exactly one"):
                    ppu_codegen_backend(root)

    def test_identity_rejects_foreign_package_before_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "selected"
            root.mkdir()
            foreign = Path(tmp) / "foreign.py"
            foreign.write_text("raise AssertionError('must not execute')\n")
            with mock.patch.object(sys, "path", [*sys.path, str(root)]), \
                    mock.patch("importlib.util.find_spec", return_value=SimpleNamespace(origin=str(foreign))), \
                    mock.patch("importlib.import_module") as import_module:
                with self.assertRaisesRegex(IdentityError, "outside configured root"):
                    identify(root, root)
                self.assertEqual(sys.path.count(str(root)), 1)
                import_module.assert_not_called()

    def test_identity_rejects_missing_package_before_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(sys, "path", list(sys.path)), \
                    mock.patch("importlib.util.find_spec", return_value=None), \
                    mock.patch("importlib.import_module") as import_module:
                with self.assertRaisesRegex(IdentityError, "has no Triton package"):
                    identify(root, root)
                import_module.assert_not_called()

    def test_selected_package_does_not_shadow_standard_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "selected"
            (root / "triton").mkdir(parents=True)
            (root / "triton/__init__.py").write_text(
                "from dataclasses import dataclass\n"
                "@dataclass\n"
                "class Options:\n    value: int = 1\n"
            )
            (root / "dataclasses.py").write_text(
                "raise AssertionError('loaded obsolete dataclasses backport')\n"
            )
            (root / "_csv.py").write_text(
                "raise AssertionError('shadowed a stdlib extension')\n"
            )
            other = Path(tmp) / "other"
            other.mkdir()
            (other / "triton.py").write_text(
                "raise AssertionError('loaded a different Triton installation')\n"
            )
            probe = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from triton_compat import configure_triton_path
root, other, placement = sys.argv[2:]
sys.path.append(other)
if placement == "first":
    sys.path.insert(0, root)
elif placement == "last":
    sys.path.append(root)
configure_triton_path(root)
configure_triton_path(root)
import triton, dataclasses, _csv
assert Path(triton.__file__).parent.parent == Path(root)
assert not Path(dataclasses.__file__).is_relative_to(root)
assert not Path(_csv.__file__).is_relative_to(root)
assert triton.Options().value == 1
assert sys.path.count(root) == 1
"""
            for placement in ("absent", "first", "last"):
                with self.subTest(placement=placement):
                    result = subprocess.run(
                        [sys.executable, "-B", "-c", probe,
                         str(Path(__file__).resolve().parents[1]),
                         str(root), str(other), placement],
                        capture_output=True, text=True,
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                        timeout=30,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )

    def test_jit_identity_build_and_install_layouts(self):
        for relative in (
            "build/TritonJITConfig.cmake",
            "lib/cmake/TritonJIT/TritonJITConfig.cmake",
            "lib64/cmake/TritonJIT/TritonJITConfig.cmake",
        ):
            with self.subTest(layout=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = root / relative
                config.parent.mkdir(parents=True)
                config.write_text('set(TritonJIT_BACKEND "CUDA")\n')
                self.assertEqual(_jit_backend(root), (config.resolve(), "CUDA"))
                config.write_text('set(TritonJIT_BACKEND "HCU")\n')
                with self.assertRaisesRegex(IdentityError, "not exactly CUDA"):
                    _jit_backend(root)

    def test_installed_jit_preflight_uses_its_own_content(self):
        for libdir in ("lib", "lib64"):
            with self.subTest(libdir=libdir), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp)
                (parent / ".git").mkdir()
                root = parent / "sdk"
                config = root / libdir / "cmake/TritonJIT/TritonJITConfig.cmake"
                config.parent.mkdir(parents=True)
                config.write_text('set(TritonJIT_BACKEND "CUDA")\n')
                library = root / libdir / "libtriton_jit.so"
                library.write_bytes(b"library-v1")
                scripts = root / "share/triton_jit/scripts"
                scripts.mkdir(parents=True)
                for name in ("standalone_compile.py", "gen_ssig.py"):
                    (scripts / name).write_text("# fixture\n")
                first = _probe_triton_jit(root)
                self.assertEqual(first["repository_root"], str(root))
                self.assertEqual(first["source_identity"]["kind"], "content_sha256")
                self.assertEqual(first["library"], str(library))
                library.write_bytes(b"library-v2")
                self.assertNotEqual(first["source_identity"], _probe_triton_jit(root)["source_identity"])

    def test_jit_identity_rejects_missing_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(IdentityError, "build/install"):
                _jit_backend(Path(tmp))

    def test_jit_identity_rejects_config_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "jit"
            (root / "build").mkdir(parents=True)
            config = parent / "outside.cmake"
            config.write_text('set(TritonJIT_BACKEND "CUDA")\n')
            (root / "build/TritonJITConfig.cmake").symlink_to(config)
            with self.assertRaisesRegex(IdentityError, "outside configured root"):
                _jit_backend(root)

    def test_non_ppu_distribution_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.metadata(root, "triton", "3.6.0")
            with self.assertRaisesRegex(RuntimeError, "PPU-qualified"):
                ppu_codegen_backend(root)

    def test_loader_alias_is_exact_and_refreshes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = SimpleNamespace(
                compile_a_kernel=lambda *args, **kwargs: tmp
            )
            install_cuda_jit_bridge(module)
            wrapper = module.compile_a_kernel
            install_cuda_jit_bridge(module)
            self.assertIs(module.compile_a_kernel, wrapper)
            for content in (b"\x7fELFfirst", b"\x7fELFsecond"):
                (root / "kernel.hgbin").write_bytes(content)
                self.assertEqual(wrapper("source.py", "kernel"), tmp)
                self.assertEqual((root / "kernel.cubin").read_bytes(), content)
            (root / "kernel.hgbin").write_bytes(b"invalid")
            with self.assertRaisesRegex(RuntimeError, "not ELF"):
                wrapper("source.py", "kernel")
            (root / "kernel.hgbin").unlink()
            with self.assertRaisesRegex(RuntimeError, "regular hgbin"):
                wrapper("source.py", "kernel")

    def test_flagtree_symlink_is_materialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b"\x7fELFbinary"
            (root / "kernel.hgbin").write_bytes(content)
            destination = root / "kernel.cubin"
            destination.symlink_to("kernel.hgbin")
            module = SimpleNamespace(compile_a_kernel=lambda *args: tmp)
            install_cuda_jit_bridge(module)
            module.compile_a_kernel("source.py", "kernel")
            self.assertFalse(destination.is_symlink())
            self.assertEqual(destination.read_bytes(), content)

    def test_loader_alias_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "kernel.hgbin"
            binary.write_bytes(b"\x7fELFbinary")
            other = root / "other.hgbin"
            other.write_bytes(b"\x7fELFother")
            (root / "kernel.cubin").symlink_to(other)
            module = SimpleNamespace(compile_a_kernel=lambda *args: tmp)
            install_cuda_jit_bridge(module)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                module.compile_a_kernel("source.py", "kernel")


if __name__ == "__main__":
    unittest.main()
