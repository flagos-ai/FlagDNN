# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the private MTGPU helper without importing a GPU runtime."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


PATCHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "cmake"
    / "patch_standalone_compile.py"
)
SPEC = importlib.util.spec_from_file_location(
    "mthreads_helper_patcher", PATCHER_PATH
)
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

# Keep the actual upstream import/fallback and legacy exception-handler shape.
# Mock compilation so the test requires neither Triton nor MUSA.
UPSTREAM = r"""from pathlib import Path
from typing import List
import triton


def _bracket_aware_split(sig: str) -> List[str]:
    return [token.strip() for token in sig.split(",")]


def _parse_type_token(token: str):
    return token.strip()


def _compile_a_kernel(fn, signature):
    backend = get_backend()
    ccinfo = triton.compile(fn, options={"num_warps": 4})
    cache_dir = ccinfo.cache_dir
    if backend == "NPU":
        pass
    elif backend == "MTGPU":
        import shutil

        try:
            from triton._C.libtriton import mtgpu
        except ImportError:
            from triton._C.libtriton import mthreads as mtgpu

        kernel_name = fn.__name__
        llir_path = Path(cache_dir) / f"{kernel_name}.llir"
        if llir_path.exists():
            try:
                asm_str, mubin_tmp_path = mtgpu.translate_llvmir_to_mubin(
                    llir_path.read_text(), "legacy-options", 22, 0
                )
                mubin_path = Path(cache_dir) / f"{kernel_name}.mubin"
                shutil.copy2(mubin_tmp_path, mubin_path)
            except Exception as e:
                import sys
                import traceback

                sys.stderr.write(
                    f"[MTGPU] Compilation failed: {e}\n"
                    f"{traceback.format_exc()}\n"
                )
    return cache_dir
"""


class StandaloneCompilePatchContract(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name)
        self.stderr = io.StringIO()
        self.fn = SimpleNamespace(__name__="kernel")

    def write_artifacts(self, files: dict[str, bytes]) -> None:
        for path in self.cache.glob("kernel.*"):
            path.unlink()
        for suffix, content in files.items():
            (self.cache / f"kernel.{suffix}").write_bytes(content)

    def compile(
        self,
        asm: dict,
        *,
        translator=None,
        backend="MTGPU",
        fallback_import=False,
        compile_error=None,
        patched=True,
    ):
        triton = ModuleType("triton")
        triton.compile = mock.Mock(
            return_value=SimpleNamespace(asm=asm, cache_dir=str(self.cache)),
            side_effect=compile_error,
        )
        bindings = ModuleType("triton._C.libtriton")
        mtgpu = SimpleNamespace()
        if translator is not None:
            mtgpu.translate_llvmir_to_mubin = translator
        setattr(bindings, "mthreads" if fallback_import else "mtgpu", mtgpu)
        modules = {
            "triton": triton,
            "triton._C": ModuleType("triton._C"),
            "triton._C.libtriton": bindings,
        }
        namespace = {"get_backend": lambda: backend}
        source = PATCHER.patch_source(UPSTREAM) if patched else UPSTREAM
        with mock.patch.dict(sys.modules, modules), contextlib.redirect_stderr(
            self.stderr
        ):
            exec(
                compile(source, "private_standalone_compile.py", "exec"),
                namespace,
            )
            result = namespace["_compile_a_kernel"](self.fn, "*fp32")
        triton.compile.assert_called_once_with(
            self.fn, options={"num_warps": 4}
        )
        return result

    def test_current_backend_preserves_matching_compiled_artifacts(
        self,
    ) -> None:
        for suffix in ("mubin", "o", "so", "llir"):
            for value in (
                b"\x00\xffcompiled",
                bytearray(b"compiled"),
                "IR \u03bb\n",
            ):
                with self.subTest(
                    suffix=suffix, value_type=type(value).__name__
                ):
                    data = (
                        value.encode("utf-8")
                        if isinstance(value, str)
                        else bytes(value)
                    )
                    self.write_artifacts({suffix: data})
                    self.assertEqual(
                        self.compile({suffix: value}), str(self.cache)
                    )
                    self.assertEqual(
                        (self.cache / f"kernel.{suffix}").read_bytes(), data
                    )
                    self.assertEqual(self.stderr.getvalue(), "")

    def test_fallback_module_and_noncallable_legacy_attribute(self) -> None:
        self.write_artifacts({"mubin": b"compiled"})
        self.assertEqual(
            self.compile({"mubin": b"compiled"}, fallback_import=True),
            str(self.cache),
        )
        self.assertEqual(
            self.compile({"mubin": b"compiled"}, translator="not callable"),
            str(self.cache),
        )
        self.assertEqual(self.stderr.getvalue(), "")

    def test_legacy_callable_still_runs_original_translation(self) -> None:
        self.write_artifacts({"llir": b"legacy IR"})
        generated = self.cache / "legacy-output.mubin"
        generated.write_bytes(b"legacy binary")
        translator = mock.Mock(return_value=("assembly", str(generated)))
        self.assertEqual(
            self.compile({}, translator=translator), str(self.cache)
        )
        translator.assert_called_once_with(
            "legacy IR", "legacy-options", 22, 0
        )
        self.assertEqual(
            (self.cache / "kernel.mubin").read_bytes(), b"legacy binary"
        )
        self.assertEqual(self.stderr.getvalue(), "")

    def test_missing_empty_or_unverifiable_artifacts_fail(self) -> None:
        cases = (
            ("missing", {}, {"mubin": b"compiled"}),
            ("empty", {"mubin": b""}, {"mubin": b""}),
            ("mismatch", {"mubin": b"stale"}, {"mubin": b"compiled"}),
            ("no asm entry", {"mubin": b"compiled"}, {}),
            (
                "unsupported asm type",
                {"mubin": b"compiled"},
                {"mubin": object()},
            ),
            ("wrong suffix", {"llir": b"compiled"}, {"mubin": b"compiled"}),
        )
        for name, files, asm in cases:
            with self.subTest(name=name):
                self.write_artifacts(files)
                with self.assertRaises(RuntimeError) as error:
                    self.compile(asm)
                self.assertTrue(str(error.exception))
                self.assertEqual(self.stderr.getvalue(), "")
                for suffix, data in files.items():
                    self.assertEqual(
                        (self.cache / f"kernel.{suffix}").read_bytes(), data
                    )

    def test_first_existing_artifact_cannot_be_bypassed(self) -> None:
        suffixes = ("mubin", "o", "so", "llir")
        for preferred, fallback in zip(suffixes, suffixes[1:]):
            for stale in (b"stale", b""):
                with self.subTest(preferred=preferred, stale=stale):
                    files = {preferred: stale, fallback: b"compiled"}
                    self.write_artifacts(files)
                    with self.assertRaises(RuntimeError):
                        self.compile(
                            {preferred: b"compiled", fallback: b"compiled"}
                        )
                    for suffix, data in files.items():
                        self.assertEqual(
                            (self.cache / f"kernel.{suffix}").read_bytes(),
                            data,
                        )
        self.assertEqual(self.stderr.getvalue(), "")

    def test_real_compile_failure_is_not_swallowed(self) -> None:
        failure = ValueError("actual Triton compiler failure")
        self.write_artifacts({"mubin": b"compiled"})
        with self.assertRaises(ValueError) as raised:
            self.compile({"mubin": b"compiled"}, compile_error=failure)
        self.assertIs(raised.exception, failure)
        self.assertEqual(self.stderr.getvalue(), "")

    def test_other_backend_paths_are_unchanged(self) -> None:
        for backend in (
            "CUDA",
            "HCU",
            "MUSA",
            "NPU",
            "GCU",
            "MLU",
            "MACA",
            "IX",
        ):
            with self.subTest(backend=backend):
                self.assertEqual(
                    self.compile({}, backend=backend),
                    self.compile({}, backend=backend, patched=False),
                )
        self.assertEqual(self.stderr.getvalue(), "")

    def test_splitter_preserves_descriptor_and_tuple_groups(self) -> None:
        namespace = {}
        with mock.patch.dict(sys.modules, {"triton": ModuleType("triton")}):
            exec(PATCHER.patch_source(UPSTREAM), namespace)
        split = namespace["_bracket_aware_split"]
        self.assertEqual(
            split("tensordesc<fp32[16,32]>, (*fp32, i32), i64"),
            ["tensordesc<fp32[16,32]>", "(*fp32, i32)", "i64"],
        )
        for malformed in (
            "tensordesc<fp32[16,32]",
            "(*fp32, i32]",
            "i32,,i64",
        ):
            with self.subTest(signature=malformed), self.assertRaises(
                ValueError
            ):
                split(malformed)

    def test_cli_writes_private_copy_and_preserves_input(self) -> None:
        original = self.cache / "upstream.py"
        output = self.cache / "private" / "standalone_compile.py"
        original.write_text(UPSTREAM, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(PATCHER_PATH), str(original), str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(original.read_bytes(), UPSTREAM.encode("utf-8"))
        self.assertEqual(
            output.read_text(encoding="utf-8"), PATCHER.patch_source(UPSTREAM)
        )
        overwrite = subprocess.run(
            [sys.executable, str(PATCHER_PATH), str(original), str(original)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(overwrite.returncode, 0)
        self.assertEqual(original.read_bytes(), UPSTREAM.encode("utf-8"))

    def test_unknown_source_anchors_are_rejected(self) -> None:
        for source in (
            UPSTREAM.replace(
                "def _bracket_aware_split(", "def unknown_split("
            ),
            UPSTREAM.replace("def _parse_type_token(", "def unknown_parse("),
            UPSTREAM.replace(
                "from triton._C.libtriton import mtgpu", "import unknown_mtgpu"
            ),
        ):
            with self.subTest(source=source), self.assertRaises(RuntimeError):
                PATCHER.patch_source(source)


if __name__ == "__main__":
    unittest.main()
