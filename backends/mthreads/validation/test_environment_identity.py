from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import environment_identity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_record(path: Path, **extra: object) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "requested_path": str(path.absolute()),
        "realpath": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256(resolved),
        **extra,
    }


def _directory_record(path: Path) -> dict[str, object]:
    return {
        "requested_path": str(path.absolute()),
        "realpath": str(path.resolve(strict=True)),
    }


class EnvironmentContract(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.musa_root = self.root / "musa"
        self.jit_root = self.root / "jit"
        self.python_root = self.root / "python"
        for directory in (
            self.musa_root / "include",
            self.musa_root / "lib",
            self.musa_root / "bin",
            self.jit_root / "include" / "triton_jit",
            self.jit_root / "lib" / "cmake" / "TritonJIT",
            self.jit_root / "share" / "triton_jit" / "scripts",
            self.python_root / "bin",
            self.python_root / "modules",
            self.root / "system",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        paths = [
            self.musa_root / "include" / "musa_runtime_api.h",
            self.musa_root / "include" / "musa.h",
            self.musa_root / "include" / "mudnn.h",
            self.musa_root / "lib" / "libmusart.so",
            self.musa_root / "lib" / "libmudnn.so",
            self.musa_root / "bin" / "mcc",
            self.root / "system" / "libmusa.so",
            self.root / "system" / "patchelf",
            self.jit_root / "lib" / "libtriton_jit.so",
            self.jit_root
            / "lib"
            / "cmake"
            / "TritonJIT"
            / "TritonJITConfig.cmake",
            self.jit_root
            / "include"
            / "triton_jit"
            / "triton_jit_function.h",
            self.jit_root
            / "share"
            / "triton_jit"
            / "scripts"
            / "gen_ssig.py",
            self.jit_root
            / "share"
            / "triton_jit"
            / "scripts"
            / "standalone_compile.py",
            self.python_root / "bin" / "python",
        ]
        for index, path in enumerate(paths):
            path.write_bytes(f"fixture-{index}\n".encode("ascii"))

        self.module_paths: dict[str, Path] = {}
        for name in ("torch", "torch_musa", "triton", "yaml"):
            path = self.python_root / "modules" / f"{name}.py"
            path.write_text(f"__version__ = 'fixture-{name}'\n", encoding="utf-8")
            self.module_paths[name] = path

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def valid_document(self) -> dict[str, object]:
        resources = {
            "musa_root": _directory_record(self.musa_root),
            "musa_runtime_header": _file_record(
                self.musa_root / "include" / "musa_runtime_api.h"
            ),
            "musa_driver_header": _file_record(
                self.musa_root / "include" / "musa.h"
            ),
            "mudnn_header": _file_record(
                self.musa_root / "include" / "mudnn.h"
            ),
            "musa_runtime": _file_record(
                self.musa_root / "lib" / "libmusart.so"
            ),
            "musa_driver": _file_record(
                self.root / "system" / "libmusa.so"
            ),
            "mudnn": _file_record(self.musa_root / "lib" / "libmudnn.so"),
            "mcc": _file_record(self.musa_root / "bin" / "mcc"),
            "patchelf": _file_record(
                self.root / "system" / "patchelf",
                version="0.17.2",
                distribution_version="0.17.2.4",
            ),
            "triton_jit_prefix": _directory_record(self.jit_root),
            "triton_jit": _file_record(
                self.jit_root / "lib" / "libtriton_jit.so",
                backend="MUSA",
                soname="libtriton_jit.so",
                runpath=["$ORIGIN", str(self.musa_root / "lib")],
                dependencies=[
                    {
                        "name": "libmusa.so.1",
                        "resolved": str(
                            (self.root / "system" / "libmusa.so").resolve()
                        ),
                    },
                    {
                        "name": "libpython3.10.so.1.0",
                        "resolved": "/usr/lib/libpython3.10.so.1.0",
                    },
                    {
                        "name": "libtorch.so",
                        "resolved": "/fixture/libtorch.so",
                    },
                    {
                        "name": "libstdc++.so.6",
                        "resolved": "/usr/lib/libstdc++.so.6",
                    },
                ],
            ),
            "triton_jit_config": _file_record(
                self.jit_root
                / "lib"
                / "cmake"
                / "TritonJIT"
                / "TritonJITConfig.cmake"
            ),
            "triton_jit_header": _file_record(
                self.jit_root
                / "include"
                / "triton_jit"
                / "triton_jit_function.h"
            ),
            "triton_jit_gen_ssig": _file_record(
                self.jit_root
                / "share"
                / "triton_jit"
                / "scripts"
                / "gen_ssig.py"
            ),
            "triton_jit_standalone_compile": _file_record(
                self.jit_root
                / "share"
                / "triton_jit"
                / "scripts"
                / "standalone_compile.py"
            ),
        }
        document: dict[str, object] = {
            "schema_version": 2,
            "platform": "mthreads",
            "resources": resources,
            "python": {
                "executable": _file_record(
                    self.python_root / "bin" / "python"
                ),
                "prefix": str(self.python_root.resolve()),
                "version": "3.10.12",
                "modules": {
                    name: _file_record(path)
                    for name, path in self.module_paths.items()
                },
            },
            "errors": [],
            "identity_sha256": "",
        }
        document["identity_sha256"] = environment_identity.identity_sha256(
            document
        )
        return document

    def test_canonical_json_is_key_order_independent(self) -> None:
        self.assertEqual(
            environment_identity.canonical_json({"b": 1, "a": 2}),
            b'{"a":2,"b":1}',
        )

    def test_readelf_library_soname_and_runpath_are_parsed(self) -> None:
        dynamic_section = """
 0x000000000000000e (SONAME)             Library soname: [libtriton_jit.so]
 0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN:/musa/lib]
"""
        self.assertEqual(
            environment_identity._parse_elf_dynamic_section(dynamic_section),
            ("libtriton_jit.so", ["$ORIGIN", "/musa/lib"]),
        )

    def test_valid_fixture_has_no_errors(self) -> None:
        self.assertEqual(
            environment_identity.validate_environment(self.valid_document()), []
        )

    def test_validation_rejects_non_musa_jit(self) -> None:
        document = self.valid_document()
        document["resources"]["triton_jit"]["backend"] = "CUDA"
        self.assertIn(
            "TritonJIT backend must be MUSA",
            environment_identity.validate_environment(document),
        )

    def test_validation_rejects_mixed_musa_roots(self) -> None:
        document = self.valid_document()
        outside = self.root / "system" / "libmudnn.so"
        outside.write_bytes(b"outside-mudnn\n")
        document["resources"]["mudnn"] = _file_record(outside)
        self.assertIn(
            "MUSA and muDNN resources do not share the selected root",
            environment_identity.validate_environment(document),
        )

    def test_validation_rejects_missing_yaml(self) -> None:
        document = self.valid_document()
        document["python"]["modules"].pop("yaml")
        self.assertIn(
            "codegen Python cannot import yaml",
            environment_identity.validate_environment(document),
        )

    def test_validation_rejects_missing_patchelf_version(self) -> None:
        document = self.valid_document()
        document["resources"]["patchelf"].pop("version")
        self.assertIn(
            "patchelf version evidence is missing",
            environment_identity.validate_environment(document),
        )

    def test_validation_rejects_changed_resource_bytes(self) -> None:
        document = self.valid_document()
        header = self.musa_root / "include" / "mudnn.h"
        header.write_bytes(b"changed\n")
        self.assertIn(
            "resource mudnn_header SHA-256 does not match current bytes",
            environment_identity.validate_environment(document),
        )

    def test_validation_rejects_identity_mismatch(self) -> None:
        document = self.valid_document()
        document["platform"] = "changed"
        self.assertIn(
            "environment identity SHA-256 mismatch",
            environment_identity.validate_environment(document),
        )

    def test_validation_rejects_unresolved_jit_dependency(self) -> None:
        document = self.valid_document()
        document["resources"]["triton_jit"]["dependencies"].append(
            {"name": "libmissing.so", "resolved": None}
        )
        self.assertIn(
            "TritonJIT dependency not found: libmissing.so",
            environment_identity.validate_environment(document),
        )

    def test_validation_rejects_unexpected_top_level_key(self) -> None:
        document = self.valid_document()
        document["extra"] = True
        self.assertIn(
            "unexpected top-level keys: extra",
            environment_identity.validate_environment(document),
        )

    def test_symlink_request_resolves_to_recorded_canonical_file(self) -> None:
        document = self.valid_document()
        original_identity = document["identity_sha256"]
        target = self.musa_root / "include" / "mudnn.h"
        link = self.musa_root / "include" / "mudnn-current.h"
        link.symlink_to(target.name)
        document["resources"]["mudnn_header"] = _file_record(link)
        document["identity_sha256"] = environment_identity.identity_sha256(
            document
        )
        self.assertEqual(document["identity_sha256"], original_identity)
        self.assertEqual(environment_identity.validate_environment(document), [])

    def test_errors_are_sorted_and_deduplicated(self) -> None:
        document = self.valid_document()
        document["resources"]["triton_jit"]["backend"] = "CUDA"
        document["python"]["modules"].pop("yaml")
        errors = environment_identity.validate_environment(document)
        self.assertEqual(errors, sorted(set(errors)))

    def test_write_atomic_replaces_complete_payload(self) -> None:
        destination = self.root / "reports" / "environment.json"
        environment_identity.write_atomic(destination, b'{"generation":1}\n')
        environment_identity.write_atomic(destination, b'{"generation":2}\n')
        self.assertEqual(destination.read_bytes(), b'{"generation":2}\n')
        self.assertEqual(list(destination.parent.glob("*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
