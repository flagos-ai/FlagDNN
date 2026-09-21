#!/usr/bin/env python3
"""Host contracts for compiler dependency hashing and invalidation."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "compiler"))
from flagdnn_codegen import main as compiler


class DependencySnapshotContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "dependency.bin"
        self.path.write_bytes(b"before")

    def snapshot(self, path=None):
        return compiler._dependency_snapshots([str(path or self.path)])[0]

    def assert_content(self, snapshot, content):
        self.assertEqual(snapshot["content_sha256"], hashlib.sha256(content).hexdigest())

    def test_unchanged_contents_are_read_once(self):
        time.sleep(2.05)
        original_open = builtins.open
        reads = []

        def record_open(path, *args, **kwargs):
            if Path(path) == self.path and args and args[0] == "rb":
                reads.append(path)
            return original_open(path, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=record_open):
            first = self.snapshot()
            self.assertEqual(first, self.snapshot())
            self.assertEqual(first, self.snapshot())
        self.assert_content(first, b"before")
        self.assertEqual(len(reads), 1, "unchanged contents were re-read")

    def test_same_size_write_with_restored_mtime_invalidates(self):
        first = self.snapshot()
        previous = self.path.stat()
        self.path.write_bytes(b"after!")
        os.utime(self.path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        second = self.snapshot()
        self.assertNotEqual(first["content_sha256"], second["content_sha256"])
        self.assert_content(second, b"after!")

    def test_recent_file_does_not_reuse_digest_when_stat_tick_is_unchanged(self):
        # Some filesystems report identical ctime for consecutive fast writes.
        fingerprint = compiler._dependency_fingerprint(str(self.path))
        with mock.patch.object(compiler, "_dependency_fingerprint", return_value=fingerprint):
            self.assert_content(self.snapshot(), b"before")
            self.path.write_bytes(b"after!")
            self.assert_content(self.snapshot(), b"after!")

    def test_future_timestamp_does_not_reuse_digest(self):
        future = time.time_ns() + 10_000_000_000
        os.utime(self.path, ns=(future, future))
        with mock.patch.object(compiler, "_dependency_fingerprint", return_value="0" * 64):
            self.assert_content(self.snapshot(), b"before")
            self.path.write_bytes(b"after!")
            os.utime(self.path, ns=(future, future))
            self.assert_content(self.snapshot(), b"after!")

    def test_replacement_with_restored_mtime_invalidates(self):
        self.snapshot()
        previous = self.path.stat()
        replacement = self.root / "replacement"
        replacement.write_bytes(b"after!")
        os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        replacement.replace(self.path)
        self.assert_content(self.snapshot(), b"after!")

    def test_symlink_retarget_invalidates(self):
        link = self.root / "link"
        link.symlink_to(self.path)
        first = self.snapshot(link)
        target = self.root / "other"
        target.write_bytes(b"after!")
        link.unlink()
        link.symlink_to(target)
        second = self.snapshot(link)
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])
        self.assert_content(second, b"after!")

    def test_deletion_and_recreation_invalidates(self):
        self.snapshot()
        self.path.unlink()
        self.assertNotIn("content_sha256", self.snapshot())
        self.path.write_bytes(b"after!")
        self.assert_content(self.snapshot(), b"after!")

    def test_directory_does_not_reuse_a_file_digest(self):
        self.snapshot()
        self.path.unlink()
        self.path.mkdir()
        self.assertNotIn("content_sha256", self.snapshot())

    def test_mutation_during_read_is_rejected_and_not_cached(self):
        original_open = builtins.open
        changed = False

        def changing_open(path, *args, **kwargs):
            nonlocal changed
            source = original_open(path, *args, **kwargs)
            if Path(path) != self.path or not args or args[0] != "rb":
                return source
            wrapped = mock.MagicMock(wraps=source)
            wrapped.__enter__.return_value = wrapped
            wrapped.__exit__.side_effect = lambda *args: source.close()

            def read(size=-1):
                nonlocal changed
                data = source.read(size)
                if data and not changed:
                    changed = True
                    time.sleep(0.01)
                    self.path.write_bytes(b"after!")
                return data

            wrapped.read.side_effect = read
            return wrapped

        with mock.patch("builtins.open", side_effect=changing_open):
            with self.assertRaises(compiler._IdentitySnapshotChanged):
                self.snapshot()
        self.assert_content(self.snapshot(), b"after!")

    def test_provider_mutation_with_restored_mtime_is_rejected(self):
        previous = self.path.stat()

        def identity(*args):
            self.path.write_bytes(b"after!")
            os.utime(self.path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
            return {"identity_sha256": "a" * 64}

        provider = SimpleNamespace(
            __file__=str(self.path),
            compiler_identity_dependencies=lambda *args: [self.path],
            compiler_identity=identity,
        )
        with self.assertRaises(compiler._IdentitySnapshotChanged):
            compiler._stable_compiler_identity(provider, "target", "libtriton_jit")

    def test_compile_rechecks_changed_contents_and_dependency_set(self):
        for change in ("unchanged", "restored_mtime", "replacement", "symlink", "added"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                dependency = root / "dependency"
                dependency.write_bytes(b"before")
                link = root / "link"
                link.symlink_to(dependency)
                paths = [link if change == "symlink" else dependency]
                provider_file = root / "provider.py"
                provider_file.write_text("# fixture provider\n")
                request = root / "request.json"
                request.write_text(json.dumps({
                    "backend": "fixture", "target": "target",
                    "compiler_identity": "a" * 64,
                }))

                def compile_request(*args):
                    previous = dependency.stat()
                    time.sleep(0.01)
                    if change == "restored_mtime":
                        dependency.write_bytes(b"after!")
                        os.utime(dependency, ns=(previous.st_atime_ns, previous.st_mtime_ns))
                    elif change in {"replacement", "symlink", "added"}:
                        other = root / "other"
                        other.write_bytes(b"after!")
                        if change == "replacement":
                            os.utime(other, ns=(previous.st_atime_ns, previous.st_mtime_ns))
                            other.replace(dependency)
                        elif change == "symlink":
                            link.unlink()
                            link.symlink_to(other)
                        else:
                            paths.append(other)
                    return {}

                provider = SimpleNamespace(
                    __file__=str(provider_file),
                    compiler_identity_dependencies=lambda *args: list(paths),
                    compiler_identity=lambda *args: {"identity_sha256": "a" * 64},
                    compile_request=compile_request,
                )
                argv = ["compiler", "--request", str(request), "--output-dir",
                        str(root / "output"), "--execution-engine", "libtriton_jit", "--quiet"]
                with mock.patch.object(compiler, "get_provider", return_value=provider), \
                        mock.patch.object(sys, "argv", argv):
                    self.assertEqual(compiler.main(), 0 if change == "unchanged" else 75)

    def test_stable_provider_identity_keeps_original_digest_and_snapshots(self):
        provider = SimpleNamespace(
            __file__=str(self.path),
            compiler_identity_dependencies=lambda *args: [self.path],
            compiler_identity=lambda *args: {"identity_sha256": "a" * 64},
        )
        identity, paths, snapshots, complete = compiler._stable_compiler_identity(
            provider, "target", "libtriton_jit"
        )
        self.assertEqual(identity, {"identity_sha256": "a" * 64})
        self.assertTrue(complete)
        self.assertEqual(paths, [item["path"] for item in snapshots])
        for item in snapshots:
            self.assertEqual(item["fingerprint"], compiler._dependency_fingerprint(item["path"]))
            self.assert_content(item, Path(item["path"]).read_bytes())


if __name__ == "__main__":
    unittest.main()
