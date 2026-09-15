"""Negative/positive host fixtures for the native NVIDIA artifact parser."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROBE = sys.argv.pop(1)
VERSION = sys.argv.pop(1)


class ArtifactContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix="flagdnn-artifact-contract-"
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "jit_cache_0_default"
        self.cache.mkdir()
        self.source = self.root / "stage.py"
        self.source.write_text("def kernel(): pass\n")
        self.metadata = self.cache / "kernel.json"
        self.metadata.write_text(
            json.dumps(
                dict(
                    name="kernel",
                    num_warps=8,
                    num_stages=3,
                    num_ctas=1,
                    global_scratch_size=0,
                    global_scratch_align=1,
                    profile_scratch_size=0,
                    tensordesc_meta=[
                        dict(
                            swizzle=3,
                            elem_size=4,
                            elem_type=7,
                            block_size=[256, 32],
                            fp4_padded=False,
                        )
                    ],
                )
            )
        )
        self.binary = self.cache / "kernel.cubin"
        self.binary.write_bytes(b"parser-only-fixture-not-loaded")
        identity = "1" * 64
        request = dict(
            schema_version=3,
            flagdnn_version=VERSION,
            backend="nvidia",
            target="sm_90",
            compiler_identity=identity,
        )
        request_bytes = json.dumps(request).encode()
        (self.root / "request.json").write_bytes(request_bytes)
        self.variant = dict(
            variant_id="default",
            source_sha256=identity,
            full_signature="tensordesc<fp32[256,32]>,*fp32:16",
            compile_options=dict(num_warps=8, num_stages=3),
            compiled_cache=dict(
                directory=self.cache.name,
                name="kernel",
                metadata=self.descriptor(self.metadata),
                binary=self.descriptor(self.binary),
            ),
            argument_abi=[
                dict(
                    kind="tensor_map",
                    uid=1,
                    size=4096 * 1024 * 4,
                    alignment=16,
                    data_type="tf32_rne",
                    shape=[4096, 1024],
                    strides=[1024, 1],
                    block_shape=[256, 32],
                ),
                dict(kind="tensor", uid=2, size=1024, alignment=16),
                dict(kind="global_scratch_pointer"),
                dict(kind="profile_scratch_pointer"),
            ],
            launch=dict(
                grid=[132, 1, 1],
                block=[256, 1, 1],
                cluster=[1, 1, 1],
                shared_memory=0,
                num_ctas=1,
                global_scratch_size=4096,
                profile_scratch_size=0,
            ),
        )
        self.stage = dict(
            stage_id=0,
            kind="kernel",
            source_node_ids=[0],
            dependencies=[],
            engine="libtriton_jit",
            kernel=dict(
                function="kernel",
                materialized_source=self.descriptor(self.source),
            ),
        )
        self.manifest = dict(
            schema_version=4,
            artifact_kind="flagdnn_execution_program",
            flagdnn_version=VERSION,
            backend="nvidia",
            target="sm_90",
            request_sha256=hashlib.sha256(request_bytes).hexdigest(),
            source_sha256=identity,
            compiler=dict(
                identity_sha256=identity,
                provider="nvidia",
                triton_version="fixture",
            ),
            workspace_size=4096,
            graph_node_count=1,
            program=dict(schema_version=2, stage_count=1, stages=[]),
        )

    @staticmethod
    def descriptor(path):
        payload = path.read_bytes()
        return dict(
            file=path.name,
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def check(self, valid, contains="", variants=None):
        if variants is None:
            stage = {**self.stage, **self.variant}
        else:
            stage = {
                **self.stage,
                "variants": variants,
                "tuning": dict(
                    schema_version=1,
                    warmup=1,
                    repetitions=1,
                    candidate_identity="2" * 64,
                    source_sha256="3" * 64,
                    key="matmul",
                    strategy="matmul",
                ),
            }
        self.manifest["program"]["stages"] = [stage]
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        result = subprocess.run(
            [PROBE, str(self.root)], capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode == 0, valid, result.stderr)
        self.assertIn(contains, result.stdout + result.stderr)

    def test_valid_cached_tensor_map(self):
        self.check(True)

    def test_cached_pointer_kernel_and_descriptor_count(self):
        self.variant["argument_abi"][0] = dict(
            kind="tensor", uid=1, size=4096 * 1024 * 4, alignment=16
        )
        self.variant["full_signature"] = "*fp32:16,*fp32:16"
        metadata = json.loads(self.metadata.read_text())
        metadata["tensordesc_meta"] = []
        self.metadata.write_text(json.dumps(metadata))
        self.variant["compiled_cache"]["metadata"] = self.descriptor(
            self.metadata
        )
        self.check(True)
        metadata["tensordesc_meta"] = [{}]
        self.metadata.write_text(json.dumps(metadata))
        self.variant["compiled_cache"]["metadata"] = self.descriptor(
            self.metadata
        )
        self.check(False, "TensorMap count")

    def test_variants_cannot_mix_raw_and_cached_execution(self):
        self.variant["argument_abi"][0] = dict(
            kind="tensor", uid=1, size=4096 * 1024 * 4, alignment=16
        )
        self.variant["full_signature"] = "*fp32:16,*fp32:16"
        metadata = json.loads(self.metadata.read_text())
        metadata["tensordesc_meta"] = []
        self.metadata.write_text(json.dumps(metadata))
        self.variant["compiled_cache"]["metadata"] = self.descriptor(
            self.metadata
        )
        raw = copy.deepcopy(self.variant)
        raw["variant_id"] = "raw"
        del raw["compiled_cache"]
        self.check(False, "JIT cache modes", [self.variant, raw])
        self.check(False, "JIT cache modes", [raw, self.variant])

    def test_cached_specialization_uses_compiled_total_warps(self):
        self.variant["compile_options"]["num_warps"] = 12
        self.variant["launch"]["block"] = [384, 1, 1]
        metadata = json.loads(self.metadata.read_text())
        metadata["num_warps"] = 12
        self.metadata.write_text(json.dumps(metadata))
        self.variant["compiled_cache"]["metadata"] = self.descriptor(
            self.metadata
        )
        self.check(True)
        self.variant["launch"]["block"] = [128, 1, 1]
        self.check(False, "launch")

    def test_variant_local_tiles(self):
        alternate = copy.deepcopy(self.variant)
        alternate["variant_id"] = "other"
        alternate["argument_abi"][0]["block_shape"] = [128, 32]
        alternate_dir = self.root / "jit_cache_0_other"
        alternate_dir.mkdir()
        alternate_metadata = json.loads(self.metadata.read_text())
        alternate_metadata["tensordesc_meta"][0]["block_size"] = [128, 32]
        (alternate_dir / "kernel.json").write_text(
            json.dumps(alternate_metadata)
        )
        (alternate_dir / "kernel.cubin").write_bytes(self.binary.read_bytes())
        alternate["compiled_cache"] = dict(
            directory=alternate_dir.name,
            name="kernel",
            metadata=self.descriptor(alternate_dir / "kernel.json"),
            binary=self.descriptor(alternate_dir / "kernel.cubin"),
        )
        self.check(True, variants=[self.variant, alternate])
        alternate["argument_abi"][0]["shape"] = [2048, 2048]
        alternate["argument_abi"][0]["strides"] = [2048, 1]
        self.check(
            False, "incompatible argument ABIs", [self.variant, alternate]
        )

    def test_bad_map_metadata(self):
        original = copy.deepcopy(self.variant["argument_abi"][0])
        for changes in (
            dict(shape=[0, 1024]),
            dict(shape=[2**31, 1024]),
            dict(shape=[4097, 1024]),
            dict(block_shape=[128, 24]),
            dict(block_shape=[512, 32]),
            dict(strides=[1, 4096]),
            dict(data_type="float16"),
            dict(size=4),
            dict(alignment=8),
            dict(shape=[4096, 1024, 1]),
        ):
            with self.subTest(changes=changes):
                self.variant["argument_abi"][0] = {**original, **changes}
                self.check(False)

    def test_descriptor_requires_cache(self):
        del self.variant["compiled_cache"]
        self.check(False, "require an artifact-owned")

    def test_descriptor_capacity(self):
        self.variant["argument_abi"] = [
            self.variant["argument_abi"][0]
        ] * 5 + self.variant["argument_abi"][-2:]
        self.check(False, "TensorMap ABI")

    def test_expanded_argument_capacity(self):
        abi = self.variant["argument_abi"]
        self.variant["argument_abi"] = [abi[0]] + [abi[1]] * 4095 + abi[-2:]
        self.check(False, "TensorMap ABI")

    def test_corrupt_binary(self):
        self.binary.write_bytes(b"x" * self.binary.stat().st_size)
        self.check(False, "SHA-256")

    def test_metadata_option_mismatch(self):
        self.metadata.write_text(
            json.dumps(
                dict(name="kernel", num_warps=4, num_stages=3, num_ctas=1)
            )
        )
        self.variant["compiled_cache"]["metadata"] = self.descriptor(
            self.metadata
        )
        self.check(False, "options disagree")

    def test_compiled_descriptor_mismatch(self):
        original = json.loads(self.metadata.read_text())
        for changes in (
            dict(block_size=[128, 32]),
            dict(swizzle=2),
            dict(elem_type=11),
            dict(elem_size=2),
            dict(fp4_padded=True),
        ):
            with self.subTest(changes=changes):
                metadata = copy.deepcopy(original)
                metadata["tensordesc_meta"][0].update(changes)
                self.metadata.write_text(json.dumps(metadata))
                self.variant["compiled_cache"]["metadata"] = self.descriptor(
                    self.metadata
                )
                self.check(False, "TensorMap")

    def test_compiled_scratch_mismatch(self):
        metadata = json.loads(self.metadata.read_text())
        metadata["global_scratch_size"] = 4096
        self.metadata.write_text(json.dumps(metadata))
        self.variant["compiled_cache"]["metadata"] = self.descriptor(
            self.metadata
        )
        self.check(False, "scratch")

    def test_cache_path_escape(self):
        self.variant["compiled_cache"]["directory"] = "../escape"
        self.check(False, "path or kernel name")

    def test_cache_directory_symlink(self):
        link = self.root / "linked_cache"
        link.symlink_to(self.cache, target_is_directory=True)
        self.variant["compiled_cache"]["directory"] = link.name
        self.check(False, "non-symlink directory")

    def test_binary_symlink(self):
        link = self.cache / "link.cubin"
        self.binary.rename(link)
        self.binary.symlink_to(link)
        self.check(False, "not a regular file")


if __name__ == "__main__":
    unittest.main()
