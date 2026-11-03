"""Host regressions for Ascend launch ABI and unified runner integration."""

from pathlib import Path
import ast
import importlib
import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "compiler"))
from flagdnn_codegen.provider_loader import get_provider  # noqa: E402

provider = get_provider("ascend")
package = provider.__package__
emitter = importlib.import_module(package + ".codegen.extended")
attention = importlib.import_module(package + ".dispatch.attention_forward")
spec = importlib.util.spec_from_file_location(
    "ascend_runner_adapter",
    ROOT / "backends/ascend/validation/run_tests_adapter.py",
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def tensor(uid, shape, dtype="float16"):
    strides = [1] * len(shape)
    for i in range(len(shape) - 1, 0, -1):
        strides[i - 1] = strides[i] * shape[i]
    return dict(
        uid=uid,
        dimensions=shape,
        strides=strides,
        data_type=dtype,
        virtual=False,
        alignment=16,
    )


class Contracts(unittest.TestCase):
    def test_fp8_identity_emits_byte_pointers_without_enabling_fp8_math(self):
        dispatch = importlib.import_module(package + ".dispatch.extended")
        for dtype in ["fp8_e4m3", "fp8_e5m2", "fp8_e8m0"]:
            for strided in [False, True]:
                with self.subTest(dtype=dtype, strided=strided):
                    tensors = [tensor(i, [2, 257], dtype) for i in [1, 2]]
                    if strided:
                        tensors[1]["strides"] = [1030, 2]
                    parameters = dict(n_elements=514)
                    source, config = dispatch.configuration(
                        "identity", parameters, tensors
                    )
                    plan = dict(
                        configuration=config,
                        source=source,
                        tensors=tensors,
                        parameters=parameters,
                        stage_id=0,
                        operation="identity",
                        source_node_ids=[0],
                        dependencies=[],
                    )
                    with (
                        tempfile.TemporaryDirectory() as directory,
                        patch.object(
                            emitter, "plan_graph", return_value=([plan], {}, 0)
                        ),
                    ):
                        stages, _ = emitter.emit(
                            {}, Path(directory), {"identity_sha256": "a" * 64}
                        )
                    signature = stages[0]["candidates"][0]["payload"][
                        "full_signature"
                    ].split(",")
                    self.assertEqual(
                        [x.split(":")[0] for x in signature[:2]],
                        ["*u8", "*u8"],
                    )
                    with self.assertRaises(ValueError):
                        dispatch.configuration("exp", parameters, tensors)

    def test_activation_backward_dense_and_strided_launches(
        self,
    ):
        dispatch = importlib.import_module(package + ".dispatch.extended")
        common = importlib.import_module(package + ".dispatch.common")
        for operation, mode in common.BINARY_ELEMENTWISE_MODES.items():
            if not operation.endswith("_backward"):
                continue
            for strided in [False, True]:
                with self.subTest(operation=operation, strided=strided):
                    tensors = [tensor(i, [2, 3]) for i in range(1, 4)]
                    if strided:
                        for value in tensors:
                            value["strides"] = [10, 2]
                    parameters = dict(
                        n_elements=6, pointwise_mode=mode, has_upper_clip=0
                    )
                    source, config = dispatch.configuration(
                        operation, parameters, tensors
                    )
                    plan = dict(
                        configuration=config,
                        source=source,
                        tensors=tensors,
                        parameters=parameters,
                        stage_id=0,
                        operation=operation,
                        source_node_ids=[0],
                        dependencies=[],
                    )
                    with (
                        tempfile.TemporaryDirectory() as directory,
                        patch.object(
                            emitter, "plan_graph", return_value=([plan], {}, 0)
                        ),
                    ):
                        stages, size = emitter.emit(
                            {}, Path(directory), {"identity_sha256": "a" * 64}
                        )
                    self.assertEqual(len(stages[0]["argument_sources"]), 4)
                    self.assertEqual(size, 0)

    def test_attention_signature_matches_source_annotations(self):
        tensors = [tensor(i, [1, 2, 16, 32]) for i in range(1, 5)]
        tensors.append(tensor(5, [1, 2, 16, 1], "float32"))
        parameters = dict(
            batch=1,
            heads=2,
            key_heads=2,
            value_heads=2,
            sequence_q=16,
            sequence_kv=16,
            head_dimension=32,
            value_dimension=32,
            q_per_k=1,
            q_per_v=1,
            attn_scale=0.125,
            has_bias=0,
            banded=0,
            min_diag=-(2**31),
            max_diag=2**31 - 1,
            generate_stats=1,
            reverse_causal=0,
        )
        config = attention._sdpa_forward_kernel_configuration(
            parameters, tensors
        )
        plan = dict(
            configuration=config,
            source="attention",
            tensors=tensors,
            parameters=parameters,
            stage_id=0,
            operation="sdpa",
            source_node_ids=[0],
            dependencies=[],
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(emitter, "plan_graph", return_value=([plan], {}, 0)),
        ):
            stages, size = emitter.emit(
                {}, Path(directory), {"identity_sha256": "a" * 64}
            )
            stage = stages[0]
            signature = stage["candidates"][0]["payload"][
                "full_signature"
            ].split(",")
            source = next(Path(directory).glob("source-*.py")).read_text()
        entry = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == config[0]
        )
        self.assertEqual(len(signature), len(entry.args.args))
        runtime = []
        for argument, token in zip(entry.args.args, signature):
            if (
                argument.annotation
                and ast.unparse(argument.annotation) == "tl.constexpr"
            ):
                self.assertIsInstance(ast.literal_eval(token), (int, float))
            elif token.startswith("*") or token in {"i32", "fp32"}:
                runtime.append(token)
        self.assertEqual(len(runtime), len(stage["argument_sources"]))
        self.assertEqual(runtime[-1], "fp32")
        self.assertEqual(size, 0)

    def test_tf32_contract_is_rejected(self):
        dispatch = importlib.import_module(package + ".dispatch.extended")
        for operation in [
            "matmul",
            "convolution_fprop",
            "convolution_dgrad",
            "convolution_wgrad",
        ]:
            with (
                self.subTest(operation=operation),
                self.assertRaisesRegex(ValueError, "not TF32"),
            ):
                dispatch.configuration(operation, {"input_precision": 2}, [])

    def test_unused_resample_index_alias_keeps_storage_type(self):
        dispatch = importlib.import_module(package + ".dispatch.resample")
        for dtype, pointer in [
            ("float16", "*fp16"),
            ("bfloat16", "*bf16"),
            ("float32", "*fp32"),
        ]:
            tensors = [tensor(i, [1, 2, 4, 4], dtype) for i in [1, 2]]
            parameters = dict(
                mode=3,
                padding=1,
                generate_index=0,
                align_corners=0,
                window=[1, 1],
                stride=[1, 1],
                pre_padding=[0, 0],
                post_padding=[0, 0],
                n_elements=32,
            )
            config = dispatch._resample_kernel_configuration(
                parameters, tensors
            )
            self.assertEqual(config[1]["index_ptr"], pointer)

    def test_runner_selects_supplements_and_all_public_operators(self):
        for suite, suffix in [
            ("functional", "dtype"),
            ("benchmark", "fp32_output"),
        ]:
            expression = adapter.test_expression(suite, "reduction")
            self.assertRegex(f"{suite}.ascend.reduction.{suffix}", expression)
            self.assertIsNone(
                re.fullmatch(expression, f"{suite}.nvidia.reduction.{suffix}")
            )
        self.assertEqual(
            adapter.select_operator_manifests(
                {"functional": ["add", "sdpa"], "benchmark": ["add"]}
            )["benchmark"],
            ["add", "sdpa"],
        )

    def test_legacy_summary_uses_stream_latency_and_keeps_other_metrics(self):
        metrics = dict(median=2.0, p90=3.0, samples=[1.0, 2.0, 3.0])
        records = {
            "case": {
                p: {
                    "stream_us": metrics.copy(),
                    "submit_us": {"median": 1.0},
                    "end_to_end_us": {"median": 4.0},
                }
                for p in ["flagdnn", "aclnn"]
            }
        }
        result = {
            "status": "failed",
            "record_errors": [
                "skipped benchmark emitted timing provider records"
            ],
        }
        adapter.postprocess_result(
            result=result,
            ctest_reported_status="skipped",
            output="1/2 Test #5: benchmark.ascend.matmul .... Passed 1.2 sec",
            operator="matmul",
            suite="benchmark",
            records=records,
            manifest_operators=[],
        )
        self.assertEqual(result["status"], "passed")
        for record in records["case"].values():
            self.assertEqual((record["unit"], record["median"]), ("us", 2.0))
            self.assertEqual(record["end_to_end_us"]["median"], 4.0)

    def test_only_passes_and_explicit_skips_are_successful(self):
        for status in ["passed", "skipped"]:
            self.assertTrue(adapter.status_is_success(status))
        for status in ["failed", "timeout", "not_found"]:
            self.assertFalse(adapter.status_is_success(status))

    def test_unpaired_measurement_is_rejected(self):
        result = {"status": "passed"}
        adapter.postprocess_result(
            result=result,
            ctest_reported_status="passed",
            output="",
            operator="add",
            suite="benchmark",
            records={
                "case": {
                    "flagdnn": {
                        "stream_us": dict(median=2, p90=2, samples=[2])
                    }
                }
            },
            manifest_operators=[],
        )
        self.assertEqual(result["status"], "failed")


class CannInstallationContracts(unittest.TestCase):
    @staticmethod
    def _metadata(package="Ascend-cann", version="9.0.0", arch="aarch64"):
        return (
            f"package_name={package}\nversion={version}\n"
            f"innerversion=V100R001C10SPC001B250\narch={arch}\n"
        )

    def _configure(self, infos, *, processor="aarch64", missing=(), alias=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cann = root / "cann-9.0.0"
            arch_root = cann / f"{processor}-linux"
            for name in (
                "include/acl/acl.h",
                "include/acl/acl_rt.h",
                "lib64/libascendcl.so",
                "lib64/libruntime.so",
            ):
                if name not in missing:
                    path = arch_root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
            for name, content in infos.items():
                (arch_root / name).write_text(content)
            selected = cann
            if alias:
                selected = root / "latest"
                selected.symlink_to(cann, target_is_directory=True)
            (root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.23)\n"
                "project(CannInstallContract LANGUAGES NONE)\n"
                f'set(CMAKE_SYSTEM_PROCESSOR "{processor}")\n'
                f'set(CANN_ROOT "{selected}")\n'
                "add_subdirectory(production)\nadd_subdirectory(validation)\n"
            )
            # Match the real build's sibling directory scopes and imported targets.
            for child in ("production", "validation"):
                child_dir = root / child
                child_dir.mkdir()
                include = (
                    f'include("{ROOT}/backends/ascend/cmake/TritonJITCompat.cmake")\n'
                    if child == "production"
                    else ""
                )
                (child_dir / "CMakeLists.txt").write_text(
                    include
                    + 'flagdnn_ascend_prepare_triton_jit("${CANN_ROOT}")\n'
                    + 'if(NOT FLAGDNN_ASCEND_CANN_VERSION STREQUAL "9.0.0" OR\n'
                    + "   NOT FLAGDNN_ASCEND_CANN_INNER_VERSION STREQUAL\n"
                    + '       "V100R001C10SPC001B250" OR\n'
                    + f'   NOT FLAGDNN_ASCEND_CANN_ROOT STREQUAL "{cann}")\n'
                    + '  message(FATAL_ERROR "CANN resolver outputs differ")\n'
                    + "endif()\n"
                )
            return subprocess.run(
                ["cmake", "-S", str(root), "-B", str(root / "build")],
                capture_output=True,
                text=True,
                timeout=30,
            )

    def test_cann_installation_formats(self):
        for arch in ("aarch64", "x86_64"):
            toolkit = {
                "ascend_toolkit_install.info": self._metadata(
                    "Ascend-cann-toolkit", arch=arch
                )
            }
            combined = {"ascend_all_cann_install.info": self._metadata(arch=arch)}
            for infos in (toolkit, combined, toolkit | combined):
                for alias in (False, True):
                    with self.subTest(arch=arch, files=list(infos), alias=alias):
                        result = self._configure(infos, processor=arch, alias=alias)
                        self.assertEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )

    def test_cann_installation_rejects_invalid_metadata(self):
        name = "ascend_all_cann_install.info"
        valid = self._metadata()
        toolkit = "ascend_toolkit_install.info"
        cases = [
            ({}, "No ascend_toolkit_install.info"),
            ({name: self._metadata("Ascend-cann-ops")}, "unexpected package_name"),
            ({name: self._metadata(version="8.5.0")}, "requires CANN"),
            ({name: self._metadata(version="10.0.0")}, "requires CANN"),
            ({name: self._metadata(arch="x86_64")}, "does not match host"),
            ({name: valid + "version=9.0.0\n"}, "exactly one version="),
            ({name: valid.replace("arch=aarch64\n", "")}, "exactly one arch="),
            (
                {
                    toolkit: self._metadata("Ascend-cann-toolkit"),
                    name: self._metadata(version="9.1.0"),
                },
                "candidates disagree",
            ),
            (
                {
                    toolkit: self._metadata("Ascend-cann-toolkit"),
                    name: valid.replace("SPC001B250", "SPC001B251"),
                },
                "candidates disagree",
            ),
            (
                {
                    toolkit: self._metadata("Ascend-cann-ops"),
                    name: valid,
                },
                "unexpected package_name",
            ),
        ]
        for infos, error in cases:
            with self.subTest(infos=infos, error=error):
                result = self._configure(infos)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stdout + result.stderr)

    def test_cann_installation_requires_headers_and_libraries(self):
        for name, package in (
            ("ascend_toolkit_install.info", "Ascend-cann-toolkit"),
            ("ascend_all_cann_install.info", "Ascend-cann"),
        ):
            for missing in (
                ("include/acl/acl.h", "include/acl/acl_rt.h"),
                ("lib64/libascendcl.so",),
                ("lib64/libruntime.so",),
            ):
                with self.subTest(name=name, missing=missing):
                    result = self._configure(
                        {name: self._metadata(package)},
                        missing=missing,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("missing ACL headers", result.stdout + result.stderr)

    def test_cann_metadata_is_part_of_compiler_identity(self):
        identity = importlib.import_module(package + ".compiler_identity")
        for arch in ("aarch64", "x86_64"):
            for filename in (
                "ascend_toolkit_install.info",
                "ascend_all_cann_install.info",
            ):
                with self.subTest(arch=arch, filename=filename):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        info = root / f"{arch}-linux" / filename
                        info.parent.mkdir()
                        info.write_text(self._metadata(arch=arch))
                        with (
                            patch.dict(
                                identity.os.environ,
                                {"ASCEND_HOME_PATH": directory},
                                clear=True,
                            ),
                            patch.object(
                                identity.platform, "machine", return_value=arch
                            ),
                        ):
                            before = identity._configured_external_files()
                            self.assertIn("cann_package_info", before)
                            info.write_text(self._metadata(version="9.1.0", arch=arch))
                            after = identity._configured_external_files()
                            self.assertNotEqual(
                                before["cann_package_info"], after["cann_package_info"]
                            )


if __name__ == "__main__":
    unittest.main()
