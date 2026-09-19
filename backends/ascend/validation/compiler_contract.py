"""Host regressions for Ascend launch ABI and unified runner integration."""

from pathlib import Path
import ast
import importlib
import importlib.util
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "compiler"))
from flagdnn_codegen.provider_loader import get_provider

provider = get_provider("ascend")
package = provider.__package__
emitter = importlib.import_module(package + ".codegen.extended")
attention = importlib.import_module(package + ".dispatch.attention_forward")
spec = importlib.util.spec_from_file_location(
    "ascend_runner_adapter", ROOT / "backends/ascend/validation/run_tests_adapter.py"
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
    def test_activation_backward_launches_resolve_for_dense_and_strided_tensors(self):
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
                    with tempfile.TemporaryDirectory() as directory, patch.object(
                        emitter, "plan_graph", return_value=([plan], {}, 0)
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
        config = attention._sdpa_forward_kernel_configuration(parameters, tensors)
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
        with tempfile.TemporaryDirectory() as directory, patch.object(
            emitter, "plan_graph", return_value=([plan], {}, 0)
        ):
            stages, size = emitter.emit(
                {}, Path(directory), {"identity_sha256": "a" * 64}
            )
            stage = stages[0]
            signature = stage["candidates"][0]["payload"]["full_signature"].split(",")
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
            with self.subTest(operation=operation), self.assertRaisesRegex(
                ValueError, "not TF32"
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
            config = dispatch._resample_kernel_configuration(parameters, tensors)
            self.assertEqual(config[1]["index_ptr"], pointer)

    def test_runner_selects_supplements_and_all_public_operators(self):
        for suite, suffix in [("functional", "dtype"), ("benchmark", "fp32_output")]:
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
            "record_errors": ["skipped benchmark emitted timing provider records"],
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
                "case": {"flagdnn": {"stream_us": dict(median=2, p90=2, samples=[2])}}
            },
            manifest_operators=[],
        )
        self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
