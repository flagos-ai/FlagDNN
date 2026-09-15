"""Host-only regression contracts for NVIDIA graph planning."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "compiler"))
from flagdnn_codegen.provider_loader import _load_provider, get_provider


def tensor(uid, dimensions, virtual=False):
    strides = []
    stride = 1
    for extent in reversed(dimensions):
        strides.insert(0, stride)
        stride *= extent
    return dict(
        uid=uid,
        dimensions=dimensions,
        strides=strides,
        data_type="float32",
        virtual=virtual,
        alignment=16,
    )


def convolution_parameters():
    return dict(
        spatial_rank=2,
        groups=1,
        stride=[1, 1],
        pre_padding=[1, 1],
        post_padding=[1, 1],
        dilation=[1, 1],
        convolution_mode=0,
    )


class PlanningContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = get_provider("nvidia")
        package = cls.compiler.__package__
        cls.graph = importlib.import_module(f"{package}.dispatch.graph")
        cls.dispatch = importlib.import_module(f"{package}.dispatch.selection")
        cls.tensor_abi = importlib.import_module(f"{package}.dispatch.tensor")

    def _partial_gradient_stage(
        self,
        *,
        fp8=False,
        sequence=448,
        batch=1,
        heads=4,
        kv_heads=2,
        dimension=64,
    ):
        kind = "fp8_e4m3" if fp8 else "float16"
        shapes = (
            [batch, heads, sequence, dimension],
            [batch, kv_heads, 512, dimension],
            [batch, kv_heads, 512, dimension],
        )
        tensors = [
            dict(tensor(i + 1, shape), data_type=kind)
            for i, shape in enumerate(shapes)
        ]
        if fp8:
            tensors.extend(tensor(i + 1, [1]) for i in range(3, 22))
            tensors[17] = dict(tensor(18, shapes[1]), data_type=kind)
            tensors[18] = dict(tensor(19, shapes[2]), data_type=kind)
            outputs = [18, 19, 20, 21, 22]
        else:
            tensors.extend(tensor(i + 1, [1]) for i in range(3, 6))
            tensors.extend(
                dict(tensor(i + 1, shapes[1]), data_type=kind) for i in (6, 7)
            )
            outputs = [7, 8]
        operation = "sdpa_fp8_backward" if fp8 else "sdpa_backward"
        key = "_sdpa_fp8_bwd_stage" if fp8 else "_sdpa_bwd_stage"
        return dict(
            operation=operation,
            parameters={key: "dkdv", "attn_scale": 0.125},
            tensors=tensors,
            input_uids=[1, 2, 3],
            output_uids=outputs,
            source_node_ids=[0],
        )

    def test_attention_partials_preserve_output_views_and_tail_coverage(self):
        module = importlib.import_module(
            f"{self.compiler.__package__}.dispatch.attention_gradients"
        )
        stage = self._partial_gradient_stage()
        for output in stage["tensors"][-2:]:
            output["strides"] = [65536, 64, 128, 1]
        registry = {t["uid"]: t for t in stage["tensors"]}
        partial, reduce = module.split_attention_gradients([stage], registry)
        self.assertEqual(reduce["output_uids"], stage["output_uids"])
        self.assertTrue(
            set(partial["output_uids"]).isdisjoint(stage["output_uids"])
        )
        chunk = partial["parameters"]["_sdpa_query_chunk"]
        covered = [
            q
            for start in range(0, 448, chunk)
            for q in range(start, min(start + chunk, 448))
        ]
        self.assertEqual(covered, list(range(448)))
        plan = module.partial_gradient_reduction(
            reduce["operation"], reduce["parameters"], reduce["tensors"]
        )
        self.assertEqual(
            [plan[2]["K" + axis] for axis in "BHSD"],
            stage["tensors"][-2]["strides"],
        )
        self.assertEqual(plan[1]["pk_ptr"], "*fp32")
        self.assertEqual(plan[1]["dk_ptr"], "*fp16")

    def test_attention_partials_bound_workspace_and_keep_fp32_path(self):
        import copy

        module = importlib.import_module(
            f"{self.compiler.__package__}.dispatch.attention_gradients"
        )
        for stage in (
            self._partial_gradient_stage(batch=4, heads=64),
            self._partial_gradient_stage(sequence=64),
            self._partial_gradient_stage(),
        ):
            if (
                stage["tensors"][0]["dimensions"][1] == 4
                and stage["tensors"][0]["dimensions"][2] == 448
            ):
                stage["tensors"][0]["data_type"] = "float32"
            registry = {t["uid"]: t for t in stage["tensors"]}
            before = copy.deepcopy(registry)
            self.assertEqual(
                module.split_attention_gradients([stage], registry), [stage]
            )
            self.assertEqual(registry, before)

    def test_fp8_attention_partials_keep_amax_and_reject_wrong_scale_storage(
        self,
    ):
        import copy

        module = importlib.import_module(
            f"{self.compiler.__package__}.dispatch.attention_gradients"
        )
        stage = self._partial_gradient_stage(fp8=True, dimension=128)
        registry = {t["uid"]: t for t in stage["tensors"]}
        _, reduce = module.split_attention_gradients([stage], registry)
        self.assertEqual(reduce["output_uids"], [18, 19, 20, 21])
        plan = module.partial_gradient_reduction(
            reduce["operation"], reduce["parameters"], reduce["tensors"]
        )
        self.assertEqual(plan[1]["dk_ptr"], "*fp8e4nv")
        malformed = copy.deepcopy(reduce["tensors"])
        malformed[4]["data_type"] = "float16"
        with self.assertRaisesRegex(ValueError, "float32"):
            module.partial_gradient_reduction(
                reduce["operation"], reduce["parameters"], malformed
            )
        malformed = copy.deepcopy(reduce["tensors"])
        malformed[3]["data_type"] = "float16"
        with self.assertRaisesRegex(ValueError, "output metadata"):
            module.partial_gradient_reduction(
                reduce["operation"], reduce["parameters"], malformed
            )

    def test_identity_covers_all_compiler_modules(self):
        identity = self.compiler.compiler_identity("sm_90", "libtriton_jit")
        provider_root = Path(self.compiler.__file__).parent
        modules = [
            module
            for module in provider_root.rglob("*.py")
            if module.relative_to(provider_root).parts[0]
            in {"dispatch", "codegen"}
            and "__pycache__" not in module.relative_to(provider_root).parts
        ]
        self.assertTrue((provider_root / "dispatch/__init__.py").is_file())
        self.assertTrue((provider_root / "codegen/__init__.py").is_file())
        self.assertGreater(len(modules), 10)
        for module in modules:
            relative = module.relative_to(provider_root).as_posix()
            if relative == "codegen/identity.py":
                label = "provider_identity:nvidia"
            else:
                label = f"provider_module:nvidia:{relative}"
            self.assertEqual(
                identity["source_files"][label],
                hashlib.sha256(module.read_bytes()).hexdigest(),
            )

    def test_identity_keeps_common_and_platform_kernels_with_the_same_name(
        self,
    ):
        from unittest import mock
        from flagdnn_codegen.kernel_registry import (
            iter_kernel_candidates,
            resolve_kernel_source,
        )

        identity_module = importlib.import_module(
            f"{self.compiler.__package__}.codegen.identity"
        )
        before = self.compiler.compiler_identity("sm_90")
        compiler_entry = (
            Path(__file__).resolve().parents[3]
            / "compiler/flagdnn_codegen/main.py"
        )
        candidates = {
            candidate.ownership: candidate
            for candidate in iter_kernel_candidates("nvidia")
            if candidate.source == "normalization.py"
        }
        self.assertEqual(set(candidates), {"common", "platform"})
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "normalization.py"
            for candidate in candidates.values():
                source = resolve_kernel_source(compiler_entry, candidate)
                changed.write_bytes(
                    source.read_bytes() + b"\n# invalidation probe\n"
                )

                def resolve(entry, requested):
                    path = resolve_kernel_source(entry, requested)
                    return changed if path == source else path

                with self.subTest(
                    ownership=candidate.ownership
                ), mock.patch.object(
                    identity_module,
                    "resolve_kernel_source",
                    side_effect=resolve,
                ):
                    after = self.compiler.compiler_identity("sm_90")
                    self.assertNotEqual(
                        before["identity_sha256"], after["identity_sha256"]
                    )

    def test_relocated_provider_identity_tracks_nested_modules_and_initializers(
        self,
    ):
        original = self.compiler.compiler_identity("sm_90")
        source = Path(self.compiler.__file__).parent
        package = "_flagdnn_backend_nvidia_relocated_contract"
        with tempfile.TemporaryDirectory() as directory:
            relocated = Path(directory)
            shutil.copy2(source / "compiler.py", relocated / "compiler.py")
            for name in ("dispatch", "codegen"):
                shutil.copytree(
                    source / name,
                    relocated / name,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            try:
                provider = _load_provider(
                    "nvidia_relocated_contract", relocated / "compiler.py"
                )
                self.assertEqual(provider.compiler_identity("sm_90"), original)
                for relative in (
                    "dispatch/__init__.py",
                    "codegen/__init__.py",
                    "dispatch/nested/helper.py",
                    "codegen/nested/helper.py",
                ):
                    with self.subTest(module=relative):
                        module = relocated / relative
                        contents = (
                            module.read_bytes() if module.exists() else None
                        )
                        module.parent.mkdir(parents=True, exist_ok=True)
                        module.write_bytes(
                            (contents or b"")
                            + b"\n# identity invalidation probe\n"
                        )
                        try:
                            changed = provider.compiler_identity("sm_90")
                            self.assertNotEqual(
                                changed["identity_sha256"],
                                original["identity_sha256"],
                            )
                        finally:
                            if contents is None:
                                module.unlink()
                            else:
                                module.write_bytes(contents)
                self.assertEqual(provider.compiler_identity("sm_90"), original)
            finally:
                for name in tuple(sys.modules):
                    if name == package or name.startswith(package + "."):
                        del sys.modules[name]

    def test_nvidia_rejects_non_jit_engines_before_reading_request(self):
        for engine in ("external_artifact", "unknown", ""):
            with self.subTest(engine=engine):
                with self.assertRaisesRegex(
                    ValueError, "only supports.*libtriton_jit"
                ):
                    self.compiler.compiler_identity("sm_90", engine)
                with self.assertRaisesRegex(
                    ValueError, "only supports.*libtriton_jit"
                ):
                    self.compiler.compile_request(
                        Path("missing-request"), Path("unused"), engine
                    )
        self.assertEqual(
            self.compiler.compiler_identity("sm_90")["execution_engine"],
            "libtriton_jit",
        )

    def test_layernorm_row_grouping_updates_grid_and_keeps_tail(self):
        tuning = importlib.import_module(
            f"{self.compiler.__package__}.dispatch.tuning"
        )
        for rows, group, expected in ((1024, 2, 512), (129, 2, 65), (1, 2, 1)):
            grid = tuning._autotune_variant_grid(
                strategy="fixed_grid",
                key="normalized_elements",
                function_name="layer_norm_kernel",
                parameters={"rows": rows, "normalized_elements": 1024},
                constants={"ROWS_PER_PROGRAM": group},
                meta={},
                default_grid=(rows, 1, 1),
            )
            self.assertEqual(grid, (expected, 1, 1))

    def test_rmsnorm_warp_dispatch_boundaries_and_tensor_bindings(self):
        from flagdnn_codegen.kernel_registry import select_kernel_candidate

        candidate = select_kernel_candidate("nvidia", "rmsnorm_warp")
        self.assertEqual(candidate.ownership, "platform")
        for dtype in ("float16", "float32", "bfloat16"):
            for rows, extent in (
                (63, 512),
                (64, 512),
                (65, 127),
                (65, 33),
                (65, 31),
                (65, 513),
            ):
                tensors = [
                    dict(tensor(i, shape), data_type=dtype)
                    for i, shape in enumerate(
                        ([rows, extent], [extent], [extent], [rows, extent]), 1
                    )
                ] + [tensor(5, [rows])]
                config = self.dispatch._kernel_configuration(
                    "rmsnorm",
                    dict(rows=rows, normalized_elements=extent, epsilon=1e-3),
                    tensors,
                    90,
                )
                use_warp = (
                    dtype == "float16" and rows >= 64 and 32 <= extent <= 512
                )
                self.assertEqual(
                    config[0],
                    "rms_norm_warp_kernel" if use_warp else "rms_norm_kernel",
                )
                if use_warp:
                    self.assertIn(config[0], candidate.functions)
                    self.assertEqual(config[3], ((rows + 3) // 4, 1, 1))
                    self.assertEqual(config[1]["inv_variance_ptr"], "*fp32")
                    self.assertEqual(
                        config[4],
                        [("tensor_alias", i) for i in (0, 3, 1, 2, 4)],
                    )

    def test_rmsnorm_grouped_candidates_cover_rows_and_preserve_tail(self):
        from flagdnn_codegen.kernel_registry import select_kernel_candidate

        tuning = importlib.import_module(
            f"{self.compiler.__package__}.dispatch.tuning"
        )
        candidate = select_kernel_candidate("nvidia", "rmsnorm")
        for rows, extent in ((1, 64), (771, 513), (1024, 4096), (65, 8192)):
            parameters = dict(
                rows=rows, normalized_elements=extent, epsilon=1e-5
            )
            tensors = [
                tensor(i, shape)
                for i, shape in enumerate(
                    (
                        [rows, extent],
                        [extent],
                        [extent],
                        [rows, extent],
                        [rows],
                    ),
                    1,
                )
            ]
            config = self.dispatch._kernel_configuration(
                "rmsnorm", parameters, tensors, 90
            )
            variants, _, _, _ = tuning._prepare_tuning_variants(
                compiler_path=Path(__file__).resolve().parents[3]
                / "compiler/flagdnn_codegen/main.py",
                candidate=candidate,
                operation="rmsnorm",
                function_name=config[0],
                parameters=parameters,
                constants=config[2],
                default_grid=config[3],
            )
            grouped = [v for v in variants if v[1]["ROWS_PER_PROGRAM"] == 2]
            self.assertTrue(grouped)
            for _, constants, _, grid in grouped:
                self.assertEqual(grid, ((rows + 1) // 2, 1, 1))
                self.assertEqual(constants["STATIC_ROWS"], rows)
                self.assertEqual(constants["N"], extent)
                self.assertEqual(constants["RETURN_STATS"], True)

    def test_layernorm_static_row_hint_matches_runtime_argument(self):
        for rows in (1, 129, 1024):
            parameters = dict(
                rows=rows, normalized_elements=1024, epsilon=1e-5
            )
            tensors = [
                tensor(i, shape)
                for i, shape in enumerate(
                    (
                        [rows, 1024],
                        [1024],
                        [1024],
                        [rows, 1024],
                        [rows],
                        [rows],
                    ),
                    1,
                )
            ]
            config = self.dispatch._kernel_configuration(
                "layernorm", parameters, tensors, 90
            )
            self.assertEqual(config[2]["STATIC_ROWS"], rows)
            self.assertTrue(
                all(
                    type(value) in (int, float, bool)
                    for value in config[2].values()
                )
            )
            self.assertIn(("scalar_i32", "rows"), config[4])

    def test_low_precision_tma_matmul_preserves_batch_boundaries(self):
        from flagdnn_codegen.kernel_registry import select_kernel_candidate

        tuning = importlib.import_module(
            f"{self.compiler.__package__}.dispatch.tuning"
        )
        for dtype in ("float16", "bfloat16"):
            for m, n, k, expected in (
                (512, 512, 512, "matmul_batched_tma_short_kernel"),
                (1024, 1024, 1024, "matmul_batched_tma_persistent_kernel"),
                (513, 512, 512, "matmul_batched_contiguous_kernel"),
                (512, 515, 512, "matmul_batched_contiguous_kernel"),
                (512, 512, 520, "matmul_batched_contiguous_kernel"),
            ):
                tensors = [
                    {**tensor(i, dims), "data_type": dtype}
                    for i, dims in enumerate(
                        ([4, m, k], [4, k, n], [4, m, n]), 1
                    )
                ]
                parameters = dict(batch=4, m=m, n=n, k=k)
                config = self.dispatch._kernel_configuration(
                    "matmul", parameters, tensors, 90
                )
                self.assertEqual(config[0], expected)
                if expected != "matmul_batched_tma_short_kernel":
                    continue
                candidate = select_kernel_candidate("nvidia", "matmul")
                self.assertIn(expected, candidate.functions)
                variants, _, _, table = tuning._prepare_tuning_variants(
                    compiler_path=Path(__file__).resolve().parents[3]
                    / "compiler/flagdnn_codegen/main.py",
                    candidate=candidate,
                    function_name=expected,
                    operation="matmul",
                    parameters=parameters,
                    constants=config[2],
                    default_grid=config[3],
                )
                self.assertEqual(table, "matmul_short_persistent")
                for _, constants, _, grid in variants:
                    tiles = (
                        4
                        * (
                            (m + constants["BLOCK_M"] - 1)
                            // constants["BLOCK_M"]
                        )
                        * (
                            (n + constants["BLOCK_N"] - 1)
                            // constants["BLOCK_N"]
                        )
                    )
                    self.assertEqual(
                        grid, (min(tiles, constants["PERSISTENT_GRID"]), 1, 1)
                    )
                    self.assertEqual(m % constants["BLOCK_M"], 0)
                    self.assertEqual(k % constants["BLOCK_K"], 0)

    def test_short_matmul_batch_policy_preserves_large_batch_schedule(self):
        for dtype in ("float16", "bfloat16"):
            for batch in (1, 2, 4, 8, 16, 32, 33, 64):
                tensors = [
                    {**tensor(i, [batch, 512, 512]), "data_type": dtype}
                    for i in (1, 2, 3)
                ]
                config = self.dispatch._kernel_configuration(
                    "matmul",
                    dict(batch=batch, m=512, n=512, k=512),
                    tensors,
                    90,
                )
                self.assertEqual(
                    config[0],
                    (
                        "matmul_batched_tma_short_kernel"
                        if batch <= 32
                        else "matmul_batched_tma_persistent_kernel"
                    ),
                )
                if batch < 8:
                    self.assertEqual(config[2]["BLOCK_M"], 64)

    def test_low_precision_wgrad_short_split_uses_pointer_loads(self):
        from flagdnn_codegen.kernel_registry import select_kernel_candidate

        tuning = importlib.import_module(
            f"{self.compiler.__package__}.dispatch.tuning"
        )
        for dtype in ("float16", "bfloat16", "float32"):
            tensors = {
                i: {**tensor(i, shape), "data_type": dtype}
                for i, shape in enumerate(
                    ([8, 128, 28, 28], [8, 64, 28, 28], [128, 64, 1, 1]), 1
                )
            }
            group = dict(
                operation="convolution_wgrad",
                parameters={
                    **convolution_parameters(),
                    "pre_padding": [0, 0],
                    "post_padding": [0, 0],
                    "n_outputs": 128 * 64,
                },
                tensors=list(tensors.values()),
                source_node_ids=[0],
                input_uids=[1, 2],
                output_uids=[3],
            )
            stages = self.graph._expand_execution_pipelines([group], tensors)
            if dtype == "float32":
                self.assertEqual(
                    stages[0]["parameters"]["_wgrad_pipeline_algorithm"], "1x1"
                )
                continue
            self.assertEqual(len(stages), 2)
            stage = stages[0]
            self.assertEqual(stage["parameters"]["_wgrad_num_splits"], 32)
            config = self.dispatch._kernel_configuration(
                stage["operation"], stage["parameters"], stage["tensors"], 90
            )
            self.assertEqual(config[0], "_conv_wgrad2d_batched_split_kernel")
            self.assertEqual(config[3], (4, 4, 8))
            self.assertEqual(stage["tensors"][-1]["data_type"], dtype)
            candidate = select_kernel_candidate("nvidia", "convolution_wgrad")
            self.assertIn(config[0], candidate.functions)
            variants, _, _, table = tuning._prepare_tuning_variants(
                compiler_path=Path(__file__).resolve().parents[3]
                / "compiler/flagdnn_codegen/main.py",
                candidate=candidate,
                function_name=config[0],
                operation=stage["operation"],
                parameters=stage["parameters"],
                constants=config[2],
                default_grid=config[3],
            )
            self.assertEqual(table, "conv_wgrad_short_split")
            self.assertTrue(variants)
            self.assertTrue(
                all(grid[1:] == (4, 8) for _, _, _, grid in variants)
            )

    def test_compiled_scratch_is_per_cta_and_shared_between_stages(self):
        from types import SimpleNamespace

        resources = importlib.import_module(
            f"{self.compiler.__package__}.codegen.resources"
        )
        self.assertEqual(
            resources._effective_jit_warps(
                SimpleNamespace(num_warps=12), 4, True
            ),
            12,
        )
        self.assertEqual(
            resources._effective_jit_warps(
                SimpleNamespace(num_warps=4), 4, False
            ),
            4,
        )
        for effective, cached in (
            (12, False),
            (2, True),
            (33, True),
            (True, True),
        ):
            with self.assertRaises(ValueError):
                resources._effective_jit_warps(
                    SimpleNamespace(num_warps=effective), 4, cached
                )
        metadata = SimpleNamespace(
            global_scratch_size=384, global_scratch_align=128
        )
        self.assertEqual(
            resources._launch_scratch_size(metadata, (128, 1, 1), 4096), 49152
        )
        self.assertEqual(
            resources._launch_scratch_size(metadata, (2, 3, 4)), 9216
        )
        self.assertEqual(
            resources._launch_scratch_size(
                SimpleNamespace(), (1000, 1, 1), 4096
            ),
            4096,
        )
        self.assertEqual(
            resources._launch_scratch_size(SimpleNamespace(), (1000, 1, 1)), 0
        )
        launch = lambda size: {"launch": {"global_scratch_size": size}}
        self.assertEqual(
            resources._program_scratch_size(
                [
                    launch(4096),
                    {"variants": [launch(32768), launch(49152)]},
                    launch(8192),
                ]
            ),
            49152,
        )
        for bad in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                resources._launch_scratch_size(
                    SimpleNamespace(global_scratch_size=bad), (1, 1, 1)
                )
        with self.assertRaises(ValueError):
            resources._launch_scratch_size(
                SimpleNamespace(global_scratch_size=1 << 62), (4, 1, 1)
            )
        with self.assertRaises(ValueError):
            resources._launch_scratch_size(
                SimpleNamespace(profile_scratch_size=128), (1, 1, 1)
            )

    def test_batchnorm_inference_tile_clamp_compiles_without_a_device(self):
        from contextlib import ExitStack
        import os
        import tempfile
        from unittest.mock import patch
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        io = importlib.import_module(f"{self.compiler.__package__}.codegen.io")
        path = (
            Path(__file__).resolve().parents[3]
            / "kernels/common/normalization.py"
        )
        function = io._load_generated_module(
            path, 102
        ).batch_norm_inference_nchw_kernel
        pointers = {name: "*fp32" for name in function.arg_names[:6]}
        with tempfile.TemporaryDirectory(prefix="flagdnn-bn-clamp-") as cache:
            with ExitStack() as isolation:
                isolation.enter_context(
                    patch.dict(os.environ, {"TRITON_CACHE_DIR": cache})
                )
                # FlagTree's optional hint resolver probes the active device even
                # for explicit-target compilation. Isolate only that discovery;
                # the NVIDIA hint handler and the actual compiler still run.
                if (
                    importlib.util.find_spec("triton.compiler.hint_manager")
                    is not None
                ):
                    hints = importlib.import_module(
                        "triton.compiler.hint_manager"
                    )
                    isolation.enter_context(
                        patch.object(
                            hints,
                            "hint_get_flagtree_backend",
                            return_value="cuda",
                        )
                    )
                    isolation.enter_context(
                        patch.object(hints, "_global_hint_manager", None)
                    )
                for spatial in (1, 128, 257, 4096):
                    with self.subTest(spatial=spatial):
                        constants = dict(
                            C=37,
                            S=spatial,
                            eps=0.0,
                            BLOCK_SIZE=256,
                            HAS_WEIGHT=True,
                            HAS_BIAS=True,
                            STAT_IS_INV_VARIANCE=True,
                        )
                        signature = {
                            **pointers,
                            **{key: "constexpr" for key in constants},
                        }
                        compiled = triton.compile(
                            ASTSource(
                                fn=function,
                                signature=signature,
                                constexprs=constants,
                            ),
                            target=GPUTarget("cuda", 90, 32),
                            options={"num_warps": 4, "num_stages": 1},
                        )
                        self.assertEqual(
                            compiled.metadata.name,
                            "batch_norm_inference_nchw_kernel",
                        )
                        self.assertEqual(
                            compiled.metadata.global_scratch_size, 0
                        )

    def test_tensor_map_matmul_large_output_uses_64_bit_row_offsets(self):
        from contextlib import ExitStack
        from unittest.mock import patch
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        io = importlib.import_module(f"{self.compiler.__package__}.codegen.io")
        path = Path(__file__).resolve().parents[1] / "kernels/matmul.py"
        function = io._load_generated_module(
            path, 103
        ).matmul_tf32_tensor_map_kernel
        # Valid TensorMap dimensions, but rows above 32767 require offsets
        # beyond INT32_MAX. Compile the real kernel without allocating C.
        constants = dict(
            BATCH=4,
            M=65536,
            N=65536,
            K=1024,
            BLOCK_M=256,
            BLOCK_N=128,
            BLOCK_K=32,
            GROUP_M=8,
            PERSISTENT_GRID=132,
        )
        signature = {
            "a_desc": "tensordesc<fp32[256,32]>",
            "b_desc": "tensordesc<fp32[32,128]>",
            "c_ptr": "*fp32",
            **{key: "constexpr" for key in constants},
        }
        with ExitStack() as isolation:
            if (
                importlib.util.find_spec("triton.compiler.hint_manager")
                is not None
            ):
                hints = importlib.import_module("triton.compiler.hint_manager")
                isolation.enter_context(
                    patch.object(
                        hints, "hint_get_flagtree_backend", return_value="cuda"
                    )
                )
                isolation.enter_context(
                    patch.object(hints, "_global_hint_manager", None)
                )
            compiled = triton.compile(
                ASTSource(
                    fn=function, signature=signature, constexprs=constants
                ),
                target=GPUTarget("cuda", 90, 32),
                options={"num_warps": 8, "num_stages": 3},
            )
        self.assertRegex(
            compiled.asm["ttir"], r"arith\.muli[^\n]+: tensor<256x1xi64>"
        )
        self.assertNotRegex(
            compiled.asm["ttir"], r"arith\.muli[^\n]+: tensor<256x1xi32>"
        )

    def test_rank_one_pointwise_chain_is_not_a_convolution(self):
        tensors = {i: tensor(i, [16], i in (2, 3)) for i in range(1, 5)}
        nodes = [
            (i, "relu", {}, [tensors[i + 1], tensors[i + 2]], [i + 1], [i + 2])
            for i in range(3)
        ]
        groups = self.graph._lower_execution_groups(nodes, tensors)
        self.assertEqual([g["operation"] for g in groups], ["relu"] * 3)

    def test_matrix_transpose_grid_includes_both_tails(self):
        from flagdnn_codegen.kernel_registry import select_kernel_candidate

        pipeline = importlib.import_module(
            f"{self.compiler.__package__}.dispatch.pipeline_dgrad"
        )
        tuning = importlib.import_module(
            f"{self.compiler.__package__}.dispatch.tuning"
        )
        source, destination = tensor(1, [257, 259]), tensor(2, [259, 257])
        stage = pipeline._transpose_matrix_stage(
            source, destination, 257, 259, [0]
        )
        config = self.dispatch._kernel_configuration(
            "transpose", stage["parameters"], stage["tensors"], 90
        )
        self.assertEqual(config[0], "matrix_transpose_kernel")
        self.assertEqual(config[3], (9, 9, 1))
        variants, _, _, table = tuning._prepare_tuning_variants(
            compiler_path=Path(__file__).resolve().parents[3]
            / "compiler/flagdnn_codegen/main.py",
            candidate=select_kernel_candidate("nvidia", "transpose"),
            operation="transpose",
            function_name=config[0],
            parameters=stage["parameters"],
            constants=config[2],
            default_grid=config[3],
        )
        self.assertEqual(table, "matrix_transpose")
        self.assertEqual(
            {variant[3] for variant in variants}, {(17, 17, 1), (9, 9, 1)}
        )

    def test_matrix_transpose_respects_cuda_grid_y_limit(self):
        layout = importlib.import_module(
            f"{self.compiler.__package__}.dispatch.layout"
        )
        for columns, expected in (
            (65535 * 16, "matrix_transpose_kernel"),
            (65536 * 16, "layout_copy_kernel"),
        ):
            stage = layout._transpose_matrix_stage(
                tensor(1, [16, columns]),
                tensor(2, [columns, 16]),
                16,
                columns,
                [0],
            )
            config = self.dispatch._kernel_configuration(
                "transpose", stage["parameters"], stage["tensors"], 90
            )
            self.assertEqual(config[0], expected)

    def test_dgrad_p5_k_contiguous_layout_and_workspace(self):
        for dtype in ("float32", "float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                tensors = {
                    i: {**tensor(i, dims), "data_type": dtype}
                    for i, dims in enumerate(
                        ([1, 768, 20, 20], [768, 768, 3, 3], [1, 768, 40, 40]),
                        1,
                    )
                }
                group = dict(
                    operation="convolution_dgrad",
                    parameters={**convolution_parameters(), "stride": [2, 2]},
                    tensors=list(tensors.values()),
                    source_node_ids=[0],
                    input_uids=[1, 2],
                    output_uids=[3],
                )
                groups = self.graph._expand_execution_pipelines(
                    [group], tensors
                )
                self.assertEqual(
                    [stage["operation"] for stage in groups],
                    ["transpose", "transpose"] + ["convolution_dgrad"] * 4,
                )
                _, size = self.graph._workspace_layout(tensors, groups)
                self.assertEqual(
                    size,
                    (768 * 768 * 9 + 768 * 400)
                    * (4 if dtype == "float32" else 2),
                )
                configs = [
                    self.dispatch._kernel_configuration(
                        stage["operation"],
                        stage["parameters"],
                        stage["tensors"],
                        90,
                    )
                    for stage in groups
                ]
                self.assertEqual(
                    [c[0] for c in configs[:2]],
                    ["matrix_transpose_kernel"] * 2,
                )
                for config in configs[2:]:
                    self.assertEqual(config[2]["loss_stride_c"], 1)
                    self.assertEqual(config[2]["WEIGHT_STRIDE_CO"], 1)
                    self.assertEqual(config[2]["WEIGHT_STRIDE_CI"], 9 * 768)
                bad = [dict(t) for t in groups[2]["tensors"]]
                bad[0]["strides"] = [307200, 2, 15360, 768]
                with self.assertRaises(ValueError):
                    self.dispatch._kernel_configuration(
                        "convolution_dgrad", groups[2]["parameters"], bad, 90
                    )

    def test_dgrad_3d_retains_its_original_packed_layout(self):
        for dtype in ("float32", "float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                tensors = {
                    i: {**tensor(i, dims), "data_type": dtype}
                    for i, dims in enumerate(
                        (
                            [2, 16, 8, 16, 16],
                            [16, 8, 3, 3, 3],
                            [2, 8, 8, 16, 16],
                        ),
                        1,
                    )
                }
                group = dict(
                    operation="convolution_dgrad",
                    parameters={
                        **convolution_parameters(),
                        "spatial_rank": 3,
                        "stride": [1, 1, 1],
                        "pre_padding": [1, 1, 1],
                        "post_padding": [1, 1, 1],
                        "dilation": [1, 1, 1],
                    },
                    tensors=list(tensors.values()),
                    source_node_ids=[0],
                    input_uids=[1, 2],
                    output_uids=[3],
                )
                groups = self.graph._expand_execution_pipelines(
                    [group], tensors
                )
                self.assertEqual(
                    [stage["operation"] for stage in groups],
                    ["convolution_dgrad"] * 2,
                )
                self.assertEqual(
                    groups[0]["parameters"]["_dgrad_3d_pipeline_stage"], "pack"
                )
                self.assertEqual(
                    groups[1]["tensors"][1]["strides"], [1152, 384, 128, 8, 1]
                )
                for stage in groups:
                    self.dispatch._kernel_configuration(
                        stage["operation"],
                        stage["parameters"],
                        stage["tensors"],
                        90,
                    )

    def test_fp32_matmul_packing_preserves_views_precision_and_workspace(self):
        for batch, m, n, k in ((32, 512, 512, 512), (4, 513, 515, 518)):
            tensors = {
                i: tensor(i, dims)
                for i, dims in enumerate(
                    ([batch, m, k], [batch, k, n], [batch, m, n]), 1
                )
            }
            group = dict(
                operation="matmul",
                parameters=dict(batch=batch, m=m, n=n, k=k),
                tensors=list(tensors.values()),
                source_node_ids=[0],
                input_uids=[1, 2],
                output_uids=[3],
            )
            groups = self.graph._expand_execution_pipelines([group], tensors)
            tf32_packed = k % 4 == 0 and n % 4 == 0
            self.assertEqual(
                [stage["operation"] for stage in groups],
                ["matmul" if tf32_packed else "transpose", "matmul"],
            )
            self.assertEqual(
                groups[1]["tensors"][1]["strides"],
                [k * n, 1, k] if tf32_packed else [k, 1, batch * k],
            )
            self.assertTrue(
                all(t["data_type"] == "float32" for t in tensors.values())
            )
            _, size = self.graph._workspace_layout(tensors, groups)
            self.assertEqual(size, ((batch * k * n * 4 + 255) // 256) * 256)
            config = self.dispatch._kernel_configuration(
                "matmul", groups[1]["parameters"], groups[1]["tensors"], 90
            )
            if tf32_packed:
                self.assertEqual(config[0], "matmul_tf32_tma_kernel")
                pack = self.dispatch._kernel_configuration(
                    "matmul", groups[0]["parameters"], groups[0]["tensors"], 90
                )
                self.assertEqual(pack[0], "matmul_tf32_pack_b_kernel")
                self.assertTrue(pack[2]["NATIVE_TF32_RNE"])
            else:
                self.assertEqual(config[0], "matmul_strided_kernel")
                self.assertIs(config[2]["INPUT_IS_FLOAT32"], True)
                self.assertIs(config[2]["USE_TF32"], False)
                self.assertEqual(config[2]["B_STRIDE_K"], 1)
                self.assertEqual(config[2]["B_BATCH_STRIDE_5"], k)
            bad_parameters = {**groups[1]["parameters"], "batch": batch + 1}
            with self.assertRaises(ValueError):
                self.dispatch._kernel_configuration(
                    "matmul", bad_parameters, groups[1]["tensors"], 90
                )

    def test_tf32_direct_tma_avoids_packing_only_on_supported_layouts(self):
        from flagdnn_codegen.kernel_registry import select_kernel_candidate

        tuning = importlib.import_module(
            f"{self.compiler.__package__}.dispatch.tuning"
        )
        for architecture, n, alignment, direct in (
            (90, 512, 16, True),
            (80, 512, 16, False),
            (90, 516, 16, False),
            (90, 512, 8, False),
        ):
            with self.subTest(
                architecture=architecture, n=n, alignment=alignment
            ):
                tensors = {
                    i: tensor(i, dims)
                    for i, dims in enumerate(
                        ([32, 512, 512], [32, 512, n], [32, 512, n]), 1
                    )
                }
                tensors[1]["alignment"] = alignment
                group = dict(
                    operation="matmul",
                    parameters=dict(batch=32, m=512, n=n, k=512),
                    tensors=list(tensors.values()),
                    source_node_ids=[0],
                    input_uids=[1, 2],
                    output_uids=[3],
                )
                groups = self.graph._expand_execution_pipelines(
                    [group], tensors, architecture=architecture
                )
                _, workspace = self.graph._workspace_layout(tensors, groups)
                self.assertEqual(len(groups), 1 if direct else 2)
                self.assertEqual(workspace == 0, direct)
                config = self.dispatch._kernel_configuration(
                    "matmul",
                    groups[-1]["parameters"],
                    groups[-1]["tensors"],
                    architecture,
                )
                self.assertEqual(
                    config[0] == "matmul_tf32_tma_direct_kernel", direct
                )
                if not direct:
                    continue
                variants, _, _, table = tuning._prepare_tuning_variants(
                    compiler_path=Path(__file__).resolve().parents[3]
                    / "compiler/flagdnn_codegen/main.py",
                    candidate=select_kernel_candidate("nvidia", "matmul"),
                    operation="matmul",
                    function_name=config[0],
                    parameters=group["parameters"],
                    constants=config[2],
                    default_grid=config[3],
                )
                self.assertEqual(table, "matmul_tf32_tma_direct")
                for _, constants, _, grid in variants:
                    self.assertEqual(512 % constants["BLOCK_M"], 0)
                    self.assertEqual(512 % constants["BLOCK_N"], 0)
                    self.assertEqual(512 % constants["BLOCK_K"], 0)
                    tiles = (
                        32
                        * (512 // constants["BLOCK_M"])
                        * (512 // constants["BLOCK_N"])
                    )
                    self.assertEqual(
                        grid, (min(tiles, constants["PERSISTENT_GRID"]), 1, 1)
                    )

    def test_tf32_tma_boundaries_fallback_and_tuning_grids(self):
        from flagdnn_codegen.kernel_registry import select_kernel_candidate

        tuning = importlib.import_module(
            f"{self.compiler.__package__}.dispatch.tuning"
        )
        for arch, m, n, k, alignment, expected in (
            (90, 512, 512, 512, 16, "matmul_tf32_tma_kernel"),
            (90, 512, 516, 512, 16, "matmul_tf32_tma_kernel"),
            (80, 512, 512, 512, 16, "matmul_strided_kernel"),
            (90, 513, 512, 512, 16, "matmul_strided_kernel"),
            (90, 512, 512, 520, 16, "matmul_strided_kernel"),
            (90, 512, 512, 512, 8, "matmul_strided_kernel"),
        ):
            with self.subTest(arch=arch, m=m, n=n, k=k, alignment=alignment):
                tensors = [
                    tensor(i, dims)
                    for i, dims in enumerate(
                        ([4, m, k], [4, k, n], [4, m, n]), 1
                    )
                ]
                tensors[0]["alignment"] = alignment
                tensors[1]["strides"] = [k * n, 1, k]
                parameters = dict(
                    batch=4, m=m, n=n, k=k, _matmul_tf32_packed=True
                )
                config = self.dispatch._kernel_configuration(
                    "matmul", parameters, tensors, arch
                )
                self.assertEqual(config[0], expected)
                if expected != "matmul_tf32_tma_kernel":
                    self.assertIs(config[2]["USE_TF32"], True)
                    continue
                variants, _, _, table = tuning._prepare_tuning_variants(
                    compiler_path=Path(__file__).resolve().parents[3]
                    / "compiler/flagdnn_codegen/main.py",
                    candidate=select_kernel_candidate("nvidia", "matmul"),
                    operation="matmul",
                    function_name=config[0],
                    parameters=parameters,
                    constants=config[2],
                    default_grid=config[3],
                )
                self.assertEqual(table, "matmul_tf32_tma")
                for _, constants, _, grid in variants:
                    self.assertEqual(m % constants["BLOCK_M"], 0)
                    self.assertEqual(k % constants["BLOCK_K"], 0)
                    tiles = (
                        4
                        * (
                            (m + constants["BLOCK_M"] - 1)
                            // constants["BLOCK_M"]
                        )
                        * (
                            (n + constants["BLOCK_N"] - 1)
                            // constants["BLOCK_N"]
                        )
                    )
                    self.assertEqual(
                        grid, (min(tiles, constants["PERSISTENT_GRID"]), 1, 1)
                    )
                bad = [dict(t) for t in tensors]
                bad[1]["strides"] = [k, 1, 4 * k]
                with self.assertRaises(ValueError):
                    self.dispatch._kernel_configuration(
                        "matmul", parameters, bad, arch
                    )

    def test_matmul_precision_policy_matches_reference_alignment_and_rank(
        self,
    ):
        for architecture in (80, 90):
            for batch_dims, m, n, k, expected in (
                ([16], 32, 64, 128, True),
                ([2], 65, 33, 130, False),
                ([2], 65, 64, 130, False),
                ([2], 65, 33, 128, False),
                ([2], 65, 64, 128, True),
                ([], 32, 64, 128, True),
                ([2, 3], 32, 64, 128, False),
            ):
                with self.subTest(
                    architecture=architecture, dims=batch_dims, m=m, n=n, k=k
                ):
                    tensors = [
                        tensor(i, [*batch_dims, *dims])
                        for i, dims in enumerate(((m, k), (k, n), (m, n)), 1)
                    ]
                    batch = 1
                    for dim in batch_dims:
                        batch *= dim
                    config = self.dispatch._kernel_configuration(
                        "matmul",
                        dict(batch=batch, m=m, n=n, k=k),
                        tensors,
                        architecture,
                    )
                    self.assertIs(config[2]["INPUT_IS_FLOAT32"], True)
                    self.assertIs(config[2]["USE_TF32"], expected)
                    if config[0] == "matmul_strided_kernel":
                        self.assertIs(
                            config[2]["NATIVE_TF32_RNE"], architecture >= 90
                        )
                    self.assertTrue(
                        all(t["data_type"] == "float32" for t in tensors)
                    )

    def test_matmul_packing_does_not_change_small_low_precision_or_broadcast(
        self,
    ):
        for dtype, batch, m, a_batch in (
            ("float16", 32, 512, 32),
            ("bfloat16", 32, 512, 32),
            ("float32", 32, 64, 32),
            ("float32", 1, 512, 1),
            ("float32", 32, 512, 1),
        ):
            tensors = {
                i: {**tensor(i, dims), "data_type": dtype}
                for i, dims in enumerate(
                    ([a_batch, m, 512], [batch, 512, 512], [batch, m, 512]), 1
                )
            }
            group = dict(
                operation="matmul",
                parameters=dict(batch=batch, m=m, n=512, k=512),
                tensors=list(tensors.values()),
                source_node_ids=[0],
                input_uids=[1, 2],
                output_uids=[3],
            )
            self.assertEqual(
                self.graph._expand_execution_pipelines([group], tensors),
                [group],
            )
            self.assertEqual(len(tensors), 3)

    def fused_graph(self):
        x, w, bias = (
            tensor(1, [2, 4, 8, 8]),
            tensor(2, [4, 4, 3, 3]),
            tensor(3, [1, 4, 1, 1]),
        )
        a, b, y = (
            tensor(4, [2, 4, 8, 8], True),
            tensor(5, [2, 4, 8, 8], True),
            tensor(6, [2, 4, 8, 8]),
        )
        tensors = {t["uid"]: t for t in (x, w, bias, a, b, y)}
        nodes = [
            (
                0,
                "convolution_fprop",
                convolution_parameters(),
                [x, w, a],
                [1, 2],
                [4],
            ),
            (1, "add", {"alpha": 1}, [a, bias, b], [4, 3], [5]),
            (2, "relu", {}, [b, y], [5], [6]),
        ]
        return nodes, tensors

    def test_fused_intermediates_do_not_allocate_workspace(self):
        nodes, tensors = self.fused_graph()
        groups = self.graph._lower_execution_groups(nodes, tensors)
        self.assertEqual(len(groups), 1)
        layout, size = self.graph._workspace_layout(tensors, groups)
        self.assertEqual(layout, {})
        self.assertEqual(size, 0)

    def test_live_branch_prevents_fusion_and_keeps_workspace(self):
        nodes, tensors = self.fused_graph()
        tensors[7] = tensor(7, [2, 4, 8, 8])
        nodes.append((3, "relu", {}, [tensors[4], tensors[7]], [4], [7]))
        groups = self.graph._lower_execution_groups(nodes, tensors)
        self.assertEqual(len(groups), 4)
        layout, size = self.graph._workspace_layout(tensors, groups)
        self.assertEqual(set(layout), {4, 5})
        self.assertEqual(size, 4096)

    def test_fp32_stride2_and_backward_preserve_operand_range(self):
        cases = [
            (
                "convolution_fprop",
                [1, 128, 40, 40],
                [256, 128, 3, 3],
                [1, 256, 20, 20],
            ),
            (
                "convolution_fprop",
                [2, 64, 56, 56],
                [128, 64, 3, 3],
                [2, 128, 28, 28],
            ),
            (
                "convolution_dgrad",
                [1, 768, 20, 20],
                [768, 768, 3, 3],
                [1, 768, 40, 40],
            ),
            (
                "convolution_wgrad",
                [1, 768, 20, 20],
                [1, 768, 40, 40],
                [768, 768, 3, 3],
            ),
        ]
        for operation, a, b, c in cases:
            with self.subTest(operation=operation, dimensions=a):
                tensors = {
                    i: tensor(i, dims) for i, dims in enumerate((a, b, c), 1)
                }
                parameters = {**convolution_parameters(), "stride": [2, 2]}
                group = dict(
                    operation=operation,
                    parameters=parameters,
                    tensors=list(tensors.values()),
                    source_node_ids=[0],
                    input_uids=[1, 2],
                    output_uids=[3],
                )
                groups = self.graph._expand_execution_pipelines(
                    [group], tensors
                )
                self.assertGreater(len(groups), 1)
                self.assertEqual(
                    {t["data_type"] for g in groups for t in g["tensors"]},
                    {"float32"},
                )
                workspace, _ = self.graph._workspace_layout(tensors, groups)
                for stage in groups:
                    configuration = self.dispatch._kernel_configuration(
                        stage["operation"],
                        stage["parameters"],
                        stage["tensors"],
                        90,
                    )
                    self.tensor_abi._build_argument_abi(
                        configuration[4],
                        stage["tensors"],
                        stage["parameters"],
                        workspace,
                    )

    def test_fp32_im2col_preserves_operand_range(self):
        x, w, y = (
            tensor(1, [4, 64, 32, 32]),
            tensor(2, [64, 64, 3, 3]),
            tensor(3, [4, 64, 32, 32]),
        )
        tensors = {t["uid"]: t for t in (x, w, y)}
        group = dict(
            operation="convolution_fprop",
            parameters=convolution_parameters(),
            tensors=[x, w, y],
            source_node_ids=[0],
            input_uids=[1, 2],
            output_uids=[3],
        )
        groups = self.graph._expand_execution_pipelines([group], tensors)
        self.assertEqual(
            {t["data_type"] for g in groups for t in g["tensors"]}, {"float32"}
        )
        for stage in groups:
            configuration = self.dispatch._kernel_configuration(
                stage["operation"], stage["parameters"], stage["tensors"], 90
            )
            workspace, _ = self.graph._workspace_layout(tensors, groups)
            self.tensor_abi._build_argument_abi(
                configuration[4],
                stage["tensors"],
                stage["parameters"],
                workspace,
            )

    def test_p5_fp32_layout_is_explicit_and_low_precision_is_unchanged(self):
        for dtype in ("float32", "float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                tensors = {
                    i: {**tensor(i, dims), "data_type": dtype}
                    for i, dims in enumerate(
                        ([1, 128, 40, 40], [256, 128, 3, 3], [1, 256, 20, 20]),
                        1,
                    )
                }
                group = dict(
                    operation="convolution_fprop",
                    parameters={**convolution_parameters(), "stride": [2, 2]},
                    tensors=list(tensors.values()),
                    source_node_ids=[0],
                    input_uids=[1, 2],
                    output_uids=[3],
                )
                groups = self.graph._expand_execution_pipelines(
                    [group], tensors
                )
                self.assertEqual(len(groups), 3)
                configurations = [
                    self.dispatch._kernel_configuration(
                        g["operation"], g["parameters"], g["tensors"], 90
                    )
                    for g in groups
                ]
                columns = groups[0]["tensors"][1]
                self.assertEqual(columns["dimensions"], [1, 1152, 400])
                if dtype == "float32":
                    self.assertEqual(columns["strides"], [460800, 1, 1152])
                    self.assertEqual(
                        configurations[0][0],
                        "conv2d_im2col_nchw_transposed_kernel",
                    )
                    self.assertEqual(configurations[1][2]["B_STRIDE_N"], 1152)
                else:
                    self.assertEqual(columns["strides"], [460800, 400, 1])
                    self.assertEqual(
                        configurations[0][0],
                        "conv2d_im2col_nchw_3x3_stride2_pad1_kernel",
                    )
                    self.assertEqual(configurations[1][2]["B_STRIDE_N"], 1)
                self.assertEqual(configurations[1][2]["SPLITS"], 4)
                columns["strides"] = [460800, 2, 1152]
                with self.assertRaises(ValueError):
                    self.dispatch._kernel_configuration(
                        groups[0]["operation"],
                        groups[0]["parameters"],
                        groups[0]["tensors"],
                        90,
                    )

    def test_fp32_stem_reuses_spatial_kernel_without_workspace(self):
        for dtype in ("float32", "float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                tensors = {
                    i: {**tensor(i, dims), "data_type": dtype}
                    for i, dims in enumerate(
                        ([1, 3, 640, 640], [64, 3, 3, 3], [1, 64, 320, 320]), 1
                    )
                }
                group = dict(
                    operation="convolution_fprop",
                    parameters={
                        **convolution_parameters(),
                        "stride": [2, 2],
                        "n_outputs": 64 * 320 * 320,
                    },
                    tensors=list(tensors.values()),
                    source_node_ids=[0],
                    input_uids=[1, 2],
                    output_uids=[3],
                )
                groups = self.graph._expand_execution_pipelines(
                    [group], tensors
                )
                _, size = self.graph._workspace_layout(tensors, groups)
                if dtype == "float32":
                    self.assertEqual(len(groups), 1)
                    self.assertEqual(size, 0)
                    configuration = self.dispatch._kernel_configuration(
                        groups[0]["operation"],
                        groups[0]["parameters"],
                        groups[0]["tensors"],
                        90,
                    )
                    self.assertEqual(
                        configuration[0], "conv2d_spatial_nchw_kernel"
                    )
                    self.assertEqual(configuration[2]["INPUT_PRECISION"], 1)
                else:
                    self.assertEqual(len(groups), 2)
                    self.assertGreater(size, 0)

    def test_host_tf32_maps_have_variant_local_tiles_and_safe_fallbacks(self):
        maps = importlib.import_module(
            f"{self.compiler.__package__}.codegen.tensor_map"
        )
        for arch, k, n, virtual, enabled in (
            (90, 1024, 1024, False, True),
            (80, 1024, 1024, False, False),
            (100, 1024, 1024, False, False),
            (90, 512, 1024, False, True),
            (90, 256, 1024, False, False),
            (90, 1024, 516, False, False),
            (90, 1024, 1024, True, False),
        ):
            with self.subTest(arch=arch, k=k, n=n, virtual=virtual):
                tensors = {
                    i: tensor(i, dims)
                    for i, dims in enumerate(
                        ([16, 1024, k], [16, k, n], [16, 1024, n]), 1
                    )
                }
                tensors[1]["virtual"] = virtual
                group = dict(
                    operation="matmul",
                    parameters=dict(batch=16, m=1024, n=n, k=k),
                    tensors=list(tensors.values()),
                    source_node_ids=[0],
                    input_uids=[1, 2],
                    output_uids=[3],
                )
                groups = self.graph._expand_execution_pipelines(
                    [group], tensors, architecture=arch
                )
                config = self.dispatch._kernel_configuration(
                    "matmul",
                    groups[-1]["parameters"],
                    groups[-1]["tensors"],
                    arch,
                )
                self.assertEqual(
                    config[0]
                    in {
                        "matmul_tf32_tensor_map_kernel",
                        "matmul_tf32_short_kernel",
                    },
                    enabled,
                )
                if not enabled:
                    continue
                self.assertEqual(len(groups), 1)
                self.assertEqual(
                    self.graph._workspace_layout(tensors, groups)[1], 0
                )
                template = self.tensor_abi._build_argument_abi(
                    config[4], group["tensors"], {}, {}
                )
                signature, abi = maps.resolve_tensor_maps(
                    config[1], template, config[2]
                )
                block_m = 256
                self.assertEqual(
                    signature["a_desc"], f"tensordesc<fp32[{block_m},32]>"
                )
                block_n = 64 if k == 512 else 128
                self.assertEqual(
                    signature["b_desc"], f"tensordesc<fp32[32,{block_n}]>"
                )
                if k == 512:
                    from flagdnn_codegen.kernel_registry import (
                        select_kernel_candidate,
                    )

                    tuning = importlib.import_module(
                        f"{self.compiler.__package__}.dispatch.tuning"
                    )
                    variants, _, _, table = tuning._prepare_tuning_variants(
                        compiler_path=Path(__file__).resolve().parents[3]
                        / "compiler/flagdnn_codegen/main.py",
                        candidate=select_kernel_candidate(
                            "nvidia", "matmul_tf32_short"
                        ),
                        operation="matmul",
                        function_name=config[0],
                        parameters=group["parameters"],
                        constants=config[2],
                        default_grid=config[3],
                    )
                    self.assertEqual(table, "matmul_tf32_short")
                    self.assertEqual({v[1]["SLOTS"] for v in variants}, {3, 4})
                    self.assertTrue(
                        all(v[2]["num_warps"] == 4 for v in variants)
                    )
                    gs, ga = maps.resolve_tensor_maps(
                        config[1], template, config[2], gluon=True
                    )
                    self.assertIn("NVMMASharedLayout", gs["a_desc"])
                    self.assertEqual(ga, abi)
                self.assertEqual(abi[0]["shape"], [16384, k])
                self.assertEqual(abi[1]["shape"], [16 * k, n])
                self.assertEqual(abi[0]["data_type"], "tf32_rne")
                alternate = {**config[2], "BLOCK_M": 128}
                _, second = maps.resolve_tensor_maps(
                    config[1], template, alternate
                )
                self.assertEqual(second[0]["block_shape"], [128, 32])
                self.assertEqual(abi[0]["block_shape"], [block_m, 32])
                self.assertEqual(
                    template[0]["block_shape"], ["BLOCK_M", "BLOCK_K"]
                )
                with self.assertRaises(ValueError):
                    maps.resolve_tensor_maps(
                        config[1], template, {**config[2], "BLOCK_K": 24}
                    )


if __name__ == "__main__":
    unittest.main()
