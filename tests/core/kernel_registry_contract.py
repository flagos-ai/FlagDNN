# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Host-only contracts for portable kernel ownership and source dependencies."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
from flagdnn_codegen.kernel_registry import (
    iter_kernel_candidates,
    resolve_kernel_source,
    select_kernel_candidate,
)


class KernelRegistryContracts(unittest.TestCase):
    def test_registered_entrypoints_exist_in_every_backend_source(self):
        backends = [
            p.name
            for p in (ROOT / "backends").iterdir()
            if (p / "kernels/registry.json").is_file()
        ]
        for backend in backends:
            for candidate in iter_kernel_candidates(backend):
                with self.subTest(
                    backend=backend, operation=candidate.operation
                ):
                    source = resolve_kernel_source(
                        ROOT / "compiler/flagdnn_codegen/main.py", candidate
                    )
                    names = {
                        node.name
                        for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef)
                    }
                    self.assertLessEqual(set(candidate.functions), names)

    def test_common_sources_have_no_vendor_or_runtime_dependency(self):
        for source in (ROOT / "kernels/common").glob("*.py"):
            with self.subTest(source=source.name):
                tree = ast.parse(source.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        self.assertEqual(node.level, 0)
                        modules = [node.module or ""]
                        if node.module == "triton.language.extra":
                            self.assertEqual(
                                [a.name for a in node.names], ["libdevice"]
                            )
                    else:
                        modules = []
                    for module in modules:
                        self.assertTrue(
                            module
                            in {
                                "triton",
                                "triton.language",
                                "triton.language.extra",
                            }
                            or module.split(".")[0] in sys.stdlib_module_names,
                            f"{source.name}: platform dependency {module}",
                        )
                    if isinstance(node, ast.Attribute):
                        self.assertNotIn(
                            node.attr,
                            {
                                "inline_asm_elementwise",
                                "float8e4nv",
                                "autotune",
                                "heuristics",
                                "cuda",
                                "nvidia",
                            },
                        )

    def test_platform_override_does_not_replace_other_backends_fallback(self):
        common = select_kernel_candidate("portable_contract", "causal_conv1d")
        nvidia = select_kernel_candidate("nvidia", "causal_conv1d")
        self.assertEqual(common.ownership, "common")
        self.assertEqual(nvidia.ownership, "platform")
        entry = ROOT / "compiler/flagdnn_codegen/main.py"
        self.assertNotEqual(
            resolve_kernel_source(entry, common),
            resolve_kernel_source(entry, nvidia),
        )


if __name__ == "__main__":
    unittest.main()
