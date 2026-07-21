# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).parents[1]
PORTABLE_BINARY = REPO_ROOT / "src" / "flag_dnn" / "ops" / "binary.py"
PREPARED_POINTWISE = (
    REPO_ROOT / "src" / "flag_dnn" / "graph" / "prepared" / "pointwise.py"
)
ASCEND_HOOKS = (
    REPO_ROOT
    / "src"
    / "flag_dnn"
    / "runtime"
    / "backend"
    / "_ascend"
    / "hooks.py"
)
BACKEND_INIT = (
    REPO_ROOT / "src" / "flag_dnn" / "runtime" / "backend" / "__init__.py"
)


def test_portable_layers_do_not_import_or_branch_on_ascend():
    sources = (
        PORTABLE_BINARY.read_text(encoding="utf-8"),
        PREPARED_POINTWISE.read_text(encoding="utf-8"),
    )

    for source in sources:
        assert "runtime.backend._ascend" not in source
        assert 'vendor_name == "ascend"' not in source
        assert "vendor_name == 'ascend'" not in source
        assert "torch.npu" not in source
        assert "torch_npu" not in source
        assert "get_backend_hook(" in source


def test_ascend_specializations_are_exported_only_through_private_hooks():
    source = ASCEND_HOOKS.read_text(encoding="utf-8")

    assert "launch_dense_binary" in source
    assert "prepare_dense_binary" in source
    assert "__all__" in source


def _isolated_get_backend_hook(import_module):
    tree = ast.parse(BACKEND_INIT.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_backend_hook"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    namespace = {
        "__name__": "flag_dnn.runtime.backend",
        "backend_hooks_modules": {},
        "get_vendor_info": lambda: SimpleNamespace(vendor_name="nvidia"),
        "importlib": SimpleNamespace(import_module=import_module),
    }
    exec(compile(module, str(BACKEND_INIT), "exec"), namespace)
    return namespace["get_backend_hook"]


def test_backend_hook_modules_are_cached_independently_by_vendor():
    calls = []

    def ascend_hook():
        return None

    ascend_module = SimpleNamespace(example_hook=ascend_hook)

    def fake_import(module_name):
        calls.append(module_name)
        if module_name.endswith("._ascend.hooks"):
            return ascend_module
        error = ModuleNotFoundError(f"No module named {module_name!r}")
        error.name = module_name
        raise error

    get_backend_hook = _isolated_get_backend_hook(fake_import)

    assert get_backend_hook("example_hook", "nvidia") is None
    assert get_backend_hook("example_hook", "nvidia") is None
    assert get_backend_hook("example_hook", "ascend") is ascend_hook
    assert get_backend_hook("example_hook", "ascend") is ascend_hook
    assert calls == [
        "flag_dnn.runtime.backend._nvidia.hooks",
        "flag_dnn.runtime.backend._ascend.hooks",
    ]


def test_functional_add_and_abs_tests_only_express_operator_semantics():
    sources = (
        (REPO_ROOT / "tests" / "test_add.py").read_text(encoding="utf-8"),
        (REPO_ROOT / "tests" / "test_abs.py").read_text(encoding="utf-8"),
    )
    forbidden = (
        "aclnn",
        "cudnn",
        "torch_npu",
        "torch.cuda",
        "torch.npu",
        "ctypes",
        "benchmark",
    )

    for source in sources:
        assert "dnn_reference.run(" in source
        assert all(token not in source.lower() for token in forbidden)
