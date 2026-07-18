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

import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import flag_dnn
from flag_dnn import runtime
from flag_dnn.runtime import common


POINTWISE_NAMES = (
    "pow",
    "tan",
    "tanh",
    "sigmoid",
    "sigmoid_backward",
)
NVIDIA_OVERRIDE_NAMES = POINTWISE_NAMES + ("reduction",)


def _generic_source(name):
    root = Path(flag_dnn.__file__).resolve().parent
    return (root / "ops" / f"{name}.py").read_text()


def test_nvidia_common_vendor_override_is_applied(monkeypatch):
    def replacement(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(runtime.device, "vendor", common.vendors.NVIDIA)
    monkeypatch.setattr(
        runtime.backend,
        "get_current_device_extend_op",
        lambda vendor_name: [("portable_probe", replacement)],
    )
    monkeypatch.setattr(
        runtime.backend,
        "BackendArchEvent",
        lambda: SimpleNamespace(has_arch=False),
    )
    namespace = {}

    runtime.replace_customized_ops(namespace)

    assert namespace["portable_probe"] is replacement


def test_nvidia_backend_exports_are_exact():
    nvidia_ops = importlib.import_module(
        "flag_dnn.runtime.backend._nvidia.ops"
    )
    public = {
        name
        for name, value in vars(nvidia_ops).items()
        if inspect.isfunction(value) and not name.startswith("_")
    }
    assert tuple(nvidia_ops.__all__) == NVIDIA_OVERRIDE_NAMES
    assert public == set(NVIDIA_OVERRIDE_NAMES)


@pytest.mark.parametrize("op_name", NVIDIA_OVERRIDE_NAMES)
def test_nvidia_and_generic_signatures_match(op_name):
    generic = importlib.import_module(f"flag_dnn.ops.{op_name}")
    nvidia = importlib.import_module(
        f"flag_dnn.runtime.backend._nvidia.ops.{op_name}"
    )
    assert inspect.signature(getattr(nvidia, op_name)) == (
        inspect.signature(getattr(generic, op_name))
    )


def test_generic_unary_has_no_direct_cuda_libdevice_import():
    assert "triton.language.extra.cuda" not in _generic_source("unary")


def test_generic_pow_has_no_direct_cuda_libdevice_import():
    assert "triton.language.extra.cuda" not in _generic_source("pow")


def test_generic_tanh_has_no_direct_cuda_libdevice_import():
    assert "triton.language.extra.cuda" not in _generic_source("tanh")


def test_generic_sigmoid_has_no_direct_cuda_libdevice_import():
    assert "triton.language.extra.cuda" not in _generic_source("sigmoid")


def test_generic_sigmoid_backward_has_no_direct_cuda_libdevice_import():
    assert "triton.language.extra.cuda" not in _generic_source(
        "sigmoid_backward"
    )


def test_generic_reduction_uses_selected_tune_config():
    source = _generic_source("reduction")
    assert source.count('runtime.get_tuned_config("reduction")') == 1
    assert "_REDUCE_CONFIGS = [" not in source


def test_generic_reduction_has_no_sum_or_mean_dependencies():
    source = _generic_source("reduction")
    assert "flag_dnn.ops.sum" not in source
    assert "flag_dnn.ops.mean" not in source
    assert "flag_dnn.ops.abs" not in source
    assert "flag_dnn.ops.mul" not in source
    assert "flag_dnn.ops.sqrt" not in source


def test_generic_reduction_has_no_prod_dependency():
    source = _generic_source("reduction")
    assert "flag_dnn.ops.prod" not in source
    assert "def _prod_reduce(" not in source


def test_generic_reduction_has_no_fp64_kernel_path():
    source = _generic_source("reduction")
    assert "tl.float64" not in source
    assert "IS_FP64" not in source


def test_nvidia_fresh_process_uses_backend_reduction():
    repo_root = Path(flag_dnn.__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        DNN_VENDOR="nvidia",
        FLAGGEMS_DB_URL="sqlite:///:memory:",
    )
    code = """
import flag_dnn

public_fn = flag_dnn.reduction
eager_fn = getattr(public_fn, "__flagdnn_eager_fn__", public_fn)
assert eager_fn.__module__ == "flag_dnn.runtime.backend._nvidia.ops.reduction"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    flag_dnn.vendor_name != "ascend",
    reason="module identity is asserted for the Ascend route",
)
@pytest.mark.parametrize("op_name", POINTWISE_NAMES)
def test_ascend_uses_generic_pointwise_function(op_name):
    public_fn = getattr(flag_dnn, op_name)
    eager_fn = getattr(public_fn, "__flagdnn_eager_fn__", public_fn)
    assert eager_fn.__module__ == f"flag_dnn.ops.{op_name}"


@pytest.mark.skipif(
    flag_dnn.vendor_name != "ascend",
    reason="module identity is asserted for the Ascend route",
)
def test_ascend_uses_generic_reduction_function():
    public_fn = flag_dnn.reduction
    eager_fn = getattr(public_fn, "__flagdnn_eager_fn__", public_fn)
    assert eager_fn.__module__ == "flag_dnn.ops.reduction"


def _call_pointwise(op_name, dtype):
    x = torch.ones(8, dtype=dtype)
    if op_name == "pow":
        return flag_dnn.pow(x, x)
    if op_name == "sigmoid_backward":
        return flag_dnn.sigmoid_backward(x, x)
    return getattr(flag_dnn, op_name)(x)


@pytest.mark.skipif(
    flag_dnn.vendor_name != "ascend",
    reason="portable validation is asserted for the Ascend route",
)
@pytest.mark.parametrize("op_name", POINTWISE_NAMES)
def test_ascend_generic_rejects_fp64_before_launch(op_name):
    with pytest.raises(NotImplementedError, match="float64"):
        _call_pointwise(op_name, torch.float64)


@pytest.mark.skipif(
    flag_dnn.vendor_name != "ascend",
    reason="portable validation is asserted for the Ascend route",
)
@pytest.mark.parametrize("op_name", POINTWISE_NAMES)
def test_ascend_generic_rejects_cpu_before_launch(op_name):
    with pytest.raises(RuntimeError, match="npu"):
        _call_pointwise(op_name, torch.float32)
