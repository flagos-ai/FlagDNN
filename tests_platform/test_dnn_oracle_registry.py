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

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests import consts
from tests.oracles import registry


def test_importing_registry_does_not_load_vendor_modules():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        DNN_VENDOR="ascend",
        FLAGGEMS_DB_URL="sqlite:///:memory:",
    )
    code = f"""
import sys

sys.path.insert(0, {str(repo_root)!r})
import tests.oracles.registry

forbidden_prefixes = (
    "cudnn",
    "devtools.dnn_reference.providers.nvidia",
    "devtools.dnn_reference.providers.ascend",
)
loaded = sorted(
    module_name
    for module_name in sys.modules
    if any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in forbidden_prefixes
    )
)
if loaded:
    raise AssertionError("eager imports: " + ", ".join(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_add_case_matrix_is_common_plus_vendor_extras():
    common = consts.ADD_COMMON_CASES
    ascend = registry.get_add_test_cases("ascend")
    nvidia = registry.get_add_test_cases("nvidia")

    assert nvidia == common
    assert ascend == common + consts.ADD_EXTRA_CASES_BY_VENDOR["ascend"]
    assert ((2, 3, 17), (1, 1, 17)) in ascend
    assert ((2, 3, 17), (1, 1, 17)) not in nvidia


def test_create_oracle_is_a_thin_shared_provider_alias(monkeypatch):
    provider = object()
    requested_vendors = []

    def fake_create_provider(vendor_name=None):
        requested_vendors.append(vendor_name)
        return provider

    monkeypatch.setattr(registry, "create_provider", fake_create_provider)

    oracle = registry.create_oracle()

    assert oracle is provider
    assert requested_vendors == [None]


def test_registered_provider_setup_error_is_not_hidden(monkeypatch):
    def fail_factory(vendor_name=None):
        raise RuntimeError("CANN is unavailable")

    monkeypatch.setattr(registry, "create_provider", fail_factory)

    with pytest.raises(RuntimeError, match="CANN is unavailable"):
        registry.create_oracle("ascend")


def test_unregistered_vendor_has_explicit_error():
    with pytest.raises(
        registry.OracleNotImplementedError,
        match="DNN reference provider not implemented for vendor: metax",
    ):
        registry.create_oracle("metax")
