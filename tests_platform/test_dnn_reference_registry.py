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

import sys
from types import SimpleNamespace

import pytest

from devtools.dnn_reference import registry


class _FakeProvider:
    vendor_name = "ascend"
    implementation = "aclnn"
    display_name = "ACLNN"

    def add(self, x, y, *, alpha=1):
        raise AssertionError("not used")

    def abs(self, x):
        raise AssertionError("not used")

    def prepare_add(self, x, y, *, alpha=1):
        raise AssertionError("not used")

    def supports_dtype(self, dtype):
        return True

    def synchronize(self):
        return None

    def close(self):
        return None


def test_provider_registry_uses_flagdnn_vendor_and_lazy_import(monkeypatch):
    imported = []

    def fake_import(module_name):
        imported.append(module_name)
        return SimpleNamespace(create_provider=_FakeProvider)

    monkeypatch.setitem(
        sys.modules,
        "flag_dnn",
        SimpleNamespace(vendor_name="ascend"),
    )
    monkeypatch.setattr(registry.importlib, "import_module", fake_import)

    provider = registry.create_provider()

    assert provider.vendor_name == "ascend"
    assert provider.implementation == "aclnn"
    assert imported == ["devtools.dnn_reference.providers.ascend"]


def test_provider_registry_reports_unimplemented_vendor():
    with pytest.raises(
        registry.DnnProviderNotImplementedError,
        match="not implemented for vendor: metax",
    ):
        registry.create_provider("metax")


def test_provider_registry_rejects_missing_prepared_interface(monkeypatch):
    provider = _FakeProvider()
    provider.prepare_add = None
    monkeypatch.setattr(
        registry.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(create_provider=lambda: provider),
    )

    with pytest.raises(
        RuntimeError, match="missing callable method: prepare_add"
    ):
        registry.create_provider("ascend")
