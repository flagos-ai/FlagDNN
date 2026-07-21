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

from types import SimpleNamespace

import pytest

import flag_dnn
from flag_dnn.runtime import backend
from flag_dnn.runtime.backend.device import DeviceDetector
from flag_dnn.runtime.configloader import ConfigLoader


def test_device_detector_can_retry_after_failed_initialization(monkeypatch):
    attempts = 0
    info = SimpleNamespace(
        vendor_name="ascend",
        device_name="npu",
        dispatch_key="PrivateUse1",
    )

    def get_vendor(self, vendor_name=None):
        del self, vendor_name
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient detection failure")
        return info

    monkeypatch.setattr(DeviceDetector, "_instance", None)
    monkeypatch.setattr(DeviceDetector, "get_vendor", get_vendor)
    monkeypatch.setattr(
        backend,
        "gen_torch_device_object",
        lambda vendor_name=None: SimpleNamespace(device_count=lambda: 1),
    )

    with pytest.raises(RuntimeError, match="transient"):
        DeviceDetector()
    assert DeviceDetector._instance is None

    detector = DeviceDetector()
    assert detector.initialized
    assert detector.vendor_name == "ascend"


def test_config_loader_can_retry_after_failed_initialization(monkeypatch):
    attempts = 0
    fake_device = SimpleNamespace(vendor_name="ascend")

    def vendor_config(self):
        del self
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient config failure")
        return {}

    monkeypatch.setattr(ConfigLoader, "_instance", None)
    monkeypatch.setattr(
        "flag_dnn.runtime.configloader.DeviceDetector", lambda: fake_device
    )
    monkeypatch.setattr(ConfigLoader, "get_vendor_tune_config", vendor_config)
    monkeypatch.setattr(
        ConfigLoader, "get_default_tune_config", lambda self: {}
    )
    monkeypatch.setattr(
        ConfigLoader, "get_vendor_heuristics_config", lambda self: {}
    )
    monkeypatch.setattr(
        ConfigLoader, "get_default_heuristics_config", lambda self: {}
    )
    monkeypatch.setattr(
        backend, "BackendArchEvent", lambda: SimpleNamespace(has_arch=False)
    )

    with pytest.raises(RuntimeError, match="transient"):
        ConfigLoader()
    assert ConfigLoader._instance is None

    loader = ConfigLoader()
    assert loader.initialized
    assert loader.vendor_primitive_yaml_config == {}


def test_backend_arch_event_can_retry_after_failed_initialization(monkeypatch):
    attempts = 0

    def get_arch(self, device=0):
        del self, device
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient arch failure")
        return None

    monkeypatch.setattr(backend.BackendArchEvent, "_instance", None)
    monkeypatch.setattr(backend.BackendArchEvent, "_initialized", False)
    monkeypatch.setattr(backend.BackendArchEvent, "get_arch", get_arch)

    with pytest.raises(RuntimeError, match="transient"):
        backend.BackendArchEvent()
    assert backend.BackendArchEvent._instance is None
    assert not backend.BackendArchEvent._initialized

    event = backend.BackendArchEvent()
    assert backend.BackendArchEvent._initialized
    assert event.arch is None


def test_torch_version_comparison_is_semantic():
    assert flag_dnn._torch_version_at_least("2.5", current="2.10.0")
    assert flag_dnn._torch_version_at_least("2.5", current="2.5.0+cu124")
    assert not flag_dnn._torch_version_at_least("2.5", current="2.4.1")
