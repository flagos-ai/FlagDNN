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

"""Regression tests for runtime vendor detection."""

from types import SimpleNamespace
import time

import torch

from flag_dnn.runtime import backend
from flag_dnn.runtime.backend.device import DeviceDetector
from flag_dnn.runtime.common import vendors


def _uninitialized_detector():
    detector = object.__new__(DeviceDetector)
    detector.vendor_list = vendors.get_all_vendors().keys()
    return detector


def test_explicit_vendor_takes_precedence(monkeypatch):
    detector = _uninitialized_detector()
    selected = object()
    requested = []

    monkeypatch.setenv("DNN_VENDOR", "ascend")
    monkeypatch.setattr(
        backend,
        "get_vendor_info",
        lambda vendor_name: requested.append(vendor_name) or selected,
    )
    monkeypatch.setattr(
        detector,
        "_get_vendor_from_quick_cmd",
        lambda: "ascend",
    )

    assert detector.get_vendor("nvidia") is selected
    assert requested == ["nvidia"]


def test_quick_detection_skips_unavailable_torch_device(monkeypatch):
    detector = _uninitialized_detector()
    unavailable_npu = SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 1,
    )
    monkeypatch.setattr(torch, "npu", unavailable_npu, raising=False)

    assert detector._get_vendor_from_quick_cmd() is None


def test_system_detection_is_ordered_and_bounded(monkeypatch):
    detector = _uninitialized_detector()
    first = SimpleNamespace(
        vendor_name="first",
        device_name="first_device",
        device_query_cmd="first-smi info",
    )
    second = SimpleNamespace(
        vendor_name="second",
        device_name="second_device",
        device_query_cmd="second-smi info",
    )
    timeouts = []

    def fake_run(args, *, capture_output, text, timeout=None):
        del capture_output, text
        timeouts.append(timeout)
        if args[0] == "first-smi":
            time.sleep(0.02)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(backend, "get_vendor_infos", lambda: [first, second])
    monkeypatch.setattr(
        "flag_dnn.runtime.backend.device.subprocess.run",
        fake_run,
    )

    assert detector._get_vendor_from_sys() is first
    assert timeouts
    assert all(timeout is not None and timeout > 0 for timeout in timeouts)
