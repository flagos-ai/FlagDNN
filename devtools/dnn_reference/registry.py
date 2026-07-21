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

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Optional, cast

from .interfaces import DnnProvider


class DnnProviderNotImplementedError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderSpec:
    vendor_name: str
    implementation: str
    module_name: str


_PROVIDER_SPECS = {
    "nvidia": ProviderSpec(
        vendor_name="nvidia",
        implementation="cudnn",
        module_name="devtools.dnn_reference.providers.nvidia",
    ),
    "ascend": ProviderSpec(
        vendor_name="ascend",
        implementation="aclnn",
        module_name="devtools.dnn_reference.providers.ascend",
    ),
}


def _selected_vendor(vendor_name: Optional[str]) -> str:
    if vendor_name is not None:
        return vendor_name

    import flag_dnn

    return flag_dnn.vendor_name


def get_provider_spec(vendor_name: Optional[str] = None) -> ProviderSpec:
    selected = _selected_vendor(vendor_name)
    try:
        return _PROVIDER_SPECS[selected]
    except KeyError as exc:
        raise DnnProviderNotImplementedError(
            f"DNN reference provider not implemented for vendor: {selected}"
        ) from exc


def _validate_provider(provider: Any, spec: ProviderSpec) -> DnnProvider:
    if getattr(provider, "vendor_name", None) != spec.vendor_name:
        raise RuntimeError(
            "DNN provider vendor mismatch: "
            f"selected={spec.vendor_name}, "
            f"provider={getattr(provider, 'vendor_name', None)}"
        )
    if getattr(provider, "implementation", None) != spec.implementation:
        raise RuntimeError(
            "DNN provider implementation mismatch: "
            f"expected={spec.implementation}, "
            f"provider={getattr(provider, 'implementation', None)}"
        )
    for method_name in (
        "get_operation",
        "supports",
        "run",
        "prepare",
        "synchronize",
        "close",
    ):
        if not callable(getattr(provider, method_name, None)):
            raise RuntimeError(
                f"DNN provider is missing callable method: {method_name}"
            )
    operation_names = getattr(provider, "operation_names", None)
    if not isinstance(operation_names, tuple):
        raise RuntimeError(
            "DNN provider operation_names must be a tuple of op names"
        )
    return cast(DnnProvider, provider)


def create_provider(vendor_name: Optional[str] = None) -> DnnProvider:
    spec = get_provider_spec(vendor_name)
    module = importlib.import_module(spec.module_name)
    factory = getattr(module, "create_provider")
    return _validate_provider(factory(), spec)
