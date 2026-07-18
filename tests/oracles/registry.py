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
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
)

if TYPE_CHECKING:
    import torch


Shape = Tuple[int, ...]
AddCase = Tuple[Shape, Shape]


class DnnOracle(Protocol):
    vendor_name: str
    implementation: str

    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: Union[int, float] = 1,
    ) -> torch.Tensor: ...

    def abs(self, x: torch.Tensor) -> torch.Tensor: ...

    def supports_dtype(self, dtype: torch.dtype) -> bool: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...


class OracleNotImplementedError(RuntimeError):
    pass


@dataclass(frozen=True)
class OracleSpec:
    vendor_name: str
    implementation: str
    module_name: str


_ORACLE_SPECS = {
    "nvidia": OracleSpec(
        vendor_name="nvidia",
        implementation="cudnn",
        module_name="tests.oracles.nvidia",
    ),
    "ascend": OracleSpec(
        vendor_name="ascend",
        implementation="aclnnAdd",
        module_name="tests.oracles.ascend",
    ),
}


def _selected_vendor(vendor_name: Optional[str]) -> str:
    if vendor_name is not None:
        return vendor_name

    import flag_dnn

    return flag_dnn.vendor_name


def get_oracle_spec(vendor_name: Optional[str] = None) -> OracleSpec:
    selected = _selected_vendor(vendor_name)
    try:
        return _ORACLE_SPECS[selected]
    except KeyError as exc:
        raise OracleNotImplementedError(
            f"DNN oracle not implemented for vendor: {selected}"
        ) from exc


def get_add_test_cases(
    vendor_name: Optional[str] = None,
) -> Tuple[AddCase, ...]:
    from tests import consts

    selected = _selected_vendor(vendor_name)
    extras = consts.ADD_EXTRA_CASES_BY_VENDOR.get(selected, ())
    return consts.ADD_COMMON_CASES + extras


def _validate_oracle(oracle: Any, spec: OracleSpec) -> DnnOracle:
    if getattr(oracle, "vendor_name", None) != spec.vendor_name:
        raise RuntimeError(
            "DNN oracle vendor mismatch: "
            f"selected={spec.vendor_name}, "
            f"provider={getattr(oracle, 'vendor_name', None)}"
        )
    if getattr(oracle, "implementation", None) != spec.implementation:
        raise RuntimeError(
            "DNN oracle implementation mismatch: "
            f"expected={spec.implementation}, "
            f"provider={getattr(oracle, 'implementation', None)}"
        )
    for method_name in (
        "add",
        "abs",
        "supports_dtype",
        "synchronize",
        "close",
    ):
        if not callable(getattr(oracle, method_name, None)):
            raise RuntimeError(
                f"DNN oracle is missing callable method: {method_name}"
            )
    return cast(DnnOracle, oracle)


def create_oracle(vendor_name: Optional[str] = None) -> DnnOracle:
    spec = get_oracle_spec(vendor_name)
    module = importlib.import_module(spec.module_name)
    factory = getattr(module, "create_oracle")
    return _validate_oracle(factory(), spec)
