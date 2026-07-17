import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests import consts
from tests.oracles import registry


class FakeOracle:
    def __init__(self, vendor_name, implementation):
        self.vendor_name = vendor_name
        self.implementation = implementation

    def add(self, x, y, *, alpha=1):
        raise AssertionError("not used by registry tests")

    def abs(self, x):
        raise AssertionError("not used by registry tests")

    def supports_dtype(self, dtype):
        return True

    def synchronize(self):
        return None

    def close(self):
        return None


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
    "flag_dnn",
    "torch_npu",
    "cudnn",
    "tests.oracles.nvidia",
    "tests.oracles.ascend",
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


def test_create_oracle_uses_flagdnn_selected_vendor(monkeypatch):
    imported = []

    def fake_import(module_name):
        imported.append(module_name)
        return SimpleNamespace(
            create_oracle=lambda: FakeOracle("ascend", "aclnnAdd")
        )

    monkeypatch.setitem(
        sys.modules,
        "flag_dnn",
        SimpleNamespace(vendor_name="ascend"),
    )
    monkeypatch.setattr(registry.importlib, "import_module", fake_import)

    oracle = registry.create_oracle()

    assert imported == ["tests.oracles.ascend"]
    assert oracle.vendor_name == "ascend"
    assert oracle.implementation == "aclnnAdd"


@pytest.mark.parametrize(
    ("actual_vendor", "actual_implementation", "message"),
    (
        ("nvidia", "aclnnAdd", "vendor mismatch"),
        ("ascend", "torch.add", "implementation mismatch"),
    ),
)
def test_create_oracle_rejects_provider_identity_mismatch(
    monkeypatch, actual_vendor, actual_implementation, message
):
    module = SimpleNamespace(
        create_oracle=lambda: FakeOracle(actual_vendor, actual_implementation)
    )
    monkeypatch.setattr(
        registry.importlib, "import_module", lambda module_name: module
    )

    with pytest.raises(RuntimeError, match=message):
        registry.create_oracle("ascend")


def test_create_oracle_rejects_provider_missing_abs(monkeypatch):
    class MissingAbsOracle:
        vendor_name = "ascend"
        implementation = "aclnnAdd"

        def add(self, x, y, *, alpha=1):
            raise AssertionError("not used by registry tests")

        def supports_dtype(self, dtype):
            return True

        def synchronize(self):
            return None

        def close(self):
            return None

    module = SimpleNamespace(create_oracle=MissingAbsOracle)
    monkeypatch.setattr(
        registry.importlib, "import_module", lambda module_name: module
    )

    with pytest.raises(RuntimeError, match="missing callable method: abs"):
        registry.create_oracle("ascend")


def test_registered_provider_setup_error_is_not_hidden(monkeypatch):
    def fail_factory():
        raise RuntimeError("CANN is unavailable")

    monkeypatch.setattr(
        registry.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(create_oracle=fail_factory),
    )

    with pytest.raises(RuntimeError, match="CANN is unavailable"):
        registry.create_oracle("ascend")


def test_unregistered_vendor_has_explicit_error():
    with pytest.raises(
        registry.OracleNotImplementedError,
        match="DNN oracle not implemented for vendor: metax",
    ):
        registry.create_oracle("metax")
