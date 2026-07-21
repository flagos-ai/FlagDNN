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

from pathlib import Path

PROVIDERS = (
    Path(__file__).parents[1] / "devtools" / "dnn_reference" / "providers"
)
NVIDIA_PROVIDER = PROVIDERS / "nvidia.py"
NVIDIA_COMMON = PROVIDERS / "nvidia_ops" / "common.py"
NVIDIA_ADD = PROVIDERS / "nvidia_ops" / "add.py"
NVIDIA_ABS = PROVIDERS / "nvidia_ops" / "abs.py"
TEST_BASE = Path(__file__).parents[1] / "tests" / "base.py"


def test_nvidia_oracle_uses_only_cudnn_for_add_reference():
    common_source = NVIDIA_COMMON.read_text(encoding="utf-8")
    source = NVIDIA_ADD.read_text(encoding="utf-8")

    assert "import cudnn" in common_source
    assert "cuDNN frontend is unavailable" in common_source
    assert "graph.add(" in source
    assert "graph.mul(" in source
    assert "torch.add(" not in source
    assert "operator.add(" not in source


def test_nvidia_oracle_uses_only_cudnn_for_abs_reference():
    source = NVIDIA_ABS.read_text(encoding="utf-8")

    assert "graph.abs(" in source
    assert '"abs"' in source
    assert "torch.abs(" not in source
    assert "operator.abs(" not in source
    assert "x.abs(" not in source


def test_cudnn_executor_retains_opt_in_strict_build_behavior():
    source = TEST_BASE.read_text(encoding="utf-8")

    assert "skip_unsupported=True" in source
    assert "if skip_unsupported:" in source


def test_nvidia_oracle_destroys_handle_on_its_own_device():
    source = NVIDIA_COMMON.read_text(encoding="utf-8")
    close_body = source.split("    def close(self) -> None:", 1)[1]

    assert "handle = self.handle" in close_body
    assert "self.handle = None" in close_body
    assert "with torch.cuda.device(self.device):" in close_body
    assert "cudnn.destroy_handle(handle)" in close_body


def test_nvidia_oracle_preserves_initialization_error_during_cleanup():
    source = NVIDIA_COMMON.read_text(encoding="utf-8")
    constructor_body = source.split("    def __init__(self) -> None:", 1)[1]
    constructor_body = constructor_body.split("    def activate_stream", 1)[0]

    assert "except Exception as exc:" in constructor_body
    assert "except Exception as cleanup_exc:" in constructor_body
    assert "exc.add_note(" in constructor_body


def test_nvidia_provider_only_registers_independent_operations():
    source = NVIDIA_PROVIDER.read_text(encoding="utf-8")

    assert "NvidiaAddOperation(self._context)" in source
    assert "NvidiaAbsOperation(self._context)" in source
    assert "graph.add(" not in source
    assert "graph.abs(" not in source
