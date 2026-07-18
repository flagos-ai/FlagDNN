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
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    # Import cuDNN frontend before torch so it binds the toolchain libstdc++.
    importlib.import_module("cudnn")
except ImportError:
    pass

torch = importlib.import_module("torch")
flag_dnn = importlib.import_module("flag_dnn")

QUICK_MODE = False
MODE_OPTION = (
    "--fg_mode"
    if flag_dnn.vendor_name == "kunlunxin" and torch.__version__ < "2.5"
    else "--mode"
)


def _getoption(config, name, default=None):
    try:
        return config.getoption(name)
    except (AttributeError, ValueError):
        return default


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--quick",
            action="store_true",
            help="run tests on quick mode",
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            MODE_OPTION,
            action="store",
            default="normal",
            required=False,
            choices=["normal", "quick"],
            help="run tests on normal or quick mode",
        )
    except ValueError:
        pass


def pytest_configure(config):
    global QUICK_MODE

    QUICK_MODE = _getoption(config, "--quick", False) is True
    QUICK_MODE = QUICK_MODE or _getoption(config, MODE_OPTION) == "quick"


@pytest.fixture()
def cudnn_handle():
    from tests.base import get_cudnn

    cudnn = get_cudnn()
    import torch

    try:
        cudnn.backend_version()
    except Exception as exc:
        pytest.skip(f"cuDNN backend is not available: {exc}")

    handle = cudnn.create_handle()
    cudnn.set_stream(
        handle=handle,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    try:
        yield handle
    finally:
        cudnn.destroy_handle(handle)


@pytest.fixture(scope="module")
def dnn_oracle():
    from tests.oracles import OracleNotImplementedError, create_oracle

    try:
        oracle = create_oracle()
    except OracleNotImplementedError as exc:
        pytest.skip(str(exc))

    try:
        yield oracle
    finally:
        oracle.close()
