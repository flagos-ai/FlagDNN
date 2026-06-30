import sys
from pathlib import Path

import pytest
import torch

import flag_dnn

QUICK_MODE = False
MODE_OPTION = (
    "--fg_mode"
    if flag_dnn.vendor_name == "kunlunxin" and torch.__version__ < "2.5"
    else "--mode"
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
            help="run graph tests on quick mode",
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
            help="run graph tests on normal or quick mode",
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
