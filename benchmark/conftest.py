import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture()
def cudnn_handle():
    from benchmark.base import get_cudnn

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
