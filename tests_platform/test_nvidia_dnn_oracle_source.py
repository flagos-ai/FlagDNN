from pathlib import Path


NVIDIA_ORACLE = (
    Path(__file__).parents[1]
    / "devtools"
    / "dnn_reference"
    / "providers"
    / "nvidia.py"
)
TEST_BASE = Path(__file__).parents[1] / "tests" / "base.py"


def _oracle_method_source(name):
    source = NVIDIA_ORACLE.read_text(encoding="utf-8")
    body = source.split(f"    def {name}(", 1)[1]
    return body.split("\n    def ", 1)[0]


def test_nvidia_oracle_uses_only_cudnn_for_add_reference():
    source = NVIDIA_ORACLE.read_text(encoding="utf-8")

    assert "import cudnn" in source
    assert "cuDNN frontend is unavailable" in source
    assert "graph.add(" in source
    assert "graph.mul(" in source
    assert "skip_unsupported=False" in source
    assert "torch.add(" not in source
    assert "operator.add(" not in source


def test_nvidia_oracle_uses_only_cudnn_for_abs_reference():
    source = _oracle_method_source("abs")

    assert "graph.abs(" in source
    assert '"abs"' in source
    assert "skip_unsupported=False" in source
    assert "torch.abs(" not in source
    assert "operator.abs(" not in source
    assert "x.abs(" not in source


def test_cudnn_executor_retains_opt_in_strict_build_behavior():
    source = TEST_BASE.read_text(encoding="utf-8")

    assert "skip_unsupported=True" in source
    assert "if skip_unsupported:" in source


def test_nvidia_oracle_destroys_handle_on_its_own_device():
    source = NVIDIA_ORACLE.read_text(encoding="utf-8")
    close_body = source.split("    def close(self) -> None:", 1)[1]

    assert "handle = self._handle" in close_body
    assert "self._handle = None" in close_body
    assert "with torch.cuda.device(self._device):" in close_body
    assert "cudnn.destroy_handle(handle)" in close_body


def test_nvidia_oracle_preserves_initialization_error_during_cleanup():
    source = NVIDIA_ORACLE.read_text(encoding="utf-8")
    constructor_body = source.split("    def __init__(self) -> None:", 1)[1]
    constructor_body = constructor_body.split("    def supports_dtype", 1)[0]

    assert "except Exception as exc:" in constructor_body
    assert "except Exception as cleanup_exc:" in constructor_body
    assert "exc.add_note(" in constructor_body
