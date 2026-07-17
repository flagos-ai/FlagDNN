"""Prepared vendor-DNN baselines used by performance tests."""

from .registry import DnnBaseline, create_baseline

__all__: list[str] = ["DnnBaseline", "create_baseline"]
