from __future__ import annotations

from typing import Any, Iterable

import torch

from flag_dnn import runtime


def is_runtime_device_tensor(value: Any) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.device.type == runtime.device.name
    )


def has_runtime_device_tensor(values: Iterable[Any]) -> bool:
    return any(is_runtime_device_tensor(value) for value in values)


def synchronize_runtime_device(values: Iterable[Any]) -> None:
    if not has_runtime_device_tensor(values):
        return
    synchronize_current_runtime_device()


def create_runtime_device_event(*, enable_timing: bool = True) -> Any:
    event_cls = getattr(runtime.torch_device_fn, "Event", None)
    if event_cls is None:
        return None
    return event_cls(enable_timing=enable_timing)


def synchronize_current_runtime_device() -> None:
    synchronize = getattr(runtime.torch_device_fn, "synchronize", None)
    if synchronize is not None:
        synchronize()
