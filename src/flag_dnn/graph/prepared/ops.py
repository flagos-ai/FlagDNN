from __future__ import annotations

from importlib import import_module

from flag_dnn.graph.prepared import prepare_run_fn

# Import prepared op modules for registration side effects.
for _module_name in ("conv", "pointwise", "sdpa_backward", "sdpa_forward"):
    import_module(f"{__package__}.{_module_name}")

__all__ = ("prepare_run_fn",)
