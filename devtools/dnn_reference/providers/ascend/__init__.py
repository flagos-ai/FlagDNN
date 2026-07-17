"""Ascend ACLNN DNN reference provider."""

from .provider import AscendDnnProvider, create_provider

__all__: list[str] = ["AscendDnnProvider", "create_provider"]
