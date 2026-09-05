from __future__ import annotations

from mlquant.factors.base import (
    FactorContext,
    FactorDefinition,
    FactorRegistry,
    FactorSpec,
    FieldDefinition,
)


def __getattr__(name: str):
    if name == "REGISTRY":
        from mlquant.factors.library import REGISTRY

        return REGISTRY
    raise AttributeError(name)


__all__ = [
    "REGISTRY",
    "FactorContext",
    "FactorDefinition",
    "FactorRegistry",
    "FactorSpec",
    "FieldDefinition",
]
