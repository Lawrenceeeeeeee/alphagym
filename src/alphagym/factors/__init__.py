from __future__ import annotations

from alphagym.factors.base import (
    FactorContext,
    FactorDefinition,
    FactorRegistry,
    FactorSpec,
    FieldDefinition,
)


def __getattr__(name: str):
    if name == "REGISTRY":
        from alphagym.factors.library import REGISTRY

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
