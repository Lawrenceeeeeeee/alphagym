"""Explicit, portable workspace configuration; importing never creates files."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from mlquant.equity_data import DataContractError


def resolve_root(root: str | Path | None = None) -> Path:
    value = root if root is not None else os.environ.get("MLQUANT_DATA_ROOT")
    if value is None or not str(value).strip():
        raise DataContractError("--root or MLQUANT_DATA_ROOT is required")
    from mlquant.storage_io import register_root

    return register_root(Path(value).expanduser().resolve())


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    """Data location. An explicit root takes precedence over the environment."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", resolve_root(self.root))

    @classmethod
    def from_env(cls) -> WorkspaceConfig:
        return cls(resolve_root())

    @property
    def catalog(self) -> Path:
        return self.root / "factor_library" / "catalog.sqlite"
