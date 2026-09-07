"""AlphaGYM — an extensible multi-market, multi-asset quant research toolbox."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphagym.api import Workspace as Workspace
    from alphagym.config import WorkspaceConfig as WorkspaceConfig
    from alphagym.equity_data import DataContractError as DataContractError
    from alphagym.equity_data import EquityDataBundle as EquityDataBundle
    from alphagym.factors.base import FactorContext as FactorContext
    from alphagym.factors.base import FactorDefinition as FactorDefinition
    from alphagym.factors.base import FactorRegistry as FactorRegistry
    from alphagym.factors.base import FactorSpec as FactorSpec
    from alphagym.report_spec import ReportSpec as ReportSpec

__all__ = [
    "DataContractError",
    "EquityDataBundle",
    "FactorContext",
    "FactorDefinition",
    "FactorRegistry",
    "FactorSpec",
    "ReportSpec",
    "Workspace",
    "WorkspaceConfig",
]
__version__ = "0.2.1"

_EXPORTS = {
    "Workspace": "alphagym.api", "WorkspaceConfig": "alphagym.config",
    "EquityDataBundle": "alphagym.equity_data", "DataContractError": "alphagym.equity_data",
    "FactorDefinition": "alphagym.factors.base", "FactorRegistry": "alphagym.factors.base",
    "FactorContext": "alphagym.factors.base", "FactorSpec": "alphagym.factors.base",
    "ReportSpec": "alphagym.report_spec",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
