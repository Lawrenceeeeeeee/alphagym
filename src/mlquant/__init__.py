"""Point-in-time A-share factor research toolkit."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlquant.api import Workspace as Workspace
    from mlquant.config import WorkspaceConfig as WorkspaceConfig
    from mlquant.equity_data import DataContractError as DataContractError
    from mlquant.equity_data import EquityDataBundle as EquityDataBundle
    from mlquant.factors.base import FactorContext as FactorContext
    from mlquant.factors.base import FactorDefinition as FactorDefinition
    from mlquant.factors.base import FactorRegistry as FactorRegistry
    from mlquant.factors.base import FactorSpec as FactorSpec
    from mlquant.report_spec import ReportSpec as ReportSpec

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
__version__ = "0.2.0"

_EXPORTS = {
    "Workspace": "mlquant.api", "WorkspaceConfig": "mlquant.config",
    "EquityDataBundle": "mlquant.equity_data", "DataContractError": "mlquant.equity_data",
    "FactorDefinition": "mlquant.factors.base", "FactorRegistry": "mlquant.factors.base",
    "FactorContext": "mlquant.factors.base", "FactorSpec": "mlquant.factors.base",
    "ReportSpec": "mlquant.report_spec",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
