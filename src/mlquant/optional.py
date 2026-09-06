"""Actionable errors for optional integrations."""
from importlib import import_module


class OptionalDependencyError(ImportError):
    """Install the indicated extra before using this capability."""


def require(module: str, extra: str):
    try:
        return import_module(module)
    except ImportError as error:
        raise OptionalDependencyError(
            f"{module} is required for this operation; install 'mlquant[{extra}]'"
        ) from error
