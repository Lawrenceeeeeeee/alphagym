"""Compatibility entry point; use mlquant.workflows.run_equity_v0 from Python."""
from mlquant.workflows.run_equity_v0 import main

if __name__ == "__main__":
    raise SystemExit(main())
