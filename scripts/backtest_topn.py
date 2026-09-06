"""Compatibility entry point; use mlquant.workflows.backtest_topn from Python."""
from mlquant.workflows.backtest_topn import main

if __name__ == "__main__":
    raise SystemExit(main())
