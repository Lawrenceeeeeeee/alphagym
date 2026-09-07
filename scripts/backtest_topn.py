"""Compatibility entry point; use alphagym.workflows.backtest_topn from Python."""
from alphagym.workflows.backtest_topn import main

if __name__ == "__main__":
    raise SystemExit(main())
