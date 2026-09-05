"""Top-N equal-weight portfolio backtest with real trading costs.

Drives a single persistent ``CashEquityLedger`` (佣金 3bp、印花税、过户费、T+1、
整手、停牌/涨跌停拒绝、滑点) with monthly rebalanced top-N targets built from a
report's combination scores. Prices are 后复权 (open for execution, close for
marks), boolean limit flags from equity/status.parquet are mapped to blocker
price levels, corporate actions are folded into adjusted prices rather than
booked as cash. Outputs monthly NAV, trades, cost attribution and per-period
performance.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mlquant.account import CashEquityLedger, FeeSchedule, next_open_date
from mlquant.combine import combine_scores, factor_weights
from mlquant.ml_composite import frozen_composite, load_pit_context, neutralize_wide
from mlquant.report_engine import build_cross_section
from mlquant.report_spec import parse_spec
from mlquant.research import zscore


def _load_market_data(
    root: Path, member_symbols: set[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calendar = pd.read_parquet(root / "equity" / "calendar.parquet")
    daily = pd.read_parquet(
        root / "equity" / "daily.parquet",
        columns=["trade_date", "symbol", "open", "close"],
    )
    if member_symbols is not None:
        daily = daily[daily["symbol"].isin(member_symbols)]
    adjustments = pd.read_parquet(
        root / "equity" / "adjustments.parquet",
        columns=["trade_date", "symbol", "adjust_factor"],
    )
    if member_symbols is not None:
        adjustments = adjustments[adjustments["symbol"].isin(member_symbols)]
    for frame in (daily, adjustments, calendar):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    daily = pd.merge_asof(
        daily.sort_values(["trade_date", "symbol"]),
        adjustments.sort_values(["trade_date", "symbol"]),
        on="trade_date", by="symbol", direction="backward",
    )
    daily["adjust_factor"] = daily["adjust_factor"].fillna(1.0)
    daily["adj_open"] = daily["open"] * daily["adjust_factor"]
    daily["adj_close"] = daily["close"] * daily["adjust_factor"]
    status = pd.read_parquet(
        root / "equity" / "status.parquet",
        columns=["trade_date", "symbol", "is_suspended", "limit_up", "limit_down"],
    )
    status["trade_date"] = pd.to_datetime(status["trade_date"]).dt.normalize()
    return calendar, daily.sort_values(["trade_date", "symbol"]).reset_index(drop=True), status


def _scores_by_month(
    root: Path, report_row: sqlite3.Row, method: str,
) -> dict[pd.Timestamp, pd.Series]:
    rep = Path(report_row["path"])
    spec = parse_spec(json.loads(report_row["spec_json"]))
    manifest = json.loads((rep / "manifest.json").read_text(encoding="utf-8"))
    combo = json.loads((rep / "combo.json").read_text(encoding="utf-8"))
    factor_ids = [f["factor_id"] for f in manifest["factors"]]
    panel_path = root / "factor_library" / "generated_panels" / f"{manifest['run_id']}.parquet"
    wide, forward, z, ic, fm = build_cross_section(panel_path, factor_ids)
    months = sorted(wide.index.get_level_values("signal_date").unique())
    splits = spec.resolved_splits()
    selection_months = [m for m in months if m <= splits["validation"][1]]
    sel = combo["selection"]
    selected = [str(s) for s in sel["selected_factors"]]
    signs = pd.Series({str(k): float(v) for k, v in sel["signs"].items()})
    result: dict[pd.Timestamp, pd.Series] = {}
    if method.startswith("ml_"):
        cfg = sel["ml"]["config"]
        if cfg["feature_mode"] == "neutral":
            industries, cap = load_pit_context(root)
            feats = neutralize_wide(wide[selected], industries, cap)
        else:
            feats = wide[selected].groupby(level="signal_date", observed=True).transform(zscore)
        pred, _meta = frozen_composite(
            feats, forward, splits, method[3:], label_mode=cfg["label_mode"]
        )
        for m in months:
            result[m] = pred.loc[m].dropna()
    else:
        history = (
            fm.loc[selection_months, selected] if method == "factor_return_decay"
            else ic.loc[selection_months, selected]
        )
        weights = factor_weights(history, method)
        signed = z[selected].mul(signs.reindex(selected).fillna(1.0), axis=1)
        for m in months:
            result[m] = combine_scores(signed.loc[m], weights)
    return result


def _execution_market(
    open_pivot: pd.DataFrame, status: pd.DataFrame, execution: pd.Timestamp,
) -> pd.DataFrame:
    market = open_pivot.loc[execution].rename("open").reset_index()
    day_status = status[status["trade_date"] == execution]
    if not day_status.empty:
        market = market.merge(
            day_status[["symbol", "is_suspended", "limit_up", "limit_down"]],
            on="symbol", how="left",
        )
    for column in ("is_suspended", "limit_up", "limit_down"):
        if column not in market:
            market[column] = np.nan
    # boolean limit flags -> blocker price levels (the ledger compares prices)
    market["limit_up"] = market["limit_up"].where(
        ~market["limit_up"].fillna(False).astype(bool), 0.0
    )
    market["limit_down"] = market["limit_down"].where(
        ~market["limit_down"].fillna(False).astype(bool), np.inf
    )
    return market


def _stats(returns: pd.Series) -> dict[str, float]:
    r = returns.dropna()
    if len(r) < 2:
        return {}
    ann = float((1 + r).prod() ** (12 / len(r)) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(12))
    wealth = (1 + r).cumprod()
    return {
        "annualized": ann,
        "sharpe": ann / vol if vol else np.nan,
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
        "win_rate": float((r > 0).mean()),
        "months": len(r),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--method", default="ml_lasso")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--smooth-months", type=int, default=1,
                        help="分数平滑月数（>1 时用滚动均值压低换手）")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--zero-cost", action="store_true", help="无手续费无滑点（成本拆解用）")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root)
    conn = sqlite3.connect(root / "factor_library" / "catalog.sqlite")
    conn.row_factory = sqlite3.Row
    report_row = conn.execute(
        "SELECT report_id, path, spec_json, run_id FROM report WHERE report_id=?", (args.report_id,)
    ).fetchone()
    if report_row is None:
        raise SystemExit(f"report not found: {args.report_id}")
    index_code = parse_spec(json.loads(report_row["spec_json"])).universe.index_code
    scores_by_month = _scores_by_month(root, report_row, args.method)
    if args.smooth_months > 1:
        score_frame = pd.DataFrame(scores_by_month).T.sort_index()
        score_frame = score_frame.rolling(args.smooth_months, min_periods=1).mean()
        scores_by_month = {
            pd.Timestamp(month): row.dropna() for month, row in score_frame.iterrows()
        }
    member_symbols = None
    if index_code != "ALL_A":
        members = pd.read_parquet(root / "equity" / "index_members.parquet")
        member_symbols = set(members.loc[members["index_code"] == index_code, "symbol"].unique())
    calendar, daily, status = _load_market_data(root, member_symbols)
    open_pivot = daily.pivot(index="trade_date", columns="symbol", values="adj_open").sort_index()
    close_pivot = daily.pivot(index="trade_date", columns="symbol", values="adj_close").sort_index()

    fees = FeeSchedule(0.0, 0.0, False) if args.zero_cost else FeeSchedule()
    slippage = 0.0 if args.zero_cost else float(args.slippage_bps)
    ledger = CashEquityLedger(initial_cash=args.initial_cash, slippage_bps=slippage, fees=fees)
    nav_rows: list[dict[str, object]] = []
    for signal_date, scores in sorted(scores_by_month.items()):
        top = scores.sort_values(ascending=False).head(args.top_n)
        targets = pd.Series(1.0 / len(top), index=top.index)
        try:
            execution = next_open_date(pd.Timestamp(signal_date), calendar)
        except ValueError:
            continue
        if execution > open_pivot.index.max():
            continue
        market = _execution_market(open_pivot, status, execution)
        pre_trade = ledger.mark_to_market(market.set_index("symbol")["open"])
        ledger.rebalance(execution, market, targets)
        post_trade = float(ledger.equity_frame().iloc[-1]["equity"])
        nav_rows.append({
            "signal_date": signal_date, "execution_date": execution,
            "pre_trade_equity": pre_trade, "post_trade_equity": post_trade,
        })
    if not nav_rows:
        raise SystemExit("no rebalances executed")
    nav = pd.DataFrame(nav_rows).reset_index(drop=True)
    nav["monthly_return"] = nav["pre_trade_equity"].shift(-1) / nav["post_trade_equity"] - 1
    # 首月：从初始资金到首次调仓后的净值；末月：到最后收盘价的净值
    nav.loc[0, "monthly_return"] = nav["post_trade_equity"].iloc[0] / args.initial_cash - 1
    final_equity = ledger.mark_to_market(close_pivot.iloc[-1])
    nav.loc[len(nav) - 1, "monthly_return"] = final_equity / nav["post_trade_equity"].iloc[-1] - 1
    trades = ledger.trade_frame()
    filled = trades[trades["status"] == "filled"]
    rejected = trades[trades["status"] == "rejected"]
    notional = float(filled["notional"].sum())
    costs = float(filled[["commission", "stamp_tax", "transfer_fee"]].sum().sum())
    slippage_cost = notional * slippage / 10_000
    avg_equity = float(nav["post_trade_equity"].mean())
    periods = {
        "development": ("2014-01-01", "2020-12-31"),
        "validation": ("2021-01-01", "2023-12-31"),
        "test": ("2024-01-01", "2025-12-31"),
        "monitoring": ("2026-01-01", "2026-12-31"),
    }
    rows = []
    for name, (a, b) in periods.items():
        part = nav[(nav["signal_date"] >= a) & (nav["signal_date"] <= b)]
        if not part.empty:
            rows.append({"period": name, **_stats(part["monthly_return"])})
    rows.append({"period": "all", **_stats(nav["monthly_return"])})
    perf = pd.DataFrame(rows).set_index("period")
    print(f"[backtest] {index_code} {args.method} top-{args.top_n} | "
          f"{'无成本' if args.zero_cost else f'滑点{args.slippage_bps}bps+佣金3bp+印花税+过户费'}")
    print(perf.round(4).to_string())
    annual_turnover = notional / avg_equity / (len(nav) / 12)
    print(f"\n换手: 单边 {annual_turnover:.2f} 倍/年 | 月度平均调仓股数 {len(filled) / len(nav):.1f} | 拒单 {len(rejected)} 笔")
    print(f"成本: 手续费 {costs:,.0f} + 滑点 {slippage_cost:,.0f} = {(costs + slippage_cost):,.0f} 元 "
          f"(约占年均净值 {100 * (costs + slippage_cost) / avg_equity / (len(nav) / 12):.2f}%/年)")
    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        nav.to_csv(out / "nav.csv", index=False)
        trades.to_csv(out / "trades.csv", index=False)
        perf.to_csv(out / "performance.csv")
        print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
