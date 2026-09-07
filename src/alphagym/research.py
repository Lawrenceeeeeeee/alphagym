from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

from alphagym.audit import assign_period


def mad_winsorize(values: pd.Series, scale: float = 5.0) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce").copy()
    median = result.median()
    mad = (result - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return result
    return result.clip(median - scale * mad, median + scale * mad)


def zscore(values: pd.Series, weights: pd.Series | None = None) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    valid = result.notna()
    if weights is None:
        mean, std = result[valid].mean(), result[valid].std(ddof=0)
    else:
        weight = weights.reindex(result.index).where(valid).fillna(0).clip(lower=0)
        if weight.sum() == 0:
            return pd.Series(np.nan, index=result.index)
        weight = weight / weight.sum()
        mean = (result.fillna(0) * weight).sum()
        std = np.sqrt(((result.fillna(mean) - mean) ** 2 * weight).sum())
    return (result - mean) / std if std and np.isfinite(std) else pd.Series(np.nan, index=result.index)


def neutralize_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    """5-MAD -> z-score -> WLS(SW1 dummies + log float cap) -> residual z-score."""
    required = {"raw_value", "industry_code", "float_market_cap"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"neutralization missing columns: {sorted(missing)}")
    result = frame.copy()
    result["winsorized_value"] = mad_winsorize(result["raw_value"])
    result["standardized_value"] = zscore(result["winsorized_value"])
    valid = result[list(required)].notna().all(axis=1) & (result["float_market_cap"] > 0)
    result["neutralized_value"] = np.nan
    if valid.sum() < 3:
        return result
    sample = result.loc[valid]
    dummies = pd.get_dummies(sample["industry_code"], drop_first=False, dtype=float)
    design = pd.concat(
        [dummies, np.log(sample["float_market_cap"]).rename("log_float_market_cap")], axis=1
    )
    design = design.loc[:, design.var() > 0]
    design.insert(0, "intercept", 1.0)
    weights = sample["float_market_cap"] / sample["float_market_cap"].median()
    x = design.to_numpy(float) * np.sqrt(weights.to_numpy())[:, None]
    y = sample["standardized_value"].to_numpy(float) * np.sqrt(weights.to_numpy())
    coefficient = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = sample["standardized_value"] - design.to_numpy(float) @ coefficient
    result.loc[valid, "neutralized_value"] = zscore(residual, sample["float_market_cap"])
    return result


def eligible_universe(
    date: pd.Timestamp,
    index_code: str,
    members: pd.DataFrame,
    securities: pd.DataFrame,
    status: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    when = pd.Timestamp(date).normalize()
    active = members[
        (members["index_code"] == index_code)
        & (members["valid_from"] <= when)
        & (members["valid_to"].isna() | (members["valid_to"] >= when))
    ].copy()
    master = securities.copy()
    opened = pd.to_datetime(calendar.loc[calendar["is_open"].astype(bool), "trade_date"])
    master["listed_trading_days"] = master["list_date"].map(lambda value: int(((opened >= value) & (opened <= when)).sum()))
    active = active.merge(master, on="symbol", how="inner")
    active = active[(active["listed_trading_days"] >= 250) & (active["list_date"] <= when)]
    active = active[active["delist_date"].isna() | (active["delist_date"] > when)]
    current_status = status[status["trade_date"] == when]
    active = active.merge(current_status, on="symbol", how="left", suffixes=("", "_status"))
    for column in ("is_st", "is_pt"):
        if column in active:
            active = active[~active[column].fillna(False).astype(bool)]
    return active


def huatai_industry_layers(frame: pd.DataFrame, *, groups: int = 5) -> pd.DataFrame:
    """Assign layer 1 to the highest factor values while matching industry weights."""
    required = {"symbol", "industry_code", "factor_value", "benchmark_weight"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"layering missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    industry_weights = frame.groupby("industry_code", observed=True)["benchmark_weight"].sum()
    industry_weights = industry_weights / industry_weights.sum()
    for industry, group in frame.groupby("industry_code", observed=True):
        ordered = group.sort_values(
            ["factor_value", "symbol"], ascending=[False, True]
        ).reset_index(drop=True)
        count = len(ordered)
        if not count:
            continue
        for position, stock in ordered.iterrows():
            stock_left, stock_right = position / count, (position + 1) / count
            for layer in range(1, groups + 1):
                layer_left, layer_right = (layer - 1) / groups, layer / groups
                overlap = max(0.0, min(stock_right, layer_right) - max(stock_left, layer_left))
                if overlap <= 0:
                    continue
                within_industry = overlap * groups
                rows.append(
                    {
                        "symbol": stock["symbol"], "industry_code": industry, "layer": layer,
                        "boundary_fraction": overlap * count,
                        "target_weight": float(industry_weights.loc[industry] * within_industry),
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        sums = result.groupby("layer", observed=True)["target_weight"].sum()
        if not np.allclose(sums, 1.0, atol=1e-10):
            raise AssertionError(f"layer weights do not sum to one: {sums.to_dict()}")
    return result


def newey_west_t(values: pd.Series, lags: int | None = None) -> float:
    sample = pd.Series(values).dropna().to_numpy(float)
    count = len(sample)
    if count < 3:
        return np.nan
    centered = sample - sample.mean()
    lags = min(lags if lags is not None else int(4 * (count / 100) ** (2 / 9)), count - 1)
    variance = centered @ centered / count
    for lag in range(1, lags + 1):
        covariance = centered[lag:] @ centered[:-lag] / count
        variance += 2 * (1 - lag / (lags + 1)) * covariance
    standard_error = np.sqrt(max(variance, 0) / count)
    return float(sample.mean() / standard_error) if standard_error else np.nan


@dataclass(slots=True)
class FactorStatistics:
    monthly: pd.DataFrame
    summary: pd.DataFrame


def evaluate_factor(panel: pd.DataFrame, value_column: str = "neutralized_value") -> FactorStatistics:
    rows: list[dict[str, float | pd.Timestamp]] = []
    for date, group in panel.groupby("signal_date", observed=True):
        sample = group[[value_column, "forward_return"]].dropna()
        if len(sample) < 3:
            continue
        ic = spearmanr(sample[value_column], sample["forward_return"]).statistic
        x = rankdata(sample[value_column])
        design = np.column_stack([np.ones(len(x)), x])
        slope = np.linalg.lstsq(design, sample["forward_return"], rcond=None)[0][1]
        quantile = 5 - pd.qcut(
            sample[value_column].rank(method="first"), 5, labels=False
        )
        group_returns = sample.groupby(quantile, observed=True)["forward_return"].mean()
        row: dict[str, float | pd.Timestamp] = {
            "signal_date": date, "rank_ic": float(ic), "factor_return": float(slope),
            "coverage": len(sample) / len(group),
        }
        row.update({f"group_{index}": float(group_returns.get(index, np.nan)) for index in range(1, 6)})
        rows.append(row)
    monthly = pd.DataFrame(rows)
    if monthly.empty:
        return FactorStatistics(monthly, pd.DataFrame())
    ic_std = monthly["rank_ic"].std(ddof=1)
    summary = pd.DataFrame(
        [{
            "rank_ic": monthly["rank_ic"].mean(),
            "icir": monthly["rank_ic"].mean() / ic_std if ic_std else np.nan,
            "hac_t": newey_west_t(monthly["rank_ic"]),
            "fama_macbeth_return": monthly["factor_return"].mean(),
            "fama_macbeth_hac_t": newey_west_t(monthly["factor_return"]),
            "coverage": monthly["coverage"].mean(),
            "monotonicity": -spearmanr(
                range(1, 6),
                [monthly[f"group_{i}"].mean() for i in range(1, 6)],
            ).statistic,
        }]
    )
    return FactorStatistics(monthly, summary)


def evaluate_factor_suite(
    panel: pd.DataFrame,
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate raw/neutralized and original/reversed directions without cherry-picking."""
    monthly_frames: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for value_column in ("raw_value", "neutralized_value"):
        for orientation, multiplier in (("original", 1.0), ("reversed", -1.0)):
            working = panel.copy()
            working[value_column] = working[value_column] * multiplier
            statistics = evaluate_factor(working, value_column)
            if statistics.monthly.empty:
                continue
            rank_panel = working.pivot(index="signal_date", columns="symbol", values=value_column).rank(
                axis=1, pct=True
            )
            turnover = rank_panel.diff().abs().mean(axis=1)
            statistics.monthly["rank_turnover"] = statistics.monthly["signal_date"].map(turnover)
            statistics.monthly["group_order"] = "descending"
            statistics.monthly["value_type"] = value_column.removesuffix("_value")
            statistics.monthly["orientation"] = orientation
            statistics.summary["value_type"] = value_column.removesuffix("_value")
            statistics.summary["orientation"] = orientation
            monthly_frames.append(statistics.monthly)
            summaries.append(statistics.summary)
    monthly = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if not monthly.empty:
        monthly["period"] = assign_period(monthly["signal_date"], periods)
        summary["period"] = "all"
        phase_rows: list[dict[str, object]] = []
        for (value_type, orientation, period), group in monthly.dropna(subset=["period"]).groupby(
            ["value_type", "orientation", "period"], observed=True
        ):
            standard_deviation = group["rank_ic"].std(ddof=1)
            phase_rows.append(
                {
                    "value_type": value_type,
                    "orientation": orientation,
                    "period": period,
                    "rank_ic": group["rank_ic"].mean(),
                    "icir": group["rank_ic"].mean() / standard_deviation
                    if standard_deviation
                    else np.nan,
                    "hac_t": newey_west_t(group["rank_ic"]),
                    "fama_macbeth_return": group["factor_return"].mean(),
                    "fama_macbeth_hac_t": newey_west_t(group["factor_return"]),
                    "coverage": group["coverage"].mean(),
                }
            )
        summary = pd.concat([summary, pd.DataFrame(phase_rows)], ignore_index=True)
    return monthly, summary


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce")
    valid = values.dropna().sort_values()
    count = len(valid)
    if not count:
        return pd.Series(np.nan, index=values.index)
    adjusted = valid * count / np.arange(1, count + 1)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    return adjusted.reindex(values.index)


def evaluate_factor_batch(
    panel: pd.DataFrame,
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every concrete factor, then apply one BH correction across the full batch."""
    if "factor_name" not in panel:
        panel = panel.assign(factor_name="factor")
    monthly_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for factor_name, group in panel.groupby("factor_name", observed=True):
        monthly, summary = evaluate_factor_suite(group, periods)
        monthly["factor_name"] = factor_name
        summary["factor_name"] = factor_name
        monthly_frames.append(monthly)
        summary_frames.append(summary)
    monthly = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    if not summary.empty:
        summary["p_value"] = 2 * norm.sf(summary["hac_t"].abs())
        summary["bh_q_value"] = benjamini_hochberg(summary["p_value"])
    return monthly, summary
