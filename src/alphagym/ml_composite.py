"""Stock-level ML composite factor research.

Trains cross-sectional models (OLS/Ridge/LASSO/decision tree/random forest/
extra trees/XGBoost/LightGBM/MLP) that map the curated factor set to next-month
open-to-open returns, then evaluates each model's prediction as a composite
factor with the same monthly quintile/IC suite used for single factors.

Point-in-time discipline:
- A model predicting month ``t`` only ever sees factor data from months < t.
- The frozen protocol trains on development and selects hyperparameters on
  validation; test and 2026 monitoring months are only evaluated.
- The walk-forward protocol refits every ``refit_months`` months using strictly
  past data, and stops refitting at the validation end so test-period rows are
  scored by the frozen selection model (a robustness comparison).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphagym import storage_io
from alphagym.audit import assert_selection_isolation
from alphagym.model_catalog import FEATURE_MODES, LABEL_MODES, MODEL_KEYS
from alphagym.optional import require
from alphagym.research import evaluate_factor_batch, neutralize_cross_section, zscore

_REFIT_GRACE = pd.Timedelta(days=400)


def model_specs() -> dict[str, dict[str, Any]]:
    """Registry of model keys with their display labels and estimator factories."""
    def factories(seed: int) -> dict[str, Any]:
        require("sklearn", "ml")
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        from sklearn.linear_model import Lasso, LinearRegression, Ridge
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeRegressor

        return {
            "ols": lambda: LinearRegression(),
            # Fixed research configuration; tune only within the allowed selection split.
            "lasso": lambda: make_pipeline(
                StandardScaler(), Lasso(alpha=0.001, max_iter=5000, random_state=seed)
            ),
            "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed)),
            "dtree": lambda: DecisionTreeRegressor(max_depth=4, min_samples_leaf=50, random_state=seed),
            "rf": lambda: RandomForestRegressor(
                n_estimators=200, max_depth=6, min_samples_leaf=25, random_state=seed, n_jobs=-1,
            ),
            "et": lambda: ExtraTreesRegressor(
                n_estimators=200, max_depth=8, min_samples_leaf=25, random_state=seed, n_jobs=-1,
            ),
            "xgb": lambda: require("xgboost", "boosting").XGBRegressor(
                n_estimators=160, max_depth=3, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=0,
            ),
            "lgbm": lambda: require("lightgbm", "boosting").LGBMRegressor(
                n_estimators=160, num_leaves=15, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1,
            ),
            "mlp": lambda: make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=(32, 16), alpha=0.01, max_iter=800, random_state=seed),
            ),
        }

    labels = {
        "ols": "OLS线性", "lasso": "LASSO", "ridge": "岭回归", "dtree": "决策树",
        "rf": "随机森林", "et": "极端随机树", "xgb": "XGBoost", "lgbm": "LightGBM",
        "mlp": "神经网络",
    }
    return {
        key: {"label": labels[key], "factory": lambda seed, key=key: factories(seed)[key]()}
        for key in MODEL_KEYS
    }


@dataclass
class CompositeDataset:
    """Wide factor features plus labels, aligned on a (signal_date, symbol) index."""

    wide_raw: pd.DataFrame
    forward: pd.Series
    index_code: str


def load_panel_wide(
    panel: pd.DataFrame,
    factor_ids: list[str],
    index_code: str = "ALL_A",
) -> CompositeDataset:
    """Pivot a generated panel into a wide feature frame and forward-return labels."""
    if "index_code" in panel:
        panel = panel[panel["index_code"] == index_code]
    panel = panel[panel["factor_name"].isin(factor_ids)].copy()
    if panel.empty:
        raise ValueError(f"panel contains none of the requested factors for {index_code}")
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    wide = panel.pivot_table(
        index=["signal_date", "symbol"], columns="factor_name", values="raw_value",
        aggfunc="first",
    )
    wide = wide.reindex(columns=list(factor_ids)).astype(np.float32)
    forward = panel.groupby(["signal_date", "symbol"])["forward_return"].first()
    forward = forward.reindex(wide.index).astype(np.float32)
    return CompositeDataset(wide_raw=wide, forward=forward, index_code=index_code)


def load_pit_context(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PIT industries and float market cap aligned to the wide feature index."""
    root = Path(root)
    daily = storage_io.read_frame(root / "equity" / "daily.parquet",
                            columns=["trade_date", "symbol", "float_market_cap"])
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    cap = daily.set_index(["trade_date", "symbol"])["float_market_cap"].rename("float_market_cap")
    industries = storage_io.read_frame(root / "equity" / "industries.parquet")
    for column in ("valid_from", "valid_to"):
        industries[column] = pd.to_datetime(industries[column])
    return industries, cap


def pit_industry(industries: pd.DataFrame, dates: pd.Index) -> pd.Series:
    """SW1 industry for each (signal_date, symbol) pair: the interval covering the date."""
    frames: list[pd.DataFrame] = []
    for date in dates:
        when = pd.Timestamp(date).normalize()
        active = industries[
            (industries["valid_from"] <= when)
            & (industries["valid_to"].isna() | (industries["valid_to"] >= when))
        ][["symbol", "industry_code"]].copy()
        active["signal_date"] = when
        frames.append(active)
    result = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["signal_date", "symbol"], keep="last"
    )
    return result.set_index(["signal_date", "symbol"])["industry_code"].rename("industry_code")


def neutralize_wide(
    wide: pd.DataFrame,
    industries: pd.DataFrame,
    cap: pd.Series,
) -> pd.DataFrame:
    """Per-month, per-factor residual z-scores via the fixed cross-section order.

    5-MAD winsorize -> z-score -> WLS on SW1 dummies + log float market cap ->
    residual z-score (see ``alphagym.research.neutralize_cross_section``).
    """
    signal_dates = wide.index.get_level_values("signal_date").unique()
    industry = pit_industry(industries, signal_dates)
    cap = cap.reindex(wide.index)
    blocks: list[pd.DataFrame] = []
    for factor in wide.columns:
        frame = wide[factor].rename("raw_value").to_frame()
        frame["industry_code"] = industry.reindex(frame.index)
        frame["float_market_cap"] = cap
        result = frame.groupby("signal_date", observed=True, group_keys=False).apply(
            neutralize_cross_section, include_groups=False
        )
        blocks.append(result["neutralized_value"].rename(factor))
    return pd.concat(blocks, axis=1).astype(np.float32)


def _month_ranks(values: pd.Series) -> pd.Series:
    return values.groupby("signal_date", observed=True).rank(pct=True)


def prepare_matrices(
    features: pd.DataFrame,
    forward: pd.Series,
    *,
    label_mode: str = "return",
    imputation: pd.Series | None = None,
    drop_nan_label: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Stack features into an (N, F) float32 matrix and labels into a vector.

    Missing factor values are imputed with ``imputation`` (per-column medians
    fitted on development data only); remaining NaNs fall back to zero. Rows
    without a label are dropped for training unless ``drop_nan_label`` is
    False (prediction of the latest month, whose forward label is not yet
    observable).
    """
    if label_mode == "rank":
        label = _month_ranks(forward.reindex(features.index)).rename("label")
    elif label_mode == "return":
        label = forward.reindex(features.index).rename("label")
    else:
        raise ValueError(f"label_mode must be return/rank, got {label_mode}")
    x = features.fillna(imputation if imputation is not None else 0.0).fillna(0.0)
    if drop_nan_label:
        valid = label.notna()
        x = x.loc[valid]
        label = label.loc[valid]
    return x.to_numpy(np.float32), label.to_numpy(np.float32), label


def fit_imputation(features: pd.DataFrame) -> pd.Series:
    return features.median(axis=0)


def _predict(
    estimator: Any,
    features: pd.DataFrame,
    forward: pd.Series,
    imputation: pd.Series,
    label_mode: str,
) -> pd.Series:
    """Predict every feature row; rows whose forward label is NaN (latest
    month) still receive a score so signals can be exported from it."""
    x, _y, _label = prepare_matrices(
        features, forward, label_mode=label_mode, imputation=imputation, drop_nan_label=False
    )
    prediction = estimator.predict(x)
    return pd.Series(prediction, index=features.index, dtype=np.float32)


def frozen_composite(
    features: pd.DataFrame,
    forward: pd.Series,
    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    model_key: str,
    *,
    label_mode: str = "return",
    seed: int = 42,
    label_end_dates: pd.Series | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Train on development, predict validation/test/monitoring with a frozen model.

    The estimator is fitted once on development months (both ends inclusive);
    validation may later select among models, test and 2026 rows are only
    scored, never seen by the fit.
    """
    development = splits["development"]
    signal_dates = features.index.get_level_values("signal_date")
    train_mask = (signal_dates >= development[0]) & (signal_dates <= development[1])
    if label_end_dates is not None:
        ends = label_end_dates.reindex(features.index)
        train_mask &= (ends.notna() & (ends <= development[1])).to_numpy()
    if train_mask.sum() < 60:
        raise ValueError(f"development sample too small for {model_key}: {int(train_mask.sum())}")
    assert_selection_isolation(pd.to_datetime(signal_dates[train_mask]))
    estimator = model_specs()[model_key]["factory"](seed)
    imputation = fit_imputation(features.loc[train_mask])
    x_train, y_train, _label = prepare_matrices(
        features.loc[train_mask], forward, label_mode=label_mode, imputation=imputation
    )
    estimator.fit(x_train, y_train)
    prediction = _predict(estimator, features, forward, imputation, label_mode)
    meta = {
        "protocol": "frozen",
        "train_start": str(development[0].date()),
        "train_end": str(development[1].date()),
        "train_rows": int(x_train.shape[0]),
        "label_end_purged": label_end_dates is not None,
        "feature_columns": list(features.columns),
        "imputation": {str(k): float(v) for k, v in imputation.to_dict().items()},
    }
    return prediction, meta


def walk_forward_composite(
    features: pd.DataFrame,
    forward: pd.Series,
    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    model_key: str,
    *,
    label_mode: str = "return",
    refit_months: int = 12,
    seed: int = 42,
) -> tuple[pd.Series, dict[str, Any]]:
    """Expanding walk-forward: refit every ``refit_months`` using strictly past data.

    The prediction for month ``t`` is produced by the last estimator fitted on
    data ending before ``t``. Refits stop at the validation end: per the
    research discipline, test-period months must not tune direction or
    parameters, so test/2026 rows are scored by the model frozen at the last
    selection refit.
    """
    signal_dates = features.index.get_level_values("signal_date")
    months = sorted(pd.to_datetime(signal_dates).unique())
    if len(months) < 2:
        raise ValueError("walk-forward needs at least two signal months")
    start = splits["development"][0]
    selection_end = splits["validation"][1]
    minimum_train = 24
    estimator = model_specs()[model_key]["factory"](seed)
    predictions: list[pd.Series] = []
    imputation = fit_imputation(features.head(0))
    last_refit_position = -refit_months - 1
    fitted = False
    refit_dates: list[str] = []
    for position, month in enumerate(months):
        mask = signal_dates == month
        block = features.loc[mask]
        target = forward.loc[mask]
        past = months[:position]
        if (
            month >= start + _REFIT_GRACE
            and month <= selection_end
            and len(past) >= minimum_train
            and position - last_refit_position >= refit_months
        ):
            train_mask = signal_dates < month
            assert_selection_isolation(pd.to_datetime(signal_dates[train_mask]))
            imputation = fit_imputation(features.loc[train_mask])
            x_train, y_train, _label = prepare_matrices(
                features.loc[train_mask], forward, label_mode=label_mode, imputation=imputation
            )
            estimator = model_specs()[model_key]["factory"](seed)
            estimator.fit(x_train, y_train)
            last_refit_position = position
            fitted = True
            refit_dates.append(str(month.date()))
        if not fitted:
            predictions.append(pd.Series(np.nan, index=block.index, dtype=np.float32))
        else:
            predictions.append(_predict(estimator, block, target, imputation, label_mode))
    prediction = pd.concat(predictions)
    meta = {
        "protocol": "walk_forward",
        "refit_months": refit_months,
        "first_refit": refit_dates[0] if refit_dates else None,
        "last_refit": refit_dates[-1] if refit_dates else None,
        "refits": len(refit_dates),
        "feature_columns": list(features.columns),
    }
    return prediction, meta


def predictions_panel(
    prediction: pd.Series,
    forward: pd.Series,
    factor_name: str,
    index_code: str = "ALL_A",
) -> pd.DataFrame:
    frame = pd.DataFrame({"raw_value": prediction, "forward_return": forward}).dropna(
        subset=["raw_value"]
    )
    frame["signal_date"] = frame.index.get_level_values("signal_date")
    frame["symbol"] = frame.index.get_level_values("symbol")
    frame["factor_name"] = factor_name
    frame["neutralized_value"] = np.nan
    frame["index_code"] = index_code
    return frame.reset_index(drop=True)[
        ["signal_date", "symbol", "factor_name", "raw_value",
         "neutralized_value", "forward_return", "index_code"]
    ]


@dataclass
class CompositeResult:
    """Per-model predictions plus the standard evaluation outputs."""

    panel: pd.DataFrame
    monthly: pd.DataFrame
    summary: pd.DataFrame
    meta: dict[str, Any]


def run_composites(
    dataset: CompositeDataset,
    industries: pd.DataFrame,
    cap: pd.Series,
    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    *,
    models: tuple[str, ...] = MODEL_KEYS,
    feature_mode: str = "neutral",
    label_mode: str = "return",
    protocol: str = "frozen",
    refit_months: int = 12,
    seed: int = 42,
) -> list[CompositeResult]:
    """Train every requested model and evaluate its predictions as a factor."""
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be {'/'.join(FEATURE_MODES)}, got {feature_mode}")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be {'/'.join(LABEL_MODES)}, got {label_mode}")
    if protocol not in {"frozen", "walk_forward"}:
        raise ValueError(f"protocol must be frozen/walk_forward, got {protocol}")
    if protocol == "walk_forward" and feature_mode == "neutral":
        raise ValueError("walk-forward with neutral features is not supported yet")
    features = (
        neutralize_wide(dataset.wide_raw, industries, cap)
        if feature_mode == "neutral"
        else dataset.wide_raw.groupby(level="signal_date", observed=True).transform(zscore)
    )
    results: list[CompositeResult] = []
    for key in models:
        if key not in MODEL_KEYS:
            raise ValueError(f"unknown model: {key}")
        try:
            if protocol == "frozen":
                prediction, meta = frozen_composite(
                    features, dataset.forward, splits, key, label_mode=label_mode, seed=seed
                )
            else:
                prediction, meta = walk_forward_composite(
                    features, dataset.forward, splits, key,
                    label_mode=label_mode, refit_months=refit_months, seed=seed,
                )
        except (ValueError, np.linalg.LinAlgError) as error:
            raise ValueError(f"{key} failed: {error}") from error
        panel = predictions_panel(prediction, dataset.forward, f"{feature_mode}_{key}", dataset.index_code)
        monthly, summary = evaluate_factor_batch(panel, periods=splits)
        meta.update({
            "model_key": key,
            "model_label": model_specs()[key]["label"],
            "feature_mode": feature_mode,
            "label_mode": label_mode,
            "protocol": protocol,
        })
        results.append(CompositeResult(
            panel=panel, monthly=monthly, summary=summary, meta=meta,
        ))
    return results


def load_composite_dataset(
    panel_path: str | Path,
    factor_ids: list[str],
    index_code: str = "ALL_A",
) -> CompositeDataset:
    panel = storage_io.read_frame(panel_path)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    return load_panel_wide(panel, factor_ids, index_code)
