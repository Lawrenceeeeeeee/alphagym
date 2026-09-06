from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mlquant.optional import require

METHODS = (
    "equal", "factor_return_decay", "ic_decay", "max_icir", "max_ic", "pca",
    "lasso", "ridge", "rf", "xgb", "mlp",
)
ML_METHODS = ("lasso", "ridge", "rf", "xgb", "mlp")
DEFAULT_SPEARMAN_THRESHOLD = 0.80

_ML_FEATURE_LAGS = 12
_ML_MIN_MONTHS = 16
_ML_MIN_SAMPLES = 40
_SELECTION_YEAR_STOP = pd.Timestamp("2026-01-01")


def align_factor_signs(values: pd.DataFrame, directions: dict[str, str]) -> pd.DataFrame:
    aligned = values.copy()
    for column in aligned:
        if directions.get(column) == "negative":
            aligned[column] = -aligned[column]
        elif directions.get(column, "unknown") == "unknown":
            raise ValueError(f"unknown direction must be locked in development: {column}")
    return aligned


def exponential_weights(length: int, half_life: float = 6.0) -> np.ndarray:
    age = np.arange(length - 1, -1, -1)
    weights = np.exp(-np.log(2) * age / half_life)
    return weights / weights.sum()


def _normalize_nonnegative(weights: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(weights, dtype=float), 0)
    return result / result.sum() if result.sum() else np.full(len(result), 1 / len(result))


def _optimized(mean: np.ndarray, covariance: np.ndarray, *, risk_adjusted: bool) -> np.ndarray:
    count = len(mean)
    def objective(weight: np.ndarray) -> float:
        expected = float(weight @ mean)
        if not risk_adjusted:
            return -expected
        variance = max(float(weight @ covariance @ weight), 1e-12)
        return -expected / np.sqrt(variance)
    result = minimize(objective, np.full(count, 1 / count), method="SLSQP",
                      bounds=[(0, 1)] * count, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return _normalize_nonnegative(result.x if result.success else np.ones(count))


def factor_weights(history: pd.DataFrame, method: str, *, half_life: float = 6.0) -> pd.Series:
    if method not in METHODS:
        raise ValueError(f"unknown combine method: {method}")
    if method in ML_METHODS:
        return _ml_factor_weights(history, method)
    sample = history.tail(12).dropna(axis=1, how="all").fillna(0.0)
    if sample.empty:
        raise ValueError("12-month history is empty")
    columns = sample.columns
    decay = exponential_weights(len(sample), half_life)
    mean = np.average(sample.to_numpy(), axis=0, weights=decay)
    if method == "equal":
        return pd.Series(np.full(len(columns), 1 / len(columns)), index=columns, name=method)
    if method in {"factor_return_decay", "ic_decay"}:
        return pd.Series(_normalize_nonnegative(mean), index=columns, name=method)
    require("sklearn", "ml")
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA

    covariance = LedoitWolf().fit(sample.to_numpy()).covariance_ if len(sample) >= 2 else np.eye(len(columns))
    if method == "equal":
        weights = np.ones(len(columns))
    elif method in {"factor_return_decay", "ic_decay"}:
        weights = mean
    elif method == "max_icir":
        weights = _optimized(mean, covariance, risk_adjusted=True)
    elif method == "max_ic":
        weights = _optimized(mean, covariance, risk_adjusted=False)
    else:
        component = PCA(n_components=1).fit(sample.to_numpy()).components_[0]
        weights = component * np.sign(component @ mean or 1.0)
    return pd.Series(_normalize_nonnegative(weights), index=columns, name=method)


def _ml_estimator(method: str) -> object:
    require("sklearn", "ml")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if method == "lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=5000, random_state=42))
    if method == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))
    if method == "rf":
        return RandomForestRegressor(
            n_estimators=200, max_depth=3, min_samples_leaf=5,
            random_state=42, n_jobs=1,
        )
    if method == "xgb":
        return require("xgboost", "boosting").XGBRegressor(
            n_estimators=120, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1,
            verbosity=0,
        )
    if method == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(16,), alpha=0.01, max_iter=1000, random_state=42),
        )
    raise ValueError(f"unknown ML combine method: {method}")


def _ml_features(own: np.ndarray, cross: np.ndarray, rank: float, breadth: float) -> list[float]:
    """Summary features for one factor from its trailing 12-month window.

    ``own`` holds the factor's own trailing ICs, ``cross`` the same window's
    ICs across all factors, ``rank`` the factor's percentile position in the
    latest month, and ``breadth`` the latest month's cross-factor mean IC.
    """
    return [
        float(np.mean(own)), float(np.mean(own[-3:])), float(np.mean(own[-6:])),
        float(np.std(own)), float(own[-1]), float(np.min(own)), float(np.max(own)),
        rank, breadth,
    ]


def _ml_factor_weights(history: pd.DataFrame, method: str) -> pd.Series:
    """Supervised factor weights: predict each factor's forward IC from its own
    trailing 12-month IC window (pooled across factors), then take the positive
    part of the final-window prediction and normalize.

    Every training pair uses features from months strictly before its target,
    and the whole history must end before 2026 (monitoring data is label-only).
    """
    sample = history.tail(36).dropna(axis=1, how="all")
    frame = sample.to_numpy(dtype=float)
    months, count = frame.shape
    if months < _ML_MIN_MONTHS or count < 2:
        raise ValueError(
            f"ML 权重需要至少 {_ML_MIN_MONTHS} 个月历史与 2 个因子，"
            f"当前 {months} 个月 × {count} 个因子"
        )
    if not sample.index.empty and pd.Timestamp(sample.index.max()) >= _SELECTION_YEAR_STOP:
        raise ValueError("2026 数据不得参与合成选模（ML 权重训练只允许开发+验证期）")
    rows: list[list[float]] = []
    targets: list[float] = []
    for month in range(_ML_FEATURE_LAGS, months):
        trailing = frame[month - _ML_FEATURE_LAGS:month]
        latest = trailing[-1]
        order = np.argsort(np.argsort(latest))
        breadth = float(np.nanmean(latest))
        for factor in range(count):
            own = trailing[:, factor]
            if not np.all(np.isfinite(own)) or not np.isfinite(frame[month, factor]):
                continue
            rows.append(_ml_features(
                own, latest, (order[factor] + 1.0) / count, breadth,
            ))
            targets.append(frame[month, factor])
    if len(rows) < _ML_MIN_SAMPLES:
        raise ValueError(
            f"ML 权重训练样本不足（{len(rows)} < {_ML_MIN_SAMPLES}），"
            "请提供更长的月度 IC 历史"
        )
    estimator = _ml_estimator(method)
    estimator.fit(np.asarray(rows, dtype=float), np.asarray(targets, dtype=float))
    final = frame[-_ML_FEATURE_LAGS:]
    latest = final[-1]
    order = np.argsort(np.argsort(latest))
    breadth = float(np.nanmean(latest))
    predictions = np.full(count, np.nan)
    for factor in range(count):
        own = final[:, factor]
        if not np.all(np.isfinite(own)):
            continue
        features = _ml_features(
            own, latest, (order[factor] + 1.0) / count, breadth,
        )
        predictions[factor] = float(
            estimator.predict(np.asarray([features], dtype=float))[0]
        )
    weights = _normalize_nonnegative(np.nan_to_num(predictions, nan=0.0))
    return pd.Series(weights, index=sample.columns, name=method)


def select_low_correlation_factors(
    correlation: pd.DataFrame,
    priority: pd.Series,
    *,
    threshold: float = DEFAULT_SPEARMAN_THRESHOLD,
) -> list[str]:
    """Greedily retain stronger factors from highly rank-correlated pairs.

    ``correlation`` and ``priority`` must already be built from data available before
    the portfolio signal date. Absolute priority is used so callers can pass signed IC.
    Ties preserve the correlation matrix's column order for deterministic results.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Spearman threshold must be in (0, 1]")
    if correlation.empty:
        return []
    if correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation matrix must be square")
    if list(correlation.index) != list(correlation.columns):
        raise ValueError("correlation matrix index and columns must match")

    order = {column: position for position, column in enumerate(correlation.columns)}
    strength = priority.reindex(correlation.columns).abs().fillna(-np.inf)
    candidates = sorted(
        correlation.columns,
        key=lambda column: (-float(strength[column]), order[column]),
    )
    selected: list[str] = []
    for candidate in candidates:
        redundant = False
        for retained in selected:
            value = correlation.loc[candidate, retained]
            if pd.notna(value) and abs(float(value)) >= threshold:
                redundant = True
                break
        if not redundant:
            selected.append(str(candidate))
    return selected


def select_validation_method(metrics: pd.DataFrame) -> str:
    ordered = metrics.sort_values(["net_information_ratio", "turnover"], ascending=[False, True])
    if ordered.empty:
        raise ValueError("validation metrics are empty")
    best = ordered.iloc[0]
    close = ordered[ordered["net_information_ratio"] >= best["net_information_ratio"] - 0.05]
    return str(close.sort_values("turnover").iloc[0]["method"])


def combine_scores(scores: pd.DataFrame, weights: pd.Series) -> pd.Series:
    columns = weights.index.intersection(scores.columns)
    available = scores[columns].notna()
    numerator = scores[columns].fillna(0).mul(weights[columns], axis=1).sum(axis=1)
    denominator = available.mul(weights[columns], axis=1).sum(axis=1).replace(0, np.nan)
    return (numerator / denominator).rename("combined_score")
