"""Report spec: parsing, validation and factor-set resolution.

A report spec is a YAML/JSON document describing one reproducible research
study: universe (index, PIT industry filter, symbol lists), window, custom
development/validation/test splits, a factor selection expression and an
optional combination section. Specs are plain data (dataclasses) so agents
can build them programmatically or load them from files.
"""
from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from alphagym.factor_store import FactorStore
from alphagym.model_catalog import FEATURE_MODES, LABEL_MODES, MODEL_KEYS

DEFAULT_SPLITS = {
    "development": ("2014-01-01", "2020-12-31"),
    "validation": ("2021-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2025-12-31"),
}
COMBINE_METHODS = (
    "equal", "factor_return_decay", "ic_decay", "max_icir", "max_ic", "pca",
    "lasso", "ridge", "rf", "xgb", "mlp",
)
METRIC_OPS = {
    ">=": operator.ge, ">": operator.gt, "<=": operator.le,
    "<": operator.lt, "==": operator.eq, "!=": operator.ne,
}
_SPLIT_NAMES = ("development", "validation", "test")
# 选模（指标筛选与排序）只允许在开发/验证期确定；测试期与全样本仅评估，不得参与选模。
SELECTION_PERIODS = ("development", "validation")


def _dates(value: object, label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} 必须是 [开始日期, 结束日期] 两个元素")
    try:
        start, end = pd.Timestamp(value[0]).normalize(), pd.Timestamp(value[1]).normalize()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 日期格式无效：{value}") from error
    if start >= end:
        raise ValueError(f"{label} 开始日期必须早于结束日期")
    return start, end


@dataclass(slots=True)
class UniverseSpec:
    index_code: str = "ALL_A"
    industries: dict[str, list[str]] = field(default_factory=dict)
    symbols: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class MetricFilter:
    metric: str
    op: str = ">="
    value: float = 0.0
    period: str = "development"
    orientation: str = "original"


@dataclass(slots=True)
class FactorSelection:
    families: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    metrics: list[MetricFilter] = field(default_factory=list)
    sort_by: MetricFilter | None = None
    top_n: int | None = None
    # 排序方向：abs（按绝对值，默认）/desc（降序）/asc（升序）
    direction: str = "abs"


@dataclass(slots=True)
class CombineSpec:
    methods: list[str] = field(default_factory=lambda: list(COMBINE_METHODS))
    correlation_threshold: float = 0.8
    rolling_months: int = 12
    selection_periods: list[str] = field(default_factory=lambda: ["development", "validation"])
    min_observations: int = 20
    # 个股层 ML 合成（可选）：把所选因子截面喂给 ML 模型预测下月收益，
    # 预测序列作为伪方法并入合成对比。protocol 仅支持 frozen（development 训练、
    # validation 选模、test/2026 只评估）；walk_forward 属于探索脚本范畴。
    ml: dict[str, Any] | None = None


@dataclass(slots=True)
class ReportSpec:
    name: str
    mode: str = "formal"
    universe: UniverseSpec = field(default_factory=UniverseSpec)
    start_date: str = "2014-01-01"
    end_date: str = "2025-12-31"
    monitoring_end: str | None = None
    splits: dict[str, list[str]] = field(default_factory=lambda: {
        name: list(pair) for name, pair in DEFAULT_SPLITS.items()
    })
    factors: FactorSelection = field(default_factory=FactorSelection)
    combine: CombineSpec | None = None
    description: str = ""
    # 持有期：信号形成后的持仓时长。当前管线仅支持月度（1M）：
    # 月末收盘形成信号、下一交易日开盘成交、持有至下月信号执行日。
    holding_period: str = "1M"

    def to_dict(self) -> dict[str, Any]:
        def clean(value: object) -> object:
            if isinstance(value, (ReportSpec, UniverseSpec, FactorSelection,
                                  CombineSpec, MetricFilter)):
                return clean(_asdict_shallow(value))
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items() if item not in (None, "", [], {})}
            if isinstance(value, list):
                return [clean(item) for item in value]
            return value

        return {
            "name": self.name,
            "description": self.description,
            "mode": self.mode,
            "holding_period": self.holding_period,
            "universe": clean(self.universe),
            "window": {
                "start": self.start_date,
                "end": self.end_date,
                **({"monitoring_end": self.monitoring_end} if self.monitoring_end else {}),
            },
            "splits": {name: [str(start.date()), str(end.date())] for name, (start, end) in self.resolved_splits().items()},
            "factors": clean(self.factors),
            **({"combine": clean(self.combine)} if self.combine else {}),
        }

    def resolved_splits(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        result: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for name in _SPLIT_NAMES:
            if name in self.splits:
                result[name] = _dates(self.splits[name], f"splits.{name}")
        return result

    def window_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return _dates((self.start_date, self.end_date), "window")

    def monitoring_bound(self) -> pd.Timestamp | None:
        if not self.monitoring_end:
            return None
        return pd.Timestamp(self.monitoring_end).normalize()


def _asdict_shallow(value: object) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cls in type(value).__mro__:
        if cls is object:
            continue
        slots = getattr(cls, "__slots__", ())
        for name in slots:
            if name in result:
                continue
            result[name] = getattr(value, name)
    return result


def load_spec(path: str | Path) -> ReportSpec:
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("报告 spec 必须是 YAML/JSON 对象")
    return parse_spec(payload)


def parse_spec(payload: dict[str, Any]) -> ReportSpec:
    name = payload.get("name")
    if not name:
        raise ValueError("报告必须提供 name")
    mode = payload.get("mode", "formal")
    if mode not in {"smoke", "formal"}:
        raise ValueError("mode 必须是 smoke 或 formal")

    holding_period = str(payload.get("holding_period") or "1M")
    if holding_period not in {"1M"}:
        raise ValueError(
            "holding_period 当前仅支持 1M（月度持有：月末收盘形成信号、"
            "下一交易日开盘成交、持有至下月信号执行日）"
        )

    universe_payload = payload.get("universe") or {}
    industries = universe_payload.get("industries") or {}
    symbols = universe_payload.get("symbols") or {}
    industry_include = _string_list(industries.get("include"), "universe.industries.include")
    industry_exclude = _string_list(industries.get("exclude"), "universe.industries.exclude")
    symbol_include = _string_list(symbols.get("include"), "universe.symbols.include")
    symbol_exclude = _string_list(symbols.get("exclude"), "universe.symbols.exclude")
    industry_overlap = sorted(set(industry_include) & set(industry_exclude))
    if industry_overlap:
        raise ValueError(f"universe.industries 包含与排除存在交集：{industry_overlap}")
    symbol_overlap = sorted(set(symbol_include) & set(symbol_exclude))
    if symbol_overlap:
        raise ValueError(f"universe.symbols 包含与排除存在交集：{symbol_overlap}")
    universe = UniverseSpec(
        index_code=str(universe_payload.get("index_code") or "ALL_A"),
        industries={"include": industry_include, "exclude": industry_exclude},
        symbols={"include": symbol_include, "exclude": symbol_exclude},
    )

    window = payload.get("window") or {}
    start_date = str(window.get("start") or payload.get("start_date") or "2014-01-01")
    end_date = str(window.get("end") or payload.get("end_date") or "2025-12-31")
    monitoring_end = window.get("monitoring_end")
    window_start, window_end = _dates((start_date, end_date), "window")
    if monitoring_end is not None:
        monitoring = pd.Timestamp(monitoring_end).normalize()
        if monitoring <= window_end:
            raise ValueError("monitoring_end 必须晚于 window.end")

    splits = payload.get("splits") or {}
    if not splits:
        splits = {name: list(pair) for name, pair in DEFAULT_SPLITS.items()}
    missing = [name for name in _SPLIT_NAMES if name not in splits]
    if missing:
        raise ValueError(f"splits 缺少阶段：{missing}；请提供 development/validation/test 三段划分")
    resolved = {
        name: _dates(splits[name], f"splits.{name}") for name in _SPLIT_NAMES
    }
    _validate_split_chain(window_start, window_end, resolved)

    factor_payload = payload.get("factors") or {}
    metric_rules = []
    for rule in factor_payload.get("metrics") or []:
        if not isinstance(rule, dict) or "metric" not in rule:
            raise ValueError(f"factors.metrics 条目必须是包含 metric 的对象：{rule}")
        op_text = str(rule.get("op", ">="))
        if op_text not in METRIC_OPS:
            raise ValueError(f"factors.metrics.op 不支持：{op_text}")
        period = str(rule.get("period", "development"))
        if period not in SELECTION_PERIODS:
            raise ValueError(
                f"factors.metrics.period 只能是 development/validation（{period} 含测试期，不可参与选模）"
            )
        metric_rules.append(MetricFilter(
            metric=str(rule["metric"]), op=op_text,
            value=float(rule.get("value", 0)), period=period,
            orientation=str(rule.get("orientation", "original")),
        ))
    sort_payload = factor_payload.get("sort_by")
    sort_by = None
    if sort_payload:
        if not isinstance(sort_payload, dict) or "metric" not in sort_payload:
            raise ValueError("factors.sort_by 必须是包含 metric 的对象")
        sort_period = str(sort_payload.get("period", "development"))
        if sort_period not in SELECTION_PERIODS:
            raise ValueError(
                f"factors.sort_by.period 只能是 development/validation（{sort_period} 含测试期，不可参与选模）"
            )
        sort_by = MetricFilter(
            metric=str(sort_payload["metric"]),
            period=sort_period,
            orientation=str(sort_payload.get("orientation", "original")),
        )
    direction = str(factor_payload.get("direction", "abs"))
    if direction not in {"abs", "desc", "asc"}:
        raise ValueError("factors.direction 必须是 abs/desc/asc")
    top_n = factor_payload.get("top_n")
    if top_n is not None:
        top_n = int(top_n)
        if top_n <= 0:
            raise ValueError("factors.top_n 必须为正整数")
    selection = FactorSelection(
        families=_string_list(factor_payload.get("families"), "factors.families"),
        include=_string_list(factor_payload.get("include"), "factors.include"),
        exclude=_string_list(factor_payload.get("exclude"), "factors.exclude"),
        metrics=metric_rules, sort_by=sort_by, top_n=top_n, direction=direction,
    )
    if not (selection.families or selection.include or selection.metrics
            or (selection.sort_by and selection.top_n)):
        raise ValueError(
            "factors 至少需要 families/include/metrics/排序取前N 之一来定义因子集合"
        )

    combine = None
    combine_payload = payload.get("combine")
    if combine_payload is not None:
        methods = combine_payload.get("methods") or list(COMBINE_METHODS)
        unknown = sorted(set(methods) - set(COMBINE_METHODS))
        if unknown:
            raise ValueError(f"combine.methods 不支持：{unknown}")
        selection_periods = list(combine_payload.get("selection_periods") or ["development", "validation"])
        unknown_periods = sorted(set(selection_periods) - set(_SPLIT_NAMES))
        if unknown_periods or not selection_periods:
            raise ValueError("combine.selection_periods 只能是 development/validation/test 的子集且非空")
        if "test" in selection_periods:
            raise ValueError("combine.selection_periods 不能包含 test（测试期不可调方向或参数）")
        combine = CombineSpec(
            methods=list(methods),
            correlation_threshold=float(combine_payload.get("correlation_threshold", 0.8)),
            rolling_months=int(combine_payload.get("rolling_months", 12)),
            selection_periods=selection_periods,
            min_observations=int(combine_payload.get("min_observations", 20)),
            ml=_parse_combine_ml(combine_payload.get("ml")),
        )
        if combine.rolling_months < 2:
            raise ValueError("combine.rolling_months 至少为 2")
        if not 0 < combine.correlation_threshold <= 1:
            raise ValueError("combine.correlation_threshold 必须在 (0, 1] 区间")

    return ReportSpec(
        name=str(name), mode=mode, universe=universe,
        start_date=str(window_start.date()), end_date=str(window_end.date()),
        monitoring_end=str(monitoring.date()) if monitoring_end is not None else None,
        splits=splits, factors=selection, combine=combine,
        description=str(payload.get("description") or ""),
        holding_period=str(payload.get("holding_period") or "1M"),
    )


def _parse_combine_ml(payload: object) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("combine.ml 必须是对象")
    models = [str(item) for item in payload.get("models") or list(MODEL_KEYS)]
    unknown = sorted(set(models) - set(MODEL_KEYS))
    if unknown:
        raise ValueError(f"combine.ml.models 不支持：{unknown}；可用：{list(MODEL_KEYS)}")
    feature_mode = str(payload.get("feature_mode") or "neutral")
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"combine.ml.feature_mode 必须是 {'/'.join(FEATURE_MODES)}")
    label_mode = str(payload.get("label_mode") or "return")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"combine.ml.label_mode 必须是 {'/'.join(LABEL_MODES)}")
    protocol = str(payload.get("protocol") or "frozen")
    if protocol != "frozen":
        raise ValueError("combine.ml.protocol 仅支持 frozen（walk_forward 属于探索脚本）")
    return {
        "models": models, "feature_mode": feature_mode,
        "label_mode": label_mode, "protocol": protocol,
    }


def _string_list(value: object, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_file():
            return sorted({
                line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            })
        return [str(value)]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError(f"{label} 必须是字符串、文件路径或列表")


def _validate_split_chain(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    resolved: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> None:
    development, validation, test = (
        resolved["development"], resolved["validation"], resolved["test"]
    )
    if development[0] != window_start:
        raise ValueError("splits.development 必须从 window.start 开始")
    if test[1] != window_end:
        raise ValueError("splits.test 必须到 window.end 结束")
    chain = (development, validation, test)
    for position in range(1, len(chain)):
        previous_end, next_start = chain[position - 1][1], chain[position][0]
        if next_start != previous_end + pd.Timedelta(days=1):
            raise ValueError(
                "splits 三段必须首尾相接：下一段的开始日期必须是上一段结束日期的次日"
            )


def resolve_factors(
    store: FactorStore, selection: FactorSelection, *, index_code: str = "ALL_A",
) -> list[dict[str, Any]]:
    """Resolve the factor selection expression against the catalog.

    The candidate pool starts from ``families`` when given, from the whole
    catalog when metric rules/sorting select from it, and is empty otherwise —
    an explicit ``include`` list must not silently expand to the full library.
    Explicit includes are unioned in after metric filtering and top-N
    truncation, so they always survive selection. Metric lookups read from
    ``index_code`` (the report universe), falling back to ALL_A when the
    universe has no succeeded runs of its own.
    """
    by_id = {item["factor_id"]: item for item in store.list_factors()}
    if selection.families:
        base = {
            factor_id: item for factor_id, item in by_id.items()
            if item["family"] in selection.families
        }
    elif selection.metrics or selection.sort_by:
        base = dict(by_id)
    else:
        base = {}

    included: dict[str, dict[str, Any]] = {}
    for factor_id in selection.include:
        if factor_id not in by_id:
            raise ValueError(f"未知因子：{factor_id}")
        included[factor_id] = by_id[factor_id]
    for factor_id in selection.exclude:
        if factor_id not in by_id:
            raise ValueError(f"未知因子：{factor_id}")
    selected = {
        factor_id: item for factor_id, item in base.items()
        if factor_id not in selection.exclude
    }
    selected.update({
        factor_id: item for factor_id, item in included.items()
        if factor_id not in selection.exclude
    })

    for rule in (*selection.metrics, selection.sort_by):
        if rule is not None and rule.period not in SELECTION_PERIODS:
            raise ValueError(
                "因子筛选/排序阶段只能是 development/validation，"
                f"got {rule.period}（测试期与全样本不可参与选模）"
            )

    if selection.metrics or (selection.sort_by and selection.top_n):
        metric_names = {rule.metric for rule in selection.metrics}
        if selection.sort_by:
            metric_names.add(selection.sort_by.metric)
        latest = store.latest_succeeded_runs()
        run_ids = sorted({item["run_id"] for item in latest.values()})
        metrics = _metric_lookup(store, run_ids, metric_names, index_code=index_code)
        for rule in selection.metrics:
            kept = {}
            for factor_id, item in selected.items():
                if factor_id in included:
                    kept[factor_id] = item
                    continue
                value = metrics.get((factor_id, rule.metric, rule.period, rule.orientation))
                if value is None or not METRIC_OPS[rule.op](value, rule.value):
                    continue
                kept[factor_id] = item
            selected = kept
        if selection.sort_by and selection.top_n:
            pool = [factor_id for factor_id in selected if factor_id not in included]
            ranked = []
            for factor_id in pool:
                value = metrics.get(
                    (factor_id, selection.sort_by.metric, selection.sort_by.period,
                     selection.sort_by.orientation)
                )
                ranked.append((factor_id, value if value is not None else float("nan")))
            if selection.direction == "abs":
                def key(pair: tuple[str, float]) -> float:
                    return abs(pair[1]) if pair[1] == pair[1] else -1.0
            elif selection.direction == "desc":
                def key(pair: tuple[str, float]) -> float:
                    return pair[1] if pair[1] == pair[1] else float("-inf")
            else:
                def key(pair: tuple[str, float]) -> float:
                    return -pair[1] if pair[1] == pair[1] else float("inf")
            ranked.sort(key=key, reverse=True)
            kept_ids = [factor_id for factor_id, _value in ranked[: selection.top_n]]
            selected = {
                factor_id: selected[factor_id]
                for factor_id in (*kept_ids, *included)
                if factor_id in selected
            }
    return sorted(selected.values(), key=lambda item: (item["family"], item["name"]))


def _metric_lookup(
    store: FactorStore, run_ids: list[str], metric_names: set[str], *,
    index_code: str = "ALL_A",
) -> dict[tuple[str, str, str, str], float]:
    """Metric values for ``index_code``, falling back to ALL_A runs.

    Per-index reports select factors on the universe's own metrics when those
    runs exist; factors without per-index coverage fall back to ALL_A so an
    explicit include list or a report created before ``--ensure-runs`` can
    still resolve. Selection never reads test-period rows.
    """
    if not run_ids or not metric_names:
        return {}
    placeholders = ",".join("?" for _ in run_ids)
    metric_placeholders = ",".join("?" for _ in metric_names)
    codes = (index_code,) if index_code == "ALL_A" else (index_code, "ALL_A")
    code_placeholders = ",".join("?" for _ in codes)
    rows = store.connection.execute(
        f"""SELECT m.factor_id, m.period, m.orientation, m.metric_name,
                  m.metric_value, m.index_code
        FROM factor_metric m
        WHERE m.run_id IN ({placeholders}) AND m.index_code IN ({code_placeholders})
          AND m.value_type='raw' AND m.cost_scenario='base_5bps'
          AND m.metric_name IN ({metric_placeholders})""",
        (*run_ids, *codes, *sorted(metric_names)),
    ).fetchall()
    result: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        if row["metric_value"] is None:
            continue
        key = (str(row["factor_id"]), str(row["metric_name"]), str(row["period"]),
               str(row["orientation"]))
        if key not in result or row["index_code"] == index_code:
            result[key] = float(row["metric_value"])
    return result
