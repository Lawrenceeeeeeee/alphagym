from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

if TYPE_CHECKING:
    from mlquant.factor_dsl import FormulaEngine

ExpectedDirection = Literal["positive", "negative", "unknown"]
FactorStatus = Literal["active", "blocked", "deprecated"]
Calculator = Callable[["FactorContext"], pd.Series]


@dataclass(frozen=True, slots=True)
class FieldDefinition:
    name: str
    dataset: str
    column: str
    dtype: str = "float"
    frequency: str = "daily"
    entity_key: str = "symbol"
    event_time: str = "trade_date"
    available_time: str | None = None
    formal: bool = True
    unit: str | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    factor_id: str
    name: str
    formula: str
    hypothesis_id: str
    family: str
    formula_version: str = "1.0"
    description: str = ""
    expected_direction: ExpectedDirection = "unknown"
    tags: tuple[str, ...] = ()
    status: FactorStatus = "active"


@dataclass(frozen=True, slots=True)
class FactorSpec:
    """Runtime view of one immutable database-backed factor revision."""

    name: str
    hypothesis_id: str
    family: str
    formula_version: str
    input_fields: tuple[str, ...]
    lookback_days: int
    min_observations: int
    availability_rule: str
    expected_direction: ExpectedDirection
    calculator: Calculator = field(repr=False, compare=False)
    factor_id: str = ""
    revision_id: str = ""
    formula: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    status: FactorStatus = "active"
    definition_hash: str = ""


@dataclass(slots=True)
class FactorContext:
    """Point-in-time data view shared by every formula operator."""

    signal_date: pd.Timestamp
    daily: pd.DataFrame
    fundamentals: pd.DataFrame
    datasets: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    models: Mapping[str, Any] = field(default_factory=dict)
    universe: tuple[str, ...] | None = None
    formal: bool = True
    cache: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.signal_date = pd.Timestamp(self.signal_date).normalize()

    def dataset(self, name: str) -> pd.DataFrame:
        if name == "market":
            frame = self.daily
        elif name == "financial":
            frame = self.fundamentals
        elif name in self.datasets:
            frame = self.datasets[name]
        else:
            raise KeyError(f"dataset unavailable: {name}")
        if self.universe is not None and "symbol" in frame:
            frame = frame[frame["symbol"].isin(self.universe)]
        return frame

    def visible(self, definition: FieldDefinition) -> pd.DataFrame:
        key = (
            f"visible:{definition.dataset}:{definition.event_time}:"
            f"{definition.available_time}"
        )
        if key in self.cache:
            return self.cache[key]
        frame = self.dataset(definition.dataset)
        if definition.column not in frame:
            raise KeyError(f"field unavailable: {definition.name} ({definition.column})")
        visible = frame
        if definition.event_time in visible:
            visible = visible[
                pd.to_datetime(visible[definition.event_time]) <= self.signal_date
            ]
        if definition.available_time:
            if definition.available_time not in visible:
                if self.formal:
                    raise ValueError(
                        f"formal point-in-time field {definition.name} requires "
                        f"{definition.available_time}"
                    )
            else:
                visible = visible[
                    pd.to_datetime(visible[definition.available_time]) <= self.signal_date
                ]
        self.cache[key] = visible
        return visible

    @property
    def latest_daily(self) -> pd.DataFrame:
        visible = self.daily[self.daily["trade_date"] <= self.signal_date]
        return (
            visible.sort_values("trade_date")
            .groupby("symbol", observed=True)
            .tail(1)
            .set_index("symbol")
        )

    @property
    def latest_fundamentals(self) -> pd.DataFrame:
        visible = self.fundamentals[
            self.fundamentals["available_date"] <= self.signal_date
        ]
        if visible.empty:
            return visible.set_index("symbol")
        visible = visible.sort_values(["symbol", "stat_date", "available_date"])
        visible = visible.drop_duplicates(["symbol", "stat_date"], keep="last")
        return visible.groupby("symbol", observed=True).tail(1).set_index("symbol")


class FactorRegistry:
    def __init__(self, engine: FormulaEngine | None = None) -> None:
        self._specs: dict[str, FactorSpec] = {}
        self._revisions: dict[str, FactorSpec] = {}
        self.engine = engine

    def register(self, spec: FactorSpec) -> None:
        if spec.name in self._specs:
            raise KeyError(f"factor already registered: {spec.name}")
        self._specs[spec.name] = spec
        if spec.revision_id:
            self._revisions[spec.revision_id] = spec

    def replace(self, spec: FactorSpec) -> None:
        self._specs[spec.name] = spec
        if spec.revision_id:
            self._revisions[spec.revision_id] = spec

    def add_revision(self, spec: FactorSpec, *, current: bool = False) -> None:
        if spec.revision_id:
            self._revisions[spec.revision_id] = spec
        if current:
            self._specs[spec.name] = spec

    def get(self, name: str) -> FactorSpec:
        return self._specs[name]

    def get_revision(self, revision_id: str) -> FactorSpec:
        return self._revisions[revision_id]

    def list(self, family: str | None = None) -> list[FactorSpec]:
        values = self._specs.values()
        if family is not None:
            values = (spec for spec in values if spec.family == family)
        return sorted(values, key=lambda spec: (spec.family, spec.name))

    def __len__(self) -> int:
        return len(self._specs)

    def compute_revision(self, revision_id: str, context: FactorContext) -> pd.Series:
        return self.get_revision(revision_id).calculator(context)
