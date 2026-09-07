from __future__ import annotations

import ast
import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alphagym.factors.base import FactorContext, FieldDefinition


class FormulaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FieldRef:
    definition: FieldDefinition


@dataclass(frozen=True, slots=True)
class ModelRef:
    model_id: str
    version: str | None = None


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    name: str
    version: str
    function: Any
    description: str = ""
    deterministic: bool = True

    @property
    def code_hash(self) -> str:
        try:
            source = inspect.getsource(self.function)
        except (OSError, TypeError):
            source = repr(self.function)
        return hashlib.sha256(source.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledFormula:
    source: str
    canonical_ast: str
    definition_hash: str
    fields: tuple[str, ...]
    operators: tuple[tuple[str, str, str], ...]
    factor_dependencies: tuple[tuple[str, str], ...]
    model_dependencies: tuple[tuple[str, str], ...]
    lookback_days: int
    min_observations: int
    tree: ast.Expression


class OperatorRegistry:
    def __init__(self) -> None:
        self._operators: dict[str, OperatorSpec] = {}

    def register(self, spec: OperatorSpec) -> None:
        name = spec.name.upper()
        if name in self._operators:
            raise KeyError(f"operator already registered: {name}")
        self._operators[name] = spec

    def get(self, name: str) -> OperatorSpec:
        try:
            return self._operators[name.upper()]
        except KeyError as error:
            raise FormulaError(f"unknown operator: {name}") from error

    def list(self) -> tuple[OperatorSpec, ...]:
        return tuple(sorted(self._operators.values(), key=lambda item: item.name))


class FieldRegistry:
    def __init__(self) -> None:
        self._fields: dict[str, FieldDefinition] = {}

    def register(self, definition: FieldDefinition) -> None:
        if definition.name in self._fields:
            raise KeyError(f"field already registered: {definition.name}")
        self._fields[definition.name] = definition

    def get(self, name: str) -> FieldDefinition:
        try:
            return self._fields[name]
        except KeyError as error:
            raise FormulaError(f"unknown field: {name}") from error

    def list(self) -> tuple[FieldDefinition, ...]:
        return tuple(sorted(self._fields.values(), key=lambda item: item.name))


class FormulaCompiler:
    def __init__(
        self,
        fields: FieldRegistry,
        operators: OperatorRegistry,
        factor_resolver: Any | None = None,
        model_resolver: Any | None = None,
    ) -> None:
        self.fields = fields
        self.operators = operators
        self.factor_resolver = factor_resolver
        self.model_resolver = model_resolver

    def compile(self, formula: str) -> CompiledFormula:
        source = formula.strip()
        expression = source[1:].strip() if source.startswith("=") else source
        if not expression or "\n" in expression or "\r" in expression:
            raise FormulaError("factor formula must be one non-empty line")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as error:
            raise FormulaError(f"invalid formula syntax: {error.msg}") from error
        state: dict[str, Any] = {
            "fields": set(), "operators": {}, "factors": {}, "models": {},
            "lookback": 0, "minimum": 1,
        }
        canonical = self._validate(tree.body, state)
        result_kind = self._infer_kind(tree.body)
        if result_kind != "series":
            raise FormulaError(f"factor formula must return a Series, got {result_kind}")
        canonical_ast = json.dumps(
            canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        operators = tuple(sorted(
            (name, version, code_hash)
            for name, (version, code_hash) in state["operators"].items()
        ))
        factors = tuple(sorted(state["factors"].items()))
        models = tuple(sorted(state["models"].items()))
        digest_payload = json.dumps(
            {"ast": canonical, "operators": operators, "factors": factors, "models": models},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        return CompiledFormula(
            source=source if source.startswith("=") else f"={source}",
            canonical_ast=canonical_ast,
            definition_hash=hashlib.sha256(digest_payload.encode()).hexdigest(),
            fields=tuple(sorted(state["fields"])),
            operators=operators,
            factor_dependencies=factors,
            model_dependencies=models,
            lookback_days=int(state["lookback"]),
            min_observations=int(state["minimum"]),
            tree=tree,
        )

    def _validate(self, node: ast.AST, state: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (str, int, float, bool, type(None))):
                raise FormulaError("unsupported literal")
            return {"const": node.value}
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name):
                raise FormulaError("nested attribute access is forbidden")
            field_name = f"{node.value.id}.{node.attr}"
            self.fields.get(field_name)
            state["fields"].add(field_name)
            return {"field": field_name}
        if isinstance(node, (ast.List, ast.Tuple)):
            return {"list": [self._validate(item, state) for item in node.elts]}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return {"unary": "+" if isinstance(node.op, ast.UAdd) else "-",
                    "value": self._validate(node.operand, state)}
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        ):
            symbols = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
            return {"binary": symbols[type(node.op)],
                    "left": self._validate(node.left, state),
                    "right": self._validate(node.right, state)}
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaError("only registered function calls are allowed")
            name = node.func.id.upper()
            if name == "FACTOR":
                factor_id = self._single_string_argument(node, "FACTOR")
                if self.factor_resolver is None:
                    raise FormulaError("factor references are unavailable")
                revision_id = str(self.factor_resolver(factor_id))
                state["factors"][factor_id] = revision_id
                return {"factor": factor_id, "revision": revision_id}
            if name == "MODEL":
                model_id = self._single_string_argument(node, "MODEL")
                version_id = (
                    str(self.model_resolver(model_id))
                    if self.model_resolver is not None else "unresolved"
                )
                state["models"][model_id] = version_id
                return {"model": model_id, "version": version_id}
            operator = self.operators.get(name)
            state["operators"][name] = (operator.version, operator.code_hash)
            self._validate_call_signature(operator, node)
            self._infer_window(name, node, state)
            return {
                "call": name, "version": operator.version,
                "args": [self._validate(arg, state) for arg in node.args],
                "kwargs": {keyword.arg: self._validate(keyword.value, state)
                           for keyword in node.keywords if keyword.arg is not None},
            }
        if isinstance(node, ast.Name) and node.id in {"True", "False", "None"}:
            return {"const": {"True": True, "False": False, "None": None}[node.id]}
        raise FormulaError(f"forbidden formula syntax: {type(node).__name__}")

    @staticmethod
    def _single_string_argument(node: ast.Call, name: str) -> str:
        if len(node.args) != 1 or node.keywords or not isinstance(node.args[0], ast.Constant):
            raise FormulaError(f"{name} requires one string argument")
        value = node.args[0].value
        if not isinstance(value, str) or not value:
            raise FormulaError(f"{name} requires one non-empty string argument")
        return value

    @staticmethod
    def _validate_call_signature(operator: OperatorSpec, node: ast.Call) -> None:
        signature = inspect.signature(operator.function)
        dummy = [None] * (len(node.args) + 1)
        kwargs = {item.arg: None for item in node.keywords if item.arg is not None}
        if any(item.arg is None for item in node.keywords):
            raise FormulaError("**kwargs expansion is forbidden")
        try:
            signature.bind(*dummy, **kwargs)
        except TypeError as error:
            raise FormulaError(f"invalid {operator.name} arguments: {error}") from error

    @staticmethod
    def _infer_window(name: str, node: ast.Call, state: dict[str, Any]) -> None:
        window_names = {
            "RETURN", "ROLLING_MEAN", "STD", "SUM", "MIN", "MAX", "EWM", "DIFF",
            "HIGH_PROXIMITY", "TURNOVER_BIAS", "AMIHUD", "VOLATILITY", "UPSIDE_VOL",
            "DOWNSIDE_VOL", "MAX_RETURN", "RSI", "CCI", "ATR", "BIAS",
        }
        if name not in window_names:
            return
        constants = [arg.value for arg in node.args
                     if isinstance(arg, ast.Constant) and isinstance(arg.value, int)]
        if constants:
            window = max(constants)
            state["lookback"] = max(state["lookback"], window)
            state["minimum"] = max(state["minimum"], max(1, int(window * 0.8)))

    def _infer_kind(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return "string"
            if isinstance(node.value, bool):
                return "bool"
            return "scalar"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return "field"
        if isinstance(node, (ast.List, ast.Tuple)):
            return "list"
        if isinstance(node, ast.UnaryOp):
            kind = self._infer_kind(node.operand)
            if kind not in {"series", "scalar"}:
                raise FormulaError(f"unary arithmetic does not accept {kind}")
            return kind
        if isinstance(node, ast.BinOp):
            left, right = self._infer_kind(node.left), self._infer_kind(node.right)
            if left not in {"series", "scalar"} or right not in {"series", "scalar"}:
                raise FormulaError(f"arithmetic requires scalar/Series, got {left} and {right}")
            return "series" if "series" in {left, right} else "scalar"
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise FormulaError(f"cannot infer type for {type(node).__name__}")
        name = node.func.id.upper()
        if name == "FACTOR":
            return "series"
        if name == "MODEL":
            return "model"
        kinds = [self._infer_kind(item) for item in node.args]
        financial = {"TTM", "MRQ", "YOY", "QOQ", "STABILITY"}
        field_series = {
            "ASOF", "RETURN", "DIFF", "ROLLING_MEAN", "STD", "SUM", "MIN", "MAX",
            "EWM", "TURNOVER_BIAS", "AMIHUD", "HIGH_PROXIMITY", "VOLATILITY",
            "UPSIDE_VOL", "DOWNSIDE_VOL", "MAX_RETURN", "ABS_RETURN", "IDIO_VOL",
            "IDIO_SKEW", "BIAS", "TRIX", "TRIX_SIGNAL", "RSI", "KDJ", "CCI",
            "OBV", "MAOBV", "BBI", "DPO", "ROC", "WR", "PSY", "PSYMA",
            "STREAK", "DAYS_SINCE", "MASS", "MASS_MA", "MFI", "VR", "EMV",
            "MAEMV", "AR", "BR", "CR", "DMI", "MACD", "ATR", "BBANDS",
            "DONCHIAN", "ASI", "ASIT", "BOLL", "DFMA", "MADPO", "EXPMA",
            "KELTNER", "MTMMA", "MAROC",
        } | financial
        if name in field_series:
            if not kinds or kinds[0] != "field":
                raise FormulaError(f"{name} requires a registered field as its first argument")
            if name in financial:
                first = node.args[0]
                assert isinstance(first, ast.Attribute) and isinstance(first.value, ast.Name)
                if self.fields.get(f"{first.value.id}.{first.attr}").dataset != "financial":
                    raise FormulaError(f"{name} requires a financial field")
            return "series"
        if name in {"SAFE_DIV", "LOG", "ABS", "CLIP", "RANK", "ZSCORE", "WINSORIZE"}:
            if any(kind not in {"series", "scalar"} for kind in kinds):
                raise FormulaError(f"{name} accepts only scalar/Series values")
            return "series" if "series" in kinds else "scalar"
        if name == "FEATURES":
            if not kinds or any(kind != "series" for kind in kinds):
                raise FormulaError("FEATURES requires one or more Series")
            return "features"
        if name == "MODEL_PREDICT":
            if kinds != ["model", "features"]:
                raise FormulaError("MODEL_PREDICT requires MODEL and FEATURES")
            return "series"
        if name == "ENSEMBLE":
            if not kinds or kinds[0] != "list":
                raise FormulaError("ENSEMBLE requires a list of predictions")
            return "series"
        if name in {"TEXT_SCORE", "EMBEDDING"}:
            if kinds[:2] != ["field", "model"]:
                raise FormulaError(f"{name} requires a text field and MODEL")
            return "series" if name == "TEXT_SCORE" else "embedding"
        if name == "DOCUMENT_AGG":
            if not kinds or kinds[0] not in {"embedding", "series"}:
                raise FormulaError("DOCUMENT_AGG requires text scores or embeddings")
            return "series"
        raise FormulaError(f"operator {name} has no registered type contract")


class FormulaEngine:
    def __init__(self, fields: FieldRegistry, operators: OperatorRegistry) -> None:
        self.fields = fields
        self.operators = operators

    def evaluate(
        self, compiled: CompiledFormula, context: FactorContext,
        factor_resolver: Any | None = None,
    ) -> pd.Series:
        cache_key = f"formula:{compiled.definition_hash}"
        if cache_key in context.cache:
            return context.cache[cache_key].copy()
        result = self._evaluate_node(
            compiled.tree.body, context, factor_resolver,
            dict(compiled.model_dependencies),
        )
        if np.isscalar(result):
            symbols = context.daily["symbol"].drop_duplicates().sort_values()
            result = pd.Series(float(result), index=symbols)
        if not isinstance(result, pd.Series):
            raise FormulaError("factor formula must produce a cross-sectional Series")
        result = pd.to_numeric(result, errors="coerce").replace([np.inf, -np.inf], np.nan)
        result = result.rename_axis("symbol")
        context.cache[cache_key] = result
        return result.copy()

    def _evaluate_node(
        self, node: ast.AST, context: FactorContext, factor_resolver: Any | None,
        model_versions: dict[str, str],
    ) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return FieldRef(self.fields.get(f"{node.value.id}.{node.attr}"))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [
                self._evaluate_node(item, context, factor_resolver, model_versions)
                for item in node.elts
            ]
        if isinstance(node, ast.UnaryOp):
            value = self._evaluate_node(
                node.operand, context, factor_resolver, model_versions
            )
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = self._evaluate_node(node.left, context, factor_resolver, model_versions)
            right = self._evaluate_node(node.right, context, factor_resolver, model_versions)
            operations = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                          ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
                          ast.Pow: lambda a, b: a**b}
            return operations[type(node.op)](left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id.upper()
            if name == "FACTOR":
                if factor_resolver is None:
                    raise FormulaError("factor references are unavailable at runtime")
                return factor_resolver(str(node.args[0].value), context)
            if name == "MODEL":
                model_id = str(node.args[0].value)
                return ModelRef(model_id, model_versions.get(model_id))
            operator = self.operators.get(name)
            args = [self._evaluate_node(arg, context, factor_resolver, model_versions)
                    for arg in node.args]
            kwargs = {item.arg: self._evaluate_node(
                item.value, context, factor_resolver, model_versions
            )
                      for item in node.keywords if item.arg is not None}
            return operator.function(context, *args, **kwargs)
        raise FormulaError(f"unsupported runtime node: {type(node).__name__}")
