"""Provider-neutral, offline LLM request cost estimation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


TOKENS_PER_MILLION = Decimal("1000000")
DISPLAY_QUANTUM = Decimal("0.000001")
DEFAULT_CATALOG_PATH = Path(__file__).with_name("pricing_catalog.json")
DEMO_PRICING_VERSION = "demo-no-charge-v1"
DEMO_LABEL = "Demo mode — no Bedrock charge incurred"


def _token_count(name: str, value: int | None) -> int | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


def _money(name: str, value: Decimal | str | int) -> Decimal:
    if isinstance(value, float) or isinstance(value, bool):
        raise ValueError(f"{name} must use an exact decimal value")
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError) as error:
        raise ValueError(f"{name} must be a valid decimal") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class ModelPricing:
    """One model/Region on-demand price entry."""

    model_id: str
    currency: str
    input_price_per_million_tokens: Decimal
    output_price_per_million_tokens: Decimal
    pricing_source: str
    pricing_effective_date: str
    pricing_version: str
    regions: tuple[str, ...] = ()
    cache_read_price_per_million_tokens: Decimal | None = None
    cache_write_price_per_million_tokens: Decimal | None = None

    def __post_init__(self) -> None:
        model_id = self.model_id.strip()
        currency = self.currency.strip().upper()
        source = self.pricing_source.strip()
        version = self.pricing_version.strip()
        if not model_id:
            raise ValueError("model_id cannot be empty")
        if len(currency) != 3 or not currency.isalpha():
            raise ValueError("currency must be a three-letter code")
        if not source:
            raise ValueError("pricing_source cannot be empty")
        if not version:
            raise ValueError("pricing_version cannot be empty")
        try:
            date.fromisoformat(self.pricing_effective_date)
        except ValueError as error:
            raise ValueError(
                "pricing_effective_date must be an ISO date"
            ) from error
        regions = tuple(region.strip().lower() for region in self.regions)
        if any(not region for region in regions) or len(set(regions)) != len(
            regions
        ):
            raise ValueError("regions must be unique and non-empty")
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "pricing_source", source)
        object.__setattr__(self, "pricing_version", version)
        object.__setattr__(self, "regions", regions)
        for field_name in (
            "input_price_per_million_tokens",
            "output_price_per_million_tokens",
            "cache_read_price_per_million_tokens",
            "cache_write_price_per_million_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self, field_name, _money(field_name, value)
                )


@dataclass(frozen=True)
class CostEstimate:
    """Typed estimate; unavailable components are represented by ``None``."""

    model_id: str
    input_token_count: int | None
    output_token_count: int | None
    cache_read_token_count: int | None = None
    cache_write_token_count: int | None = None
    input_cost: Decimal | None = None
    output_cost: Decimal | None = None
    cache_read_cost: Decimal | None = None
    cache_write_cost: Decimal | None = None
    total_estimated_cost: Decimal | None = None
    currency: str | None = None
    pricing_version: str | None = None
    pricing_effective_date: str | None = None
    pricing_source: str | None = None
    input_price_per_million_tokens: Decimal | None = None
    output_price_per_million_tokens: Decimal | None = None
    cache_read_price_per_million_tokens: Decimal | None = None
    cache_write_price_per_million_tokens: Decimal | None = None
    is_estimated: bool = True
    is_chargeable: bool = True
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")
        for name in (
            "input_token_count",
            "output_token_count",
            "cache_read_token_count",
            "cache_write_token_count",
        ):
            _token_count(name, getattr(self, name))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def available(self) -> bool:
        return self.total_estimated_cost is not None

    @property
    def formatted_total(self) -> str:
        if self.total_estimated_cost is None:
            return "Unavailable"
        amount = self.total_estimated_cost.quantize(DISPLAY_QUANTUM)
        prefix = "$" if self.currency == "USD" else f"{self.currency} "
        return f"{prefix}{amount:.6f}"

    @property
    def estimate_warning(self) -> str | None:
        return "; ".join(self.warnings) if self.warnings else None


@runtime_checkable
class CostEstimator(Protocol):
    """Provider-neutral LLM cost estimation contract."""

    def estimate(
        self,
        *,
        model_id: str,
        input_token_count: int | None,
        output_token_count: int | None,
        cache_read_token_count: int | None = None,
        cache_write_token_count: int | None = None,
        region: str | None = None,
        runtime_mode: str = "bedrock",
    ) -> CostEstimate:
        """Estimate a request without network or billing API calls."""


class CatalogCostEstimator:
    """Estimate from a validated, in-memory price catalog."""

    def __init__(self, prices: Sequence[ModelPricing]) -> None:
        self._prices = tuple(prices)
        seen: set[tuple[str, str]] = set()
        for price in self._prices:
            regions = price.regions or ("*",)
            for region in regions:
                key = (price.model_id, region)
                if key in seen:
                    raise ValueError(
                        "pricing catalog contains duplicate model/Region entry"
                    )
                seen.add(key)

    @classmethod
    def from_json_file(cls, path: str | Path) -> CatalogCostEstimator:
        source_path = Path(path)
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Unable to load pricing catalog: {source_path}"
            ) from error
        return cls(cls._parse_catalog(payload))

    @classmethod
    def _parse_catalog(
        cls, payload: Any
    ) -> tuple[ModelPricing, ...]:
        if not isinstance(payload, dict):
            raise ValueError("pricing catalog must be a JSON object")
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported pricing catalog schema_version")
        for name in (
            "pricing_version",
            "pricing_effective_date",
            "pricing_source",
        ):
            if not isinstance(payload.get(name), str) or not payload[
                name
            ].strip():
                raise ValueError(f"pricing catalog {name} is required")
        models = payload.get("models")
        if not isinstance(models, list):
            raise ValueError("pricing catalog models must be a list")
        shared = {
            "pricing_version": payload["pricing_version"],
            "pricing_effective_date": payload["pricing_effective_date"],
            "pricing_source": payload["pricing_source"],
        }
        entries: list[ModelPricing] = []
        for raw in models:
            if not isinstance(raw, dict):
                raise ValueError("each pricing model must be an object")
            required = {
                "model_id",
                "currency",
                "input_price_per_million_tokens",
                "output_price_per_million_tokens",
            }
            if not required.issubset(raw):
                raise ValueError("pricing model is missing required fields")
            regions = raw.get("regions", [])
            if not isinstance(regions, list) or not all(
                isinstance(region, str) for region in regions
            ):
                raise ValueError("pricing model regions must be a list")
            values: dict[str, Any] = dict(shared)
            values.update(
                {
                    key: raw[key]
                    for key in (
                        "model_id",
                        "currency",
                        "input_price_per_million_tokens",
                        "output_price_per_million_tokens",
                    )
                }
            )
            values["regions"] = tuple(regions)
            for optional in (
                "cache_read_price_per_million_tokens",
                "cache_write_price_per_million_tokens",
            ):
                if optional in raw:
                    values[optional] = raw[optional]
            entries.append(ModelPricing(**values))
        return tuple(entries)

    def estimate(
        self,
        *,
        model_id: str,
        input_token_count: int | None,
        output_token_count: int | None,
        cache_read_token_count: int | None = None,
        cache_write_token_count: int | None = None,
        region: str | None = None,
        runtime_mode: str = "bedrock",
    ) -> CostEstimate:
        counts = {
            "input_token_count": _token_count(
                "input_token_count", input_token_count
            ),
            "output_token_count": _token_count(
                "output_token_count", output_token_count
            ),
            "cache_read_token_count": _token_count(
                "cache_read_token_count", cache_read_token_count
            ),
            "cache_write_token_count": _token_count(
                "cache_write_token_count", cache_write_token_count
            ),
        }
        if runtime_mode.strip().lower() == "demo":
            cache_read_cost = (
                Decimal("0")
                if cache_read_token_count is not None
                else None
            )
            cache_write_cost = (
                Decimal("0")
                if cache_write_token_count is not None
                else None
            )
            return CostEstimate(
                model_id=model_id,
                **counts,
                input_cost=Decimal("0"),
                output_cost=Decimal("0"),
                cache_read_cost=cache_read_cost,
                cache_write_cost=cache_write_cost,
                total_estimated_cost=Decimal("0"),
                currency="USD",
                pricing_version=DEMO_PRICING_VERSION,
                pricing_source="Offline deterministic demo",
                pricing_effective_date=date.today().isoformat(),
                is_chargeable=False,
                warnings=(DEMO_LABEL,),
            )
        if input_token_count is None or output_token_count is None:
            missing = []
            if input_token_count is None:
                missing.append("inputTokens")
            if output_token_count is None:
                missing.append("outputTokens")
            return CostEstimate(
                model_id=model_id,
                **counts,
                warnings=(
                    "Cost unavailable because Bedrock usage omitted "
                    + " and ".join(missing)
                    + "; token counts were not fabricated.",
                ),
            )
        price = self._find_price(model_id, region)
        if price is None:
            location = f" in Region {region}" if region else ""
            return CostEstimate(
                model_id=model_id,
                **counts,
                warnings=(
                    f"Cost unavailable: no configured on-demand price for "
                    f"model {model_id}{location}.",
                ),
            )
        cache_warning = self._missing_cache_price(
            cache_read_token_count,
            cache_write_token_count,
            price,
        )
        if cache_warning:
            return CostEstimate(
                model_id=model_id,
                **counts,
                currency=price.currency,
                pricing_version=price.pricing_version,
                pricing_effective_date=price.pricing_effective_date,
                pricing_source=price.pricing_source,
                warnings=(cache_warning,),
            )
        input_cost = self._component(
            input_token_count, price.input_price_per_million_tokens
        )
        output_cost = self._component(
            output_token_count, price.output_price_per_million_tokens
        )
        cache_read_cost = (
            self._component(
                cache_read_token_count,
                price.cache_read_price_per_million_tokens
                or Decimal("0"),
            )
            if cache_read_token_count is not None
            else None
        )
        cache_write_cost = (
            self._component(
                cache_write_token_count,
                price.cache_write_price_per_million_tokens
                or Decimal("0"),
            )
            if cache_write_token_count is not None
            else None
        )
        return CostEstimate(
            model_id=model_id,
            **counts,
            input_cost=input_cost,
            output_cost=output_cost,
            cache_read_cost=cache_read_cost,
            cache_write_cost=cache_write_cost,
            total_estimated_cost=(
                input_cost
                + output_cost
                + (cache_read_cost or Decimal("0"))
                + (cache_write_cost or Decimal("0"))
            ),
            currency=price.currency,
            pricing_version=price.pricing_version,
            pricing_effective_date=price.pricing_effective_date,
            pricing_source=price.pricing_source,
            input_price_per_million_tokens=(
                price.input_price_per_million_tokens
            ),
            output_price_per_million_tokens=(
                price.output_price_per_million_tokens
            ),
            cache_read_price_per_million_tokens=(
                price.cache_read_price_per_million_tokens
            ),
            cache_write_price_per_million_tokens=(
                price.cache_write_price_per_million_tokens
            ),
        )

    def _find_price(
        self, model_id: str, region: str | None
    ) -> ModelPricing | None:
        normalized_region = region.strip().lower() if region else None
        global_match = None
        for price in self._prices:
            if price.model_id != model_id:
                continue
            if not price.regions:
                global_match = price
            elif normalized_region in price.regions:
                return price
        return global_match

    @staticmethod
    def _component(tokens: int, rate: Decimal) -> Decimal:
        return Decimal(tokens) / TOKENS_PER_MILLION * rate

    @staticmethod
    def _missing_cache_price(
        read_tokens: int | None,
        write_tokens: int | None,
        price: ModelPricing,
    ) -> str | None:
        missing = []
        if read_tokens and price.cache_read_price_per_million_tokens is None:
            missing.append("cache-read")
        if write_tokens and price.cache_write_price_per_million_tokens is None:
            missing.append("cache-write")
        if not missing:
            return None
        return (
            "Cost unavailable because "
            + " and ".join(missing)
            + " token pricing is not configured."
        )


def load_cost_estimator(
    catalog_path: str | Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> CatalogCostEstimator:
    """Load an offline catalog, optionally selected through environment."""

    values = os.environ if environment is None else environment
    selected = (
        catalog_path
        or values.get("APP_PRICING_CATALOG_PATH")
        or values.get("DEA_PRICING_CATALOG_PATH")
        or DEFAULT_CATALOG_PATH
    )
    return CatalogCostEstimator.from_json_file(selected)
