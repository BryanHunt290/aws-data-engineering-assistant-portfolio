from datetime import datetime, timezone
from decimal import Decimal
import json

import pytest

from knowledge.application_models import (
    ApplicationResponse,
    ApplicationStatus,
    ModelMetadata,
    RetrievalMetadata,
)
from knowledge.costs import (
    CatalogCostEstimator,
    CostEstimate,
    DEMO_LABEL,
    ModelPricing,
)
from knowledge.intents import Intent
from knowledge.routing import Route
from ui.formatting import cost_details
from ui.session import (
    accumulate_response_cost,
    feedback_csv,
    feedback_json,
    record_feedback,
    session_cost_totals,
)


def _price(**overrides) -> ModelPricing:
    values = {
        "model_id": "fictional.model-v1",
        "currency": "USD",
        "input_price_per_million_tokens": Decimal("2.50"),
        "output_price_per_million_tokens": Decimal("7.75"),
        "cache_read_price_per_million_tokens": Decimal("0.25"),
        "cache_write_price_per_million_tokens": Decimal("3.125"),
        "pricing_source": "https://example.test/fictional-prices",
        "pricing_effective_date": "2026-01-15",
        "pricing_version": "fictional-v3",
        "regions": ("us-test-1",),
    }
    values.update(overrides)
    return ModelPricing(**values)


def _estimate(**overrides) -> CostEstimate:
    values = {
        "model_id": "fictional.model-v1",
        "input_token_count": 400,
        "output_token_count": 100,
        "input_cost": Decimal("0.001"),
        "output_cost": Decimal("0.000775"),
        "total_estimated_cost": Decimal("0.001775"),
        "currency": "USD",
        "pricing_version": "fictional-v3",
        "pricing_effective_date": "2026-01-15",
        "pricing_source": "https://example.test",
    }
    values.update(overrides)
    return CostEstimate(**values)


def _response(
    request_id: str,
    estimate: CostEstimate,
) -> ApplicationResponse:
    return ApplicationResponse(
        request_id=request_id,
        answer="answer",
        intent=Intent.ARCHITECTURE_DESIGN,
        route=Route.DIRECT_RESPONSE,
        confidence=1.0,
        sources=(),
        retrieval_metadata=RetrievalMetadata(
            attempted=False,
            result_count=0,
            requested_top_k=None,
            minimum_similarity=None,
            context_characters=0,
        ),
        model_metadata=ModelMetadata(
            provider_name="fictional",
            model_id=estimate.model_id,
            input_token_count=estimate.input_token_count,
            output_token_count=estimate.output_token_count,
            finish_reason="end_turn",
            latency_ms=1,
            cost_estimate=estimate,
        ),
        approval_required=False,
        safety_review_required=False,
        latency_ms=2,
        warnings=(),
        status=ApplicationStatus.COMPLETED,
    )


def test_decimal_cost_calculation_is_exact_and_includes_cache():
    estimator = CatalogCostEstimator([_price()])

    result = estimator.estimate(
        model_id="fictional.model-v1",
        input_token_count=400,
        output_token_count=100,
        cache_read_token_count=200,
        cache_write_token_count=40,
        region="us-test-1",
    )

    assert result.input_cost == Decimal("0.00100")
    assert result.output_cost == Decimal("0.000775")
    assert result.cache_read_cost == Decimal("0.00005")
    assert result.cache_write_cost == Decimal("0.00012500")
    assert result.total_estimated_cost == Decimal("0.00195000")
    assert result.pricing_version == "fictional-v3"


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "input_cost", "output_cost"),
    [
        (0, 0, Decimal("0.00"), Decimal("0.00")),
        (1_000_000, 0, Decimal("2.50"), Decimal("0.00")),
        (0, 1_000_000, Decimal("0.00"), Decimal("7.75")),
    ],
)
def test_zero_input_only_and_output_only_costs(
    input_tokens, output_tokens, input_cost, output_cost
):
    result = CatalogCostEstimator([_price()]).estimate(
        model_id="fictional.model-v1",
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        region="us-test-1",
    )

    assert result.input_cost == input_cost
    assert result.output_cost == output_cost


def test_unknown_model_preserves_usage_and_returns_unavailable():
    result = CatalogCostEstimator([_price()]).estimate(
        model_id="unknown.model",
        input_token_count=17,
        output_token_count=3,
        region="us-test-1",
    )

    assert result.available is False
    assert result.input_token_count == 17
    assert result.output_token_count == 3
    assert "no configured" in result.estimate_warning


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "missing_name"),
    [(None, 3, "inputTokens"), (7, None, "outputTokens")],
)
def test_missing_authoritative_usage_is_not_fabricated(
    input_tokens, output_tokens, missing_name
):
    result = CatalogCostEstimator([_price()]).estimate(
        model_id="fictional.model-v1",
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        region="us-test-1",
    )

    assert result.available is False
    assert result.input_token_count is input_tokens
    assert result.output_token_count is output_tokens
    assert missing_name in result.estimate_warning


def test_demo_mode_is_zero_and_not_an_aws_charge():
    result = CatalogCostEstimator([]).estimate(
        model_id="deterministic-demo",
        input_token_count=23,
        output_token_count=5,
        runtime_mode="demo",
    )

    assert result.formatted_total == "$0.000000"
    assert result.total_estimated_cost == Decimal("0")
    assert result.is_chargeable is False
    assert result.estimate_warning == DEMO_LABEL


def test_six_decimal_formatting_for_small_cost():
    estimate = _estimate(total_estimated_cost=Decimal("0.0001234"))

    assert estimate.formatted_total == "$0.000123"
    assert cost_details(estimate)["Total estimated cost"] == "$0.000123"


def test_cache_tokens_without_rate_make_total_unavailable():
    result = CatalogCostEstimator(
        [
            _price(
                cache_read_price_per_million_tokens=None,
                cache_write_price_per_million_tokens=None,
            )
        ]
    ).estimate(
        model_id="fictional.model-v1",
        input_token_count=10,
        output_token_count=5,
        cache_read_token_count=2,
        region="us-test-1",
    )

    assert result.available is False
    assert "cache-read" in result.estimate_warning


def test_session_accumulation_is_deduplicated_and_excludes_demo_charge():
    state = {}
    paid = _response("paid", _estimate())
    demo = _response(
        "demo",
        _estimate(
            input_token_count=25,
            output_token_count=5,
            input_cost=Decimal("0"),
            output_cost=Decimal("0"),
            total_estimated_cost=Decimal("0"),
            is_chargeable=False,
        ),
    )

    assert accumulate_response_cost(state, paid) is True
    assert accumulate_response_cost(state, paid) is False
    assert accumulate_response_cost(state, demo) is True
    totals = session_cost_totals(state)

    assert totals.request_count == 2
    assert totals.total_input_tokens == 425
    assert totals.total_output_tokens == 105
    assert totals.total_estimated_cost == Decimal("0.001775")


def test_feedback_exports_include_cost_fields_without_losing_old_fields():
    state = {}
    response = _response("request-1", _estimate())
    record_feedback(
        state,
        request_id=response.request_id,
        rating="up",
        comment="Useful",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        response=response,
    )

    payload = json.loads(feedback_json(state))[0]
    header = feedback_csv(state).splitlines()[0]

    assert payload["request_id"] == "request-1"
    assert payload["rating"] == "up"
    assert payload["model_id"] == "fictional.model-v1"
    assert payload["input_tokens"] == 400
    assert payload["estimated_total_cost"] == "0.001775"
    assert payload["pricing_version"] == "fictional-v3"
    assert "estimated_input_cost" in header


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {
            "schema_version": 2,
            "pricing_version": "v1",
            "pricing_effective_date": "2026-01-01",
            "pricing_source": "test",
            "models": [],
        },
        {
            "schema_version": 1,
            "pricing_version": "v1",
            "pricing_effective_date": "2026-01-01",
            "pricing_source": "test",
            "models": [{"model_id": "missing-rates"}],
        },
    ],
)
def test_pricing_catalog_validation_rejects_malformed_data(
    tmp_path, payload
):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        CatalogCostEstimator.from_json_file(path)


def test_pricing_catalog_rejects_float_money_and_duplicate_scope():
    with pytest.raises(ValueError, match="exact decimal"):
        _price(input_price_per_million_tokens=0.25)
    with pytest.raises(ValueError, match="duplicate"):
        CatalogCostEstimator([_price(), _price()])
