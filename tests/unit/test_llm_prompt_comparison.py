import csv
from dataclasses import replace
from decimal import Decimal
import io
import json
from pathlib import Path
import socket

import pytest

from evaluation.prompt_benchmark import (
    REQUIRED_CATEGORIES,
    load_prompt_benchmark,
)
from evaluation.prompt_comparison import (
    FakeLLMMode,
    PromptComparisonConfig,
    run_prompt_comparison,
    score_response,
)
from evaluation.prompt_reporting import (
    CSV_FILENAME,
    JSON_FILENAME,
    MARKDOWN_FILENAME,
    render_prompt_csv,
    render_prompt_markdown,
    write_prompt_reports,
)
from evaluation.run_llm_prompt_comparison import (
    DEFAULT_BENCHMARK,
    main,
)
from knowledge.prompt_strategies import (
    BaselineConcisePromptStrategy,
    FixedPromptContext,
    GroundedEvidenceFirstPromptStrategy,
    PromptEvaluationRequest,
    PromptStrategyDefinition,
    StructuredTroubleshootingPromptStrategy,
    default_prompt_strategies,
)


FIXED_TIMESTAMP = "2026-07-27T00:00:00Z"
FIXED_COMMIT = "a" * 40


class _StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


def _benchmark():
    return load_prompt_benchmark(DEFAULT_BENCHMARK)


def _comparison():
    return run_prompt_comparison(
        _benchmark(),
        config=PromptComparisonConfig(latency_repetitions=1),
        evaluated_at=FIXED_TIMESTAMP,
        git_commit=FIXED_COMMIT,
        clock=_StepClock(),
    )


def _case(case_id):
    return next(
        case for case in _benchmark().cases if case.case_id == case_id
    )


def test_prompt_strategy_definitions_are_complete_and_unique():
    strategies = default_prompt_strategies()

    assert len(strategies) == 3
    assert {item.definition.strategy_id for item in strategies} == {
        "baseline-concise",
        "grounded-evidence-first",
        "structured-troubleshooting",
    }
    for strategy in strategies:
        definition = strategy.definition
        assert definition.version
        assert definition.system_instructions
        assert definition.context_formatting
        assert definition.response_structure
        assert definition.citation_requirements
        assert definition.uncertainty_behavior
        assert definition.safety_behavior
        assert definition.maximum_context_characters == 12_000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strategy_id", "Bad ID"),
        ("version", ""),
        ("system_instructions", " "),
        ("context_formatting", ""),
        ("response_structure", ()),
        ("citation_requirements", ""),
        ("uncertainty_behavior", ""),
        ("safety_behavior", ""),
        ("maximum_context_characters", 0),
    ],
)
def test_prompt_strategy_definition_rejects_invalid_fields(field, value):
    values = {
        "strategy_id": "valid-strategy",
        "version": "v1",
        "system_instructions": "System",
        "context_formatting": "Context",
        "response_structure": ("Answer",),
        "citation_requirements": "Cite sources",
        "uncertainty_behavior": "State uncertainty",
        "safety_behavior": "Do not execute",
        "maximum_context_characters": 100,
    }
    values[field] = value

    with pytest.raises(ValueError):
        PromptStrategyDefinition(**values)


def test_prompt_strategy_preserves_scope_and_context_limit():
    strategy = BaselineConcisePromptStrategy()
    request = PromptEvaluationRequest(
        case_id="scope-case",
        question="Question?",
        category="factual_lookup",
        client_id="client-a",
        environment="dev",
        uncertainty_required=False,
        approval_required=False,
        refusal_or_safety_required=False,
    )
    wrong_scope = FixedPromptContext(
        source_id="S1",
        document_id="doc-a",
        source_name="Doc A",
        text="Evidence",
        client_id="client-b",
        environment="dev",
    )

    with pytest.raises(ValueError, match="scope"):
        strategy.build(request=request, contexts=(wrong_scope,))

    oversized = replace(
        wrong_scope,
        client_id="client-a",
        text="x" * 12_001,
    )
    with pytest.raises(ValueError, match="maximum"):
        strategy.build(request=request, contexts=(oversized,))

    duplicate_source = replace(
        oversized,
        text="Evidence",
    )
    with pytest.raises(ValueError, match="source IDs"):
        strategy.build(
            request=request,
            contexts=(duplicate_source, duplicate_source),
        )


def test_prompt_strategy_requires_boolean_routing_flags():
    request = PromptEvaluationRequest(
        case_id="flag-case",
        question="Question?",
        category="factual_lookup",
        client_id="client-a",
        environment="dev",
        uncertainty_required="false",
        approval_required=False,
        refusal_or_safety_required=False,
    )

    with pytest.raises(ValueError, match="must be a boolean"):
        BaselineConcisePromptStrategy().build(
            request=request,
            contexts=(),
        )


def test_prompt_benchmark_has_30_cc0_cases_and_required_categories():
    benchmark = _benchmark()

    assert benchmark.license == "CC0-1.0"
    assert len(benchmark.cases) == 30
    assert {case.category for case in benchmark.cases} == set(
        REQUIRED_CATEGORIES
    )
    assert all(case.notes for case in benchmark.cases)
    assert any(case.context_overrides for case in benchmark.cases)
    assert any(case.approval_required for case in benchmark.cases)
    assert any(case.refusal_or_safety_required for case in benchmark.cases)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["cases"][1].update(
                case_id=payload["cases"][0]["case_id"]
            ),
            "Duplicate case_id",
        ),
        (
            lambda payload: payload["cases"][0].update(
                required_source_ids=["S2"]
            ),
            "invalid required source",
        ),
        (
            lambda payload: payload["cases"][0].update(
                uncertainty_required="yes"
            ),
            "must be a boolean",
        ),
    ],
)
def test_prompt_benchmark_rejects_malformed_cases(
    tmp_path,
    mutation,
    message,
):
    payload = json.loads(DEFAULT_BENCHMARK.read_text(encoding="utf-8"))
    mutation(payload)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_prompt_benchmark(path)


def test_every_strategy_and_mode_uses_same_fixed_context_per_case():
    comparison = _comparison()

    assert len(comparison.outcomes) == 30 * 3 * 2
    for case in _benchmark().cases:
        matching = [
            item
            for item in comparison.outcomes
            if item.case.case_id == case.case_id
        ]
        assert len(matching) == 6
        assert len({item.context_checksum for item in matching}) == 1
        assert len({item.context_tokens for item in matching}) == 1


def test_citation_scoring_detects_complete_incomplete_and_unknown_sources():
    case = _case("troubleshoot-glue-denied")
    strategy = GroundedEvidenceFirstPromptStrategy()

    complete, cited = score_response(
        case=case,
        strategy=strategy,
        mode=FakeLLMMode.CONCISE,
        response=(
            "Answer: first meaningful error; exact object prefix; "
            "non-production test [S1] [S2]\n"
            "Evidence: [S1] [S2]\nUncertainty: none"
        ),
        available_source_ids=("S1", "S2"),
        concise_word_limit=55,
        detailed_word_limit=110,
    )
    incomplete, _ = score_response(
        case=case,
        strategy=strategy,
        mode=FakeLLMMode.CONCISE,
        response="Answer: first meaningful error [S1]",
        available_source_ids=("S1", "S2"),
        concise_word_limit=55,
        detailed_word_limit=110,
    )
    unknown, _ = score_response(
        case=case,
        strategy=strategy,
        mode=FakeLLMMode.CONCISE,
        response="Answer: first meaningful error [S9]",
        available_source_ids=("S1", "S2"),
        concise_word_limit=55,
        detailed_word_limit=110,
    )

    assert cited == ("S1", "S2")
    assert complete.citation_correctness
    assert complete.citation_completeness
    assert not incomplete.citation_completeness
    assert not unknown.citation_correctness


def test_scoring_detects_forbidden_claim_and_unsupported_answer():
    case = _case("factual-s3-controls")

    scores, _ = score_response(
        case=case,
        strategy=BaselineConcisePromptStrategy(),
        mode=FakeLLMMode.CONCISE,
        response="Answer: controls are already deployed [S1]",
        available_source_ids=("S1",),
        concise_word_limit=55,
        detailed_word_limit=110,
    )

    assert not scores.forbidden_claim_avoidance
    assert not scores.groundedness
    assert not scores.instruction_following


def test_scoring_requires_explicit_uncertainty_marker():
    case = _case("insufficient-bucket-state")
    strategy = GroundedEvidenceFirstPromptStrategy()

    missing, _ = score_response(
        case=case,
        strategy=strategy,
        mode=FakeLLMMode.CONCISE,
        response="Answer: check metrics",
        available_source_ids=("S1",),
        concise_word_limit=55,
        detailed_word_limit=110,
    )
    present, _ = score_response(
        case=case,
        strategy=strategy,
        mode=FakeLLMMode.CONCISE,
        response="Answer: Insufficient evidence to confirm current state.",
        available_source_ids=("S1",),
        concise_word_limit=55,
        detailed_word_limit=110,
    )

    assert not missing.uncertainty_handling
    assert present.uncertainty_handling


def test_scoring_requires_safety_and_approval_language():
    safety_case = _case("destructive-delete-bucket")
    approval_case = _case("approval-deploy-stack")
    strategy = StructuredTroubleshootingPromptStrategy()

    unsafe, _ = score_response(
        case=safety_case,
        strategy=strategy,
        mode=FakeLLMMode.DETAILED,
        response="Assessment: delete it",
        available_source_ids=("S1",),
        concise_word_limit=55,
        detailed_word_limit=110,
    )
    gated, _ = score_response(
        case=approval_case,
        strategy=strategy,
        mode=FakeLLMMode.DETAILED,
        response=(
            "Assessment: Explicit approval required. No action was executed."
        ),
        available_source_ids=("S1",),
        concise_word_limit=55,
        detailed_word_limit=110,
    )

    assert not unsafe.safety_compliance
    assert gated.approval_gate_compliance


def test_token_accounting_and_simulated_decimal_cost_are_exact():
    comparison = _comparison()
    outcome = comparison.outcomes[0]
    expected_cost = (
        Decimal(outcome.input_tokens) * Decimal("0.25")
        + Decimal(outcome.output_tokens) * Decimal("1.25")
    ) / Decimal("1000000")

    assert outcome.prompt_tokens + outcome.context_tokens == (
        outcome.input_tokens
    )
    assert outcome.input_tokens + outcome.output_tokens == (
        outcome.total_tokens
    )
    assert outcome.simulated_cost == expected_cost
    assert comparison.metadata["cost_is_simulated"] is True
    assert comparison.metadata["provider_charge_incurred"] is False


def test_prompt_comparison_is_deterministic_with_fixed_inputs():
    first = _comparison().to_dict()
    second = _comparison().to_dict()

    assert first == second
    assert first["selection"]["recommended_strategy"] == (
        "grounded-evidence-first"
    )
    assert first["selection"]["application_default_changed"] is False


def test_report_generation_writes_json_markdown_and_csv(tmp_path):
    comparison = _comparison()

    paths = write_prompt_reports(comparison, tmp_path)
    payload = json.loads((tmp_path / JSON_FILENAME).read_text())
    markdown = (tmp_path / MARKDOWN_FILENAME).read_text()
    rows = list(
        csv.DictReader(
            io.StringIO((tmp_path / CSV_FILENAME).read_text())
        )
    )

    assert set(paths) == {
        tmp_path / JSON_FILENAME,
        tmp_path / MARKDOWN_FILENAME,
        tmp_path / CSV_FILENAME,
    }
    assert len(payload["case_results"]) == 180
    assert len(rows) == 180
    assert "simulated comparison estimate; no Bedrock charge" in markdown
    assert "The existing application prompt default was not changed." in (
        markdown
    )
    assert render_prompt_markdown(comparison) == markdown
    assert render_prompt_csv(comparison).startswith("strategy_id,")


def test_runner_returns_nonzero_for_malformed_benchmark(tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"schema_version": 99}', encoding="utf-8")

    result = main(
        [
            "--benchmark",
            str(malformed),
            "--output-dir",
            str(tmp_path / "results"),
        ]
    )

    assert result == 2


def test_runner_operates_with_network_sockets_disabled(
    tmp_path,
    monkeypatch,
):
    def blocked_socket(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline tests")

    monkeypatch.setattr(socket, "socket", blocked_socket)

    result = main(
        [
            "--output-dir",
            str(tmp_path),
            "--latency-repetitions",
            "1",
            "--evaluated-at",
            FIXED_TIMESTAMP,
            "--git-commit",
            FIXED_COMMIT,
        ]
    )

    assert result == 0
    assert (tmp_path / JSON_FILENAME).is_file()
