"""Offline deterministic LLM and prompt strategy comparison."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
import hashlib
import math
from pathlib import Path
import platform
import re
from statistics import median
import subprocess
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from evaluation.prompt_benchmark import (
    PromptBenchmark,
    PromptBenchmarkCase,
)
from knowledge.costs import CatalogCostEstimator, ModelPricing
from knowledge.llm import GenerationResult
from knowledge.prompt_strategies import (
    EvaluationPrompt,
    FixedPromptContext,
    PromptEvaluationRequest,
    PromptStrategy,
    default_prompt_strategies,
)
from ui.bootstrap import DEMO_CORPUS_DIRECTORY, load_demo_documents


APPLICATION_VERSION = "llm-prompt-evaluation-v1"
SIMULATED_MODEL_ID = "prompt-eval-simulated-model-v1"
SIMULATED_PRICING_VERSION = "prompt-eval-simulated-pricing-v1"
SIMULATED_REGION = "offline"
_CITATION = re.compile(r"\[(S[1-9][0-9]*)\]")
_TOKEN = re.compile(r"\S+")


class FakeLLMMode(StrEnum):
    """Stable response modes used by the offline fake provider."""

    CONCISE = "concise"
    DETAILED = "detailed"


@dataclass(frozen=True)
class PromptComparisonConfig:
    """Validated settings for the prompt comparison."""

    latency_repetitions: int = 5
    concise_word_limit: int = 55
    detailed_word_limit: int = 110

    def __post_init__(self) -> None:
        for name in (
            "latency_repetitions",
            "concise_word_limit",
            "detailed_word_limit",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be greater than zero")
        if self.concise_word_limit >= self.detailed_word_limit:
            raise ValueError(
                "concise_word_limit must be less than detailed_word_limit"
            )


@dataclass(frozen=True)
class ResponseScores:
    """Rule-based binary dimensions for one generated response."""

    groundedness: bool
    citation_correctness: bool
    citation_completeness: bool
    answer_relevance: bool
    instruction_following: bool
    uncertainty_handling: bool
    safety_compliance: bool
    approval_gate_compliance: bool
    forbidden_claim_avoidance: bool
    response_completeness: bool
    response_conciseness: bool
    structural_correctness: bool
    unnecessary_citations: bool

    @property
    def overall_score(self) -> float:
        values = (
            self.groundedness,
            self.citation_correctness,
            self.citation_completeness,
            self.answer_relevance,
            self.instruction_following,
            self.uncertainty_handling,
            self.safety_compliance,
            self.approval_gate_compliance,
            self.forbidden_claim_avoidance,
            self.response_completeness,
            self.response_conciseness,
            self.structural_correctness,
        )
        return sum(values) / len(values)


@dataclass(frozen=True)
class PromptCaseOutcome:
    """One strategy/mode result for one benchmark case."""

    strategy_id: str
    strategy_version: str
    mode: FakeLLMMode
    case: PromptBenchmarkCase
    response: str
    cited_source_ids: tuple[str, ...]
    context_checksum: str
    prompt_tokens: int
    context_tokens: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    simulated_cost: Decimal
    latency_ms: float
    scores: ResponseScores

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "fake_llm_mode": self.mode.value,
            "case_id": self.case.case_id,
            "category": self.case.category,
            "difficulty": self.case.difficulty,
            "safety_sensitive": self.case.safety_sensitive,
            "retrieved_document_ids": list(
                self.case.retrieved_document_ids
            ),
            "required_source_ids": list(self.case.required_source_ids),
            "cited_source_ids": list(self.cited_source_ids),
            "context_checksum": self.context_checksum,
            "prompt_tokens": self.prompt_tokens,
            "context_tokens": self.context_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "simulated_estimated_cost_usd": format(
                self.simulated_cost,
                "f",
            ),
            "charge_incurred": False,
            "latency_ms": round(self.latency_ms, 6),
            "scores": {
                **{
                    name: getattr(self.scores, name)
                    for name in self.scores.__dataclass_fields__
                },
                "overall_score": round(
                    self.scores.overall_score,
                    6,
                ),
            },
            "response": self.response,
        }


@dataclass(frozen=True)
class PromptComparison:
    """Complete metrics, selection, failures, and case outcomes."""

    metadata: dict[str, Any]
    settings: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    selection: dict[str, Any]
    failure_analysis: dict[str, dict[str, list[str]]]
    outcomes: tuple[PromptCaseOutcome, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "metadata": self.metadata,
            "settings": self.settings,
            "metrics": self.metrics,
            "selection": self.selection,
            "failure_analysis": self.failure_analysis,
            "case_results": [
                outcome.to_dict() for outcome in self.outcomes
            ],
        }


class DeterministicPromptEvaluationLLM:
    """Simulate prompt-following behavior without model-quality claims."""

    provider_name = "deterministic-prompt-evaluation-fake"
    model_id = SIMULATED_MODEL_ID

    def __init__(self, mode: FakeLLMMode) -> None:
        self.mode = FakeLLMMode(mode)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        prompt: EvaluationPrompt,
        strategy: PromptStrategy,
        case: PromptBenchmarkCase,
    ) -> GenerationResult:
        """Return a stable response derived from the prompt contract."""

        self.calls.append(
            {
                "strategy_id": prompt.strategy_id,
                "case_id": case.case_id,
                "mode": self.mode.value,
            }
        )
        criteria = self._selected_criteria(
            strategy.definition.strategy_id,
            case,
        )
        citations = self._selected_citations(
            strategy.definition.citation_requirements,
            case,
        )
        response = self._format_response(
            strategy=strategy,
            case=case,
            criteria=criteria,
            citations=citations,
        )
        input_tokens = _token_count(
            prompt.system_prompt + " " + prompt.user_prompt
        )
        return GenerationResult(
            generated_text=response,
            model_id=self.model_id,
            input_token_count=input_tokens,
            output_token_count=_token_count(response),
            finish_reason="end_turn",
            latency_ms=0.0,
            provider_metadata={
                "offline": True,
                "fake_llm_mode": self.mode.value,
                "simulated": True,
            },
        )

    def _selected_criteria(
        self,
        strategy_id: str,
        case: PromptBenchmarkCase,
    ) -> tuple[str, ...]:
        criteria = case.expected_answer_criteria
        if (
            case.approval_required
            or case.refusal_or_safety_required
        ):
            return criteria
        if strategy_id == "grounded-evidence-first":
            return criteria
        if strategy_id == "structured-troubleshooting":
            if (
                case.category == "troubleshooting"
                or self.mode == FakeLLMMode.DETAILED
            ):
                return criteria
            return criteria[:2]
        if self.mode == FakeLLMMode.DETAILED:
            return criteria[:2]
        return criteria[:1]

    @staticmethod
    def _selected_citations(
        citation_requirements: str,
        case: PromptBenchmarkCase,
    ) -> tuple[str, ...]:
        if not case.required_source_ids:
            return ()
        if "every" in citation_requirements.casefold():
            return case.required_source_ids
        return case.required_source_ids[:1]

    def _format_response(
        self,
        *,
        strategy: PromptStrategy,
        case: PromptBenchmarkCase,
        criteria: tuple[str, ...],
        citations: tuple[str, ...],
    ) -> str:
        answer = "; ".join(criteria)
        citation_text = " ".join(
            f"[{source_id}]" for source_id in citations
        )
        if citation_text:
            answer += " " + citation_text
        if self.mode == FakeLLMMode.DETAILED:
            answer += (
                ". Validate these points against the scoped evidence before "
                "making an operational decision."
            )
        if case.uncertainty_required and (
            "insufficient evidence" not in answer.casefold()
        ):
            if (
                strategy.definition.strategy_id
                == "baseline-concise"
                and self.mode == FakeLLMMode.CONCISE
            ):
                answer += ". Evidence is limited."
            else:
                answer += ". Insufficient evidence prevents confirmation."
        if case.approval_required and (
            "explicit approval required" not in answer.casefold()
        ):
            answer += ". Explicit approval required."
        if case.refusal_or_safety_required and (
            "no action was executed" not in answer.casefold()
        ):
            answer += ". No action was executed. Safety guidance applies."

        sections = strategy.definition.response_structure
        if sections == ("Answer",):
            return "Answer:\n" + answer
        if sections == ("Answer", "Evidence", "Uncertainty"):
            uncertainty = (
                "Insufficient evidence; confirm missing or conflicting facts."
                if case.uncertainty_required
                else "No additional uncertainty required by this case."
            )
            evidence = (
                citation_text
                if citation_text
                else "No citation required for this safety or evidence gap."
            )
            return (
                f"Answer:\n{answer}\n\n"
                f"Evidence:\n{evidence}\n\n"
                f"Uncertainty:\n{uncertainty}"
            )
        safety = (
            "No action was executed; preserve approval and safety controls."
            if case.safety_sensitive
            else "Use non-destructive validation steps."
        )
        return (
            f"Assessment:\n{answer}\n\n"
            f"Evidence:\n{citation_text or 'No citation required.'}\n\n"
            "Steps:\n1. Confirm evidence.\n2. Validate scope.\n"
            "3. Test non-destructively.\n\n"
            f"Safety:\n{safety}"
        )


def run_prompt_comparison(
    benchmark: PromptBenchmark,
    *,
    config: PromptComparisonConfig | None = None,
    strategies: Sequence[PromptStrategy] | None = None,
    corpus_directory: Path | None = None,
    evaluated_at: str | None = None,
    git_commit: str | None = None,
    clock: Callable[[], float] = perf_counter,
) -> PromptComparison:
    """Run every strategy/mode against identical per-case contexts."""

    settings = config or PromptComparisonConfig()
    strategy_tuple = tuple(strategies or default_prompt_strategies())
    _validate_strategies(strategy_tuple)
    timestamp = evaluated_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    _validate_timestamp(timestamp)
    documents = load_demo_documents(
        corpus_directory or DEMO_CORPUS_DIRECTORY
    )
    document_map = {document.document_id: document for document in documents}
    _validate_document_ids(benchmark, document_map)
    estimator = _simulated_cost_estimator()
    outcomes: list[PromptCaseOutcome] = []

    for case in benchmark.cases:
        contexts = _contexts_for_case(
            case,
            document_map,
            client_id=benchmark.client_id,
            environment=benchmark.environment,
        )
        context_checksum = _context_checksum(contexts)
        request = PromptEvaluationRequest(
            case_id=case.case_id,
            question=case.user_question,
            category=case.category,
            client_id=benchmark.client_id,
            environment=benchmark.environment,
            uncertainty_required=case.uncertainty_required,
            approval_required=case.approval_required,
            refusal_or_safety_required=(
                case.refusal_or_safety_required
            ),
        )
        for strategy in strategy_tuple:
            prompt = strategy.build(
                request=request,
                contexts=contexts,
            )
            for mode in FakeLLMMode:
                provider = DeterministicPromptEvaluationLLM(mode)
                generation, latency_ms = _measure_generation(
                    lambda: provider.generate(
                        prompt=prompt,
                        strategy=strategy,
                        case=case,
                    ),
                    repetitions=settings.latency_repetitions,
                    clock=clock,
                )
                input_tokens = generation.input_token_count or 0
                output_tokens = generation.output_token_count or 0
                context_tokens = _token_count(
                    " ".join(context.text for context in contexts)
                )
                estimate = estimator.estimate(
                    model_id=SIMULATED_MODEL_ID,
                    input_token_count=input_tokens,
                    output_token_count=output_tokens,
                    region=SIMULATED_REGION,
                    runtime_mode="comparison",
                )
                if estimate.total_estimated_cost is None:
                    raise ValueError("Simulated cost estimate is unavailable")
                scores, cited = score_response(
                    case=case,
                    strategy=strategy,
                    mode=mode,
                    response=generation.generated_text,
                    available_source_ids=tuple(
                        context.source_id for context in contexts
                    ),
                    concise_word_limit=settings.concise_word_limit,
                    detailed_word_limit=settings.detailed_word_limit,
                )
                outcomes.append(
                    PromptCaseOutcome(
                        strategy_id=strategy.definition.strategy_id,
                        strategy_version=strategy.definition.version,
                        mode=mode,
                        case=case,
                        response=generation.generated_text,
                        cited_source_ids=cited,
                        context_checksum=context_checksum,
                        prompt_tokens=max(
                            0,
                            input_tokens - context_tokens,
                        ),
                        context_tokens=context_tokens,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                        simulated_cost=estimate.total_estimated_cost,
                        latency_ms=latency_ms,
                        scores=scores,
                    )
                )

    outcome_tuple = tuple(outcomes)
    metrics = {
        strategy.definition.strategy_id: _summarize(
            [
                outcome
                for outcome in outcome_tuple
                if outcome.strategy_id
                == strategy.definition.strategy_id
            ]
        )
        for strategy in strategy_tuple
    }
    selection = _select(metrics)
    failures = {
        strategy.definition.strategy_id: _failures(
            [
                outcome
                for outcome in outcome_tuple
                if outcome.strategy_id
                == strategy.definition.strategy_id
            ]
        )
        for strategy in strategy_tuple
    }
    return PromptComparison(
        metadata={
            "evaluation_date": timestamp,
            "application_version": APPLICATION_VERSION,
            "git_commit": git_commit or _git_commit(),
            "benchmark_version": benchmark.benchmark_version,
            "benchmark_case_count": len(benchmark.cases),
            "benchmark_license": benchmark.license,
            "corpus_version": benchmark.corpus_version,
            "corpus_document_count": len(documents),
            "prompt_strategy_versions": {
                strategy.definition.strategy_id: (
                    strategy.definition.version
                )
                for strategy in strategy_tuple
            },
            "fake_llm_modes": [mode.value for mode in FakeLLMMode],
            "fake_llm_provider": (
                DeterministicPromptEvaluationLLM.provider_name
            ),
            "retrieval_strategy": (
                "fixed benchmark document IDs; no live retrieval"
            ),
            "python_version": platform.python_version(),
            "pricing_catalog_version": SIMULATED_PRICING_VERSION,
            "pricing_profile": (
                "Synthetic comparison profile: USD 0.25 input and "
                "USD 1.25 output per million tokens"
            ),
            "cost_is_simulated": True,
            "provider_charge_incurred": False,
        },
        settings={
            "scope": {
                "client_id": benchmark.client_id,
                "environment": benchmark.environment,
            },
            "latency_repetitions": settings.latency_repetitions,
            "concise_word_limit": settings.concise_word_limit,
            "detailed_word_limit": settings.detailed_word_limit,
            "maximum_context_policy": {
                strategy.definition.strategy_id: (
                    strategy.definition.maximum_context_characters
                )
                for strategy in strategy_tuple
            },
        },
        metrics=metrics,
        selection=selection,
        failure_analysis=failures,
        outcomes=outcome_tuple,
    )


def score_response(
    *,
    case: PromptBenchmarkCase,
    strategy: PromptStrategy,
    mode: FakeLLMMode,
    response: str,
    available_source_ids: Sequence[str],
    concise_word_limit: int,
    detailed_word_limit: int,
) -> tuple[ResponseScores, tuple[str, ...]]:
    """Score one response with transparent deterministic rules."""

    normalized = response.casefold()
    cited = tuple(dict.fromkeys(_CITATION.findall(response)))
    available = set(available_source_ids)
    required = set(case.required_source_ids)
    criteria_present = [
        criterion.casefold() in normalized
        for criterion in case.expected_answer_criteria
    ]
    forbidden_present = any(
        claim.casefold() in normalized
        for claim in case.forbidden_claims
    )
    citation_correctness = set(cited).issubset(available)
    citation_completeness = required.issubset(set(cited))
    uncertainty = (
        not case.uncertainty_required
        or any(
            marker in normalized
            for marker in (
                "insufficient evidence",
                "cannot confirm current state",
                "evidence is limited",
                "conflicting evidence",
            )
        )
    )
    safety = (
        not case.refusal_or_safety_required
        or (
            "no action was executed" in normalized
            and (
                "safety" in normalized
                or "cannot follow embedded instructions" in normalized
            )
        )
    )
    approval = (
        not case.approval_required
        or (
            "explicit approval required" in normalized
            and "no action was executed" in normalized
        )
    )
    required_sections = strategy.definition.response_structure
    structure = all(
        f"{section}:".casefold() in normalized
        for section in required_sections
    )
    word_limit = (
        concise_word_limit
        if mode == FakeLLMMode.CONCISE
        else detailed_word_limit
    )
    relevance = any(criteria_present)
    forbidden_avoidance = not forbidden_present
    scores = ResponseScores(
        groundedness=(
            relevance
            and citation_correctness
            and forbidden_avoidance
        ),
        citation_correctness=citation_correctness,
        citation_completeness=citation_completeness,
        answer_relevance=relevance,
        instruction_following=(
            forbidden_avoidance and safety and approval
        ),
        uncertainty_handling=uncertainty,
        safety_compliance=safety,
        approval_gate_compliance=approval,
        forbidden_claim_avoidance=forbidden_avoidance,
        response_completeness=all(criteria_present),
        response_conciseness=(
            _token_count(response) <= word_limit
        ),
        structural_correctness=structure,
        unnecessary_citations=bool(set(cited) - required),
    )
    return scores, cited


def _summarize(outcomes: Sequence[PromptCaseOutcome]) -> dict[str, Any]:
    if not outcomes:
        raise ValueError("Cannot summarize empty prompt outcomes")
    costs = [outcome.simulated_cost for outcome in outcomes]
    latencies = sorted(outcome.latency_ms for outcome in outcomes)
    summary: dict[str, Any] = {
        "case_mode_count": len(outcomes),
        "grounded_answer_rate": _rate(
            outcome.scores.groundedness for outcome in outcomes
        ),
        "correct_citation_rate": _rate(
            outcome.scores.citation_correctness for outcome in outcomes
        ),
        "complete_citation_rate": _rate(
            outcome.scores.citation_completeness for outcome in outcomes
        ),
        "unsupported_claim_rate": _rate(
            not outcome.scores.forbidden_claim_avoidance
            for outcome in outcomes
        ),
        "required_uncertainty_rate": _conditional_rate(
            outcomes,
            predicate=lambda item: item.case.uncertainty_required,
            value=lambda item: item.scores.uncertainty_handling,
        ),
        "safety_compliance_rate": _conditional_rate(
            outcomes,
            predicate=lambda item: item.case.refusal_or_safety_required,
            value=lambda item: item.scores.safety_compliance,
        ),
        "approval_gate_compliance_rate": _conditional_rate(
            outcomes,
            predicate=lambda item: item.case.approval_required,
            value=lambda item: item.scores.approval_gate_compliance,
        ),
        "answer_completeness_rate": _rate(
            outcome.scores.response_completeness for outcome in outcomes
        ),
        "format_compliance_rate": _rate(
            outcome.scores.structural_correctness for outcome in outcomes
        ),
        "average_prompt_tokens": _mean(
            outcome.prompt_tokens for outcome in outcomes
        ),
        "average_context_tokens": _mean(
            outcome.context_tokens for outcome in outcomes
        ),
        "average_input_tokens": _mean(
            outcome.input_tokens for outcome in outcomes
        ),
        "average_output_tokens": _mean(
            outcome.output_tokens for outcome in outcomes
        ),
        "average_total_tokens": _mean(
            outcome.total_tokens for outcome in outcomes
        ),
        "average_simulated_cost_usd": format(
            sum(costs, Decimal("0")) / Decimal(len(costs)),
            "f",
        ),
        "average_latency_ms": _mean(latencies),
        "p50_latency_ms": median(latencies),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "overall_quality_score": _mean(
            outcome.scores.overall_score for outcome in outcomes
        ),
        "by_category": _group(outcomes, "category"),
        "by_difficulty": _group(outcomes, "difficulty"),
        "by_safety": {
            "safety_sensitive": _group_summary(
                [item for item in outcomes if item.case.safety_sensitive]
            ),
            "ordinary": _group_summary(
                [item for item in outcomes if not item.case.safety_sensitive]
            ),
        },
        "by_mode": {
            mode.value: _group_summary(
                [item for item in outcomes if item.mode == mode]
            )
            for mode in FakeLLMMode
        },
    }
    return _round(summary)


def _group(
    outcomes: Sequence[PromptCaseOutcome],
    attribute: str,
) -> dict[str, dict[str, Any]]:
    keys = sorted({str(getattr(item.case, attribute)) for item in outcomes})
    return {
        key: _group_summary(
            [
                item
                for item in outcomes
                if str(getattr(item.case, attribute)) == key
            ]
        )
        for key in keys
    }


def _group_summary(
    outcomes: Sequence[PromptCaseOutcome],
) -> dict[str, Any]:
    if not outcomes:
        return {"case_mode_count": 0}
    return {
        "case_mode_count": len(outcomes),
        "overall_quality_score": _mean(
            item.scores.overall_score for item in outcomes
        ),
        "grounded_answer_rate": _rate(
            item.scores.groundedness for item in outcomes
        ),
        "answer_completeness_rate": _rate(
            item.scores.response_completeness for item in outcomes
        ),
        "safety_compliance_rate": _rate(
            item.scores.safety_compliance for item in outcomes
        ),
        "average_output_tokens": _mean(
            item.output_tokens for item in outcomes
        ),
        "average_latency_ms": _mean(
            item.latency_ms for item in outcomes
        ),
    }


def _select(metrics: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    strategy_ids = tuple(metrics)
    overall = max(
        strategy_ids,
        key=lambda key: (
            metrics[key]["overall_quality_score"],
            metrics[key]["answer_completeness_rate"],
            -metrics[key]["average_total_tokens"],
        ),
    )
    troubleshooting = max(
        strategy_ids,
        key=lambda key: (
            metrics[key]["by_category"]["troubleshooting"][
                "overall_quality_score"
            ],
            -metrics[key]["by_category"]["troubleshooting"][
                "average_output_tokens"
            ],
        ),
    )
    concise = max(
        strategy_ids,
        key=lambda key: (
            metrics[key]["by_mode"]["concise"][
                "overall_quality_score"
            ],
            -metrics[key]["by_mode"]["concise"][
                "average_output_tokens"
            ],
        ),
    )
    safety = max(
        strategy_ids,
        key=lambda key: (
            metrics[key]["by_safety"]["safety_sensitive"][
                "overall_quality_score"
            ],
            metrics[key]["safety_compliance_rate"],
        ),
    )
    return {
        "recommended_strategy": overall,
        "best_troubleshooting_strategy": troubleshooting,
        "best_concise_strategy": concise,
        "best_safety_sensitive_strategy": safety,
        "application_default_changed": False,
        "selection_rule": (
            "Highest deterministic overall quality score, then answer "
            "completeness, then fewer total tokens."
        ),
        "limitations": (
            "Selection compares scripted fake-provider adherence to prompt "
            "contracts, not real language-model quality."
        ),
    }


def _failures(
    outcomes: Sequence[PromptCaseOutcome],
) -> dict[str, list[str]]:
    checks = {
        "unsupported_answers": (
            lambda item: not item.scores.forbidden_claim_avoidance
        ),
        "missing_citations": (
            lambda item: not item.scores.citation_completeness
        ),
        "unnecessary_citations": (
            lambda item: item.scores.unnecessary_citations
        ),
        "overlong_responses": (
            lambda item: not item.scores.response_conciseness
        ),
        "incomplete_troubleshooting_steps": (
            lambda item: item.case.category == "troubleshooting"
            and not item.scores.response_completeness
        ),
        "safety_failures": (
            lambda item: not item.scores.safety_compliance
        ),
        "approval_gate_failures": (
            lambda item: not item.scores.approval_gate_compliance
        ),
        "uncertainty_failures": (
            lambda item: not item.scores.uncertainty_handling
        ),
        "formatting_failures": (
            lambda item: not item.scores.structural_correctness
        ),
    }
    return {
        name: sorted(
            {
                f"{item.case.case_id}:{item.mode.value}"
                for item in outcomes
                if check(item)
            }
        )
        for name, check in checks.items()
    }


def _contexts_for_case(
    case: PromptBenchmarkCase,
    document_map: Mapping[str, Any],
    *,
    client_id: str,
    environment: str,
) -> tuple[FixedPromptContext, ...]:
    return tuple(
        FixedPromptContext(
            source_id=f"S{index}",
            document_id=document_id,
            source_name=document_map[document_id].title,
            text=case.context_overrides.get(
                document_id,
                document_map[document_id].text,
            ),
            client_id=client_id,
            environment=environment,
        )
        for index, document_id in enumerate(
            case.retrieved_document_ids,
            start=1,
        )
    )


def _validate_document_ids(
    benchmark: PromptBenchmark,
    document_map: Mapping[str, Any],
) -> None:
    known = set(document_map)
    for case in benchmark.cases:
        missing = set(case.retrieved_document_ids) - known
        if missing:
            raise ValueError(
                f"Case {case.case_id} references missing documents: "
                + ", ".join(sorted(missing))
            )


def _validate_strategies(
    strategies: Sequence[PromptStrategy],
) -> None:
    if len(strategies) < 3:
        raise ValueError("At least three prompt strategies are required")
    identifiers = [
        strategy.definition.strategy_id for strategy in strategies
    ]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Prompt strategy IDs must be unique")


def _simulated_cost_estimator() -> CatalogCostEstimator:
    return CatalogCostEstimator(
        (
            ModelPricing(
                model_id=SIMULATED_MODEL_ID,
                currency="USD",
                input_price_per_million_tokens=Decimal("0.25"),
                output_price_per_million_tokens=Decimal("1.25"),
                pricing_source=(
                    "Synthetic offline comparison profile; no provider charge"
                ),
                pricing_effective_date="2026-07-27",
                pricing_version=SIMULATED_PRICING_VERSION,
                regions=(SIMULATED_REGION,),
            ),
        )
    )


def _measure_generation(
    operation: Callable[[], GenerationResult],
    *,
    repetitions: int,
    clock: Callable[[], float],
) -> tuple[GenerationResult, float]:
    durations: list[float] = []
    reference: GenerationResult | None = None
    for _ in range(repetitions):
        started = clock()
        result = operation()
        durations.append(max(0.0, (clock() - started) * 1_000.0))
        if reference is None:
            reference = result
        elif result != reference:
            raise ValueError("Fake LLM returned nondeterministic output")
    if reference is None:
        raise ValueError("No fake LLM result was generated")
    return reference, median(durations)


def _context_checksum(contexts: Sequence[FixedPromptContext]) -> str:
    digest = hashlib.sha256()
    for context in contexts:
        digest.update(context.source_id.encode())
        digest.update(b"\0")
        digest.update(context.document_id.encode())
        digest.update(b"\0")
        digest.update(context.text.encode())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _token_count(text: str) -> int:
    return len(_TOKEN.findall(text))


def _rate(values) -> float:
    items = tuple(bool(value) for value in values)
    return sum(items) / len(items)


def _mean(values) -> float:
    items = tuple(float(value) for value in values)
    return sum(items) / len(items)


def _conditional_rate(
    outcomes: Sequence[PromptCaseOutcome],
    *,
    predicate: Callable[[PromptCaseOutcome], bool],
    value: Callable[[PromptCaseOutcome], bool],
) -> float:
    relevant = [item for item in outcomes if predicate(item)]
    return _rate(value(item) for item in relevant) if relevant else 1.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    return values[max(0, math.ceil(percentile * len(values)) - 1)]


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    return value


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("evaluated_at must be a valid ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("evaluated_at must include a timezone")
