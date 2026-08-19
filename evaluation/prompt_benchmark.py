"""Load and validate the synthetic LLM and prompt benchmark."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


CURRENT_SCHEMA_VERSION = 1
MINIMUM_CASE_COUNT = 30
REQUIRED_CATEGORIES = frozenset(
    {
        "factual_lookup",
        "troubleshooting",
        "architecture_explanation",
        "sql_guidance",
        "pyspark_guidance",
        "iam_least_privilege",
        "monitoring",
        "cost_awareness",
        "ambiguous_question",
        "insufficient_evidence",
        "destructive_request",
        "approval_required",
        "prompt_injection",
        "unsupported_claim",
        "conflicting_context",
    }
)
VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
_CASE_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,63}")
_SOURCE_ID = re.compile(r"S[1-9][0-9]*")


@dataclass(frozen=True)
class PromptBenchmarkCase:
    """One fixed-context prompt evaluation case."""

    case_id: str
    user_question: str
    category: str
    difficulty: str
    retrieved_document_ids: tuple[str, ...]
    expected_answer_criteria: tuple[str, ...]
    required_source_ids: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    uncertainty_required: bool
    approval_required: bool
    refusal_or_safety_required: bool
    safety_sensitive: bool
    context_overrides: dict[str, str]
    notes: str


@dataclass(frozen=True)
class PromptBenchmark:
    """Validated benchmark metadata and cases."""

    benchmark_version: str
    corpus_version: str
    license: str
    client_id: str
    environment: str
    cases: tuple[PromptBenchmarkCase, ...]


def load_prompt_benchmark(path: Path | str) -> PromptBenchmark:
    """Read and validate one prompt benchmark JSON file."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Unable to load prompt benchmark: {source}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Prompt benchmark root must be an object")
    if payload.get("schema_version") != CURRENT_SCHEMA_VERSION:
        raise ValueError("Unsupported prompt benchmark schema_version")
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("Prompt benchmark scope must be an object")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) < MINIMUM_CASE_COUNT:
        raise ValueError(
            f"Prompt benchmark must contain at least {MINIMUM_CASE_COUNT} cases"
        )

    cases: list[PromptBenchmarkCase] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"Prompt case {index} must be an object")
        case_id = _text(raw, "case_id")
        if not _CASE_ID.fullmatch(case_id):
            raise ValueError(f"Invalid case_id: {case_id}")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        category = _text(raw, "category")
        difficulty = _text(raw, "difficulty")
        if category not in REQUIRED_CATEGORIES:
            raise ValueError(f"Unsupported category: {category}")
        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(f"Unsupported difficulty: {difficulty}")
        document_ids = _string_list(
            raw,
            "retrieved_document_ids",
            required=False,
        )
        source_ids = _string_list(
            raw,
            "required_source_ids",
            required=False,
        )
        available_sources = {
            f"S{position}"
            for position in range(1, len(document_ids) + 1)
        }
        if any(
            not _SOURCE_ID.fullmatch(source_id)
            or source_id not in available_sources
            for source_id in source_ids
        ):
            raise ValueError(
                f"Case {case_id} has an invalid required source ID"
            )
        overrides = raw.get("context_overrides", {})
        if not isinstance(overrides, dict) or any(
            not isinstance(key, str)
            or key not in document_ids
            or not isinstance(value, str)
            or not value.strip()
            for key, value in overrides.items()
        ):
            raise ValueError(
                f"Case {case_id} has invalid context_overrides"
            )
        cases.append(
            PromptBenchmarkCase(
                case_id=case_id,
                user_question=_text(raw, "user_question"),
                category=category,
                difficulty=difficulty,
                retrieved_document_ids=document_ids,
                expected_answer_criteria=_string_list(
                    raw,
                    "expected_answer_criteria",
                    required=True,
                ),
                required_source_ids=source_ids,
                forbidden_claims=_string_list(
                    raw,
                    "forbidden_claims",
                    required=False,
                ),
                uncertainty_required=_boolean(
                    raw,
                    "uncertainty_required",
                ),
                approval_required=_boolean(raw, "approval_required"),
                refusal_or_safety_required=_boolean(
                    raw,
                    "refusal_or_safety_required",
                ),
                safety_sensitive=_boolean(raw, "safety_sensitive"),
                context_overrides={
                    key: value.strip()
                    for key, value in overrides.items()
                },
                notes=_text(raw, "notes"),
            )
        )

    missing = REQUIRED_CATEGORIES - {case.category for case in cases}
    if missing:
        raise ValueError(
            "Prompt benchmark is missing categories: "
            + ", ".join(sorted(missing))
        )
    return PromptBenchmark(
        benchmark_version=_text(payload, "benchmark_version"),
        corpus_version=_text(payload, "corpus_version"),
        license=_text(payload, "license"),
        client_id=_text(scope, "client_id"),
        environment=_text(scope, "environment"),
        cases=tuple(cases),
    )


def _text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _boolean(payload: dict[str, Any], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _string_list(
    payload: dict[str, Any],
    name: str,
    *,
    required: bool,
) -> tuple[str, ...]:
    value = payload.get(name, [])
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    parsed = tuple(
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    )
    if len(parsed) != len(value) or len(set(parsed)) != len(parsed):
        raise ValueError(
            f"{name} must contain unique non-empty strings"
        )
    if required and not parsed:
        raise ValueError(f"{name} must contain at least one value")
    return parsed
