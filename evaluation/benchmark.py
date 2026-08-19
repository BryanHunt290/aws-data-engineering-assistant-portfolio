"""Load and validate the synthetic retrieval benchmark."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


CURRENT_SCHEMA_VERSION = 1
MINIMUM_CASE_COUNT = 25
REQUIRED_CATEGORIES = frozenset(
    {
        "s3_architecture",
        "glue_troubleshooting",
        "athena_troubleshooting",
        "iam_least_privilege",
        "pyspark_transformations",
        "monitoring",
        "cost_awareness",
    }
)
VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
REQUIRED_MATCH_TYPES = frozenset(
    {"exact_keyword", "paraphrase", "ambiguous"}
)
_QUERY_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{2,63}")


@dataclass(frozen=True)
class BenchmarkCase:
    """One labeled, synthetic retrieval query."""

    query_id: str
    query: str
    expected_document_ids: tuple[str, ...]
    expected_chunk_ids: tuple[str, ...]
    category: str
    difficulty: str
    match_type: str
    notes: str


@dataclass(frozen=True)
class RetrievalBenchmark:
    """Validated benchmark metadata and cases."""

    benchmark_version: str
    corpus_version: str
    license: str
    client_id: str
    environment: str
    cases: tuple[BenchmarkCase, ...]


def load_benchmark(path: Path | str) -> RetrievalBenchmark:
    """Read and validate one benchmark JSON document."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load benchmark: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError("Benchmark root must be a JSON object")
    if payload.get("schema_version") != CURRENT_SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark schema_version")

    benchmark_version = _required_text(
        payload,
        "benchmark_version",
    )
    corpus_version = _required_text(payload, "corpus_version")
    license_name = _required_text(payload, "license")
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("Benchmark scope must be an object")
    client_id = _required_text(scope, "client_id")
    environment = _required_text(scope, "environment")

    raw_cases = payload.get("queries")
    if not isinstance(raw_cases, list) or len(raw_cases) < MINIMUM_CASE_COUNT:
        raise ValueError(
            f"Benchmark must contain at least {MINIMUM_CASE_COUNT} queries"
        )

    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Benchmark query {index} must be an object")
        query_id = _required_text(raw_case, "query_id")
        if not _QUERY_ID_PATTERN.fullmatch(query_id):
            raise ValueError(f"Invalid query_id: {query_id}")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query_id: {query_id}")
        seen_ids.add(query_id)

        category = _required_text(raw_case, "category")
        difficulty = _required_text(raw_case, "difficulty")
        match_type = _required_text(raw_case, "match_type")
        if category not in REQUIRED_CATEGORIES:
            raise ValueError(f"Unsupported category: {category}")
        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(f"Unsupported difficulty: {difficulty}")
        if match_type not in REQUIRED_MATCH_TYPES:
            raise ValueError(f"Unsupported match_type: {match_type}")

        expected_documents = _string_list(
            raw_case,
            "expected_document_ids",
            required=True,
        )
        expected_chunks = _string_list(
            raw_case,
            "expected_chunk_ids",
            required=False,
        )
        cases.append(
            BenchmarkCase(
                query_id=query_id,
                query=_required_text(raw_case, "query"),
                expected_document_ids=expected_documents,
                expected_chunk_ids=expected_chunks,
                category=category,
                difficulty=difficulty,
                match_type=match_type,
                notes=_required_text(raw_case, "notes"),
            )
        )

    present_categories = {case.category for case in cases}
    missing_categories = REQUIRED_CATEGORIES - present_categories
    if missing_categories:
        raise ValueError(
            "Benchmark is missing categories: "
            + ", ".join(sorted(missing_categories))
        )
    present_match_types = {case.match_type for case in cases}
    missing_match_types = REQUIRED_MATCH_TYPES - present_match_types
    if missing_match_types:
        raise ValueError(
            "Benchmark is missing match types: "
            + ", ".join(sorted(missing_match_types))
        )

    return RetrievalBenchmark(
        benchmark_version=benchmark_version,
        corpus_version=corpus_version,
        license=license_name,
        client_id=client_id,
        environment=environment,
        cases=tuple(cases),
    )


def _required_text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


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
    if len(parsed) != len(value):
        raise ValueError(f"{name} must contain only non-empty strings")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} cannot contain duplicates")
    if required and not parsed:
        raise ValueError(f"{name} must contain at least one value")
    return parsed
