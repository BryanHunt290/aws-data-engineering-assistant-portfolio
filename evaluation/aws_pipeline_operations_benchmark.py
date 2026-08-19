"""Adapt the AWS pipeline operations dataset to existing offline evaluators."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from evaluation.benchmark import BenchmarkCase, RetrievalBenchmark
from evaluation.prompt_benchmark import PromptBenchmark, PromptBenchmarkCase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "aws_pipeline_operations"
VALID_SPLITS = frozenset({"train", "validation", "test"})
EVALUATION_CLIENT_ID = "aws-pipeline-operations-evaluation"
EVALUATION_ENVIRONMENT = "test"


@dataclass(frozen=True)
class AWSPipelineOperationsBenchmarks:
    """Evaluation-ready cases plus explicit coverage accounting."""

    split: str
    dataset_version: str
    corpus_directory: Path
    document_ids: tuple[str, ...]
    retrieval: RetrievalBenchmark
    answers: PromptBenchmark
    unanswerable_query_ids: tuple[str, ...]

    @property
    def retrieval_query_count(self) -> int:
        return len(self.retrieval.cases) + len(self.unanswerable_query_ids)


def load_aws_pipeline_operations_benchmarks(
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    *,
    split: str = "test",
) -> AWSPipelineOperationsBenchmarks:
    """Load one leakage-safe split and adapt it without changing source data."""

    if split not in VALID_SPLITS:
        raise ValueError(
            "split must be one of: " + ", ".join(sorted(VALID_SPLITS))
        )
    root = Path(dataset_root)
    split_payload = _load_json(root / "splits" / f"{split}.json")
    dataset_version = _required_text(split_payload, "dataset_version")
    document_ids = _unique_strings(split_payload, "document_ids")
    query_ids = _unique_strings(split_payload, "retrieval_query_ids")
    answer_case_ids = _unique_strings(split_payload, "answer_case_ids")

    documents = _index_jsonl(
        root / "metadata" / "documents.jsonl",
        key="document_id",
    )
    queries = _index_jsonl(
        root / "evaluation" / "retrieval_queries.jsonl",
        key="query_id",
    )
    answers = _index_jsonl(
        root / "evaluation" / "answer_evaluation.jsonl",
        key="case_id",
    )
    _require_ids(documents, document_ids, "documents")
    _require_ids(queries, query_ids, "retrieval queries")
    _require_ids(answers, answer_case_ids, "answer cases")

    document_id_set = set(document_ids)
    retrieval_cases: list[BenchmarkCase] = []
    unanswerable: list[str] = []
    for query_id in query_ids:
        raw = queries[query_id]
        if raw.get("answerable") is False:
            unanswerable.append(query_id)
            continue
        relevant_ids = _unique_strings(raw, "relevant_document_ids")
        if not relevant_ids or not set(relevant_ids).issubset(document_id_set):
            raise ValueError(
                f"Retrieval query {query_id} references documents outside "
                f"the {split} split"
            )
        primary_document_id = _required_text(raw, "primary_document_id")
        metadata = documents[primary_document_id]
        query_type = _required_text(raw, "query_type")
        retrieval_cases.append(
            BenchmarkCase(
                query_id=query_id,
                query=_required_text(raw, "query"),
                expected_document_ids=relevant_ids,
                expected_chunk_ids=(),
                category=_required_text(metadata, "domain"),
                difficulty=_required_text(raw, "difficulty"),
                match_type=_match_type(query_type),
                notes=_required_text(raw, "expected_answer_summary"),
            )
        )

    prompt_cases: list[PromptBenchmarkCase] = []
    for case_id in answer_case_ids:
        raw = answers[case_id]
        if raw.get("answerable") is not True:
            raise ValueError(f"Answer case {case_id} must be answerable")
        supporting_ids = _unique_strings(raw, "supporting_document_ids")
        if not supporting_ids or not set(supporting_ids).issubset(
            document_id_set
        ):
            raise ValueError(
                f"Answer case {case_id} references documents outside "
                f"the {split} split"
            )
        category = _required_text(raw, "category")
        prompt_cases.append(
            PromptBenchmarkCase(
                case_id=case_id,
                user_question=_required_text(raw, "question"),
                category=category,
                difficulty=_required_text(raw, "difficulty"),
                retrieved_document_ids=supporting_ids,
                expected_answer_criteria=_unique_strings(
                    raw,
                    "required_facts",
                ),
                required_source_ids=tuple(
                    f"S{index}"
                    for index in range(1, len(supporting_ids) + 1)
                )
                if raw.get("citation_required") is True
                else (),
                forbidden_claims=_unique_strings(
                    raw,
                    "prohibited_claims",
                ),
                uncertainty_required=False,
                approval_required=category == "deployment_safety",
                refusal_or_safety_required=False,
                safety_sensitive=category
                in {
                    "deployment_safety",
                    "multi_client_isolation",
                    "security",
                    "vector_operations",
                },
                context_overrides={},
                notes=_required_text(raw, "expected_answer"),
            )
        )

    if not retrieval_cases:
        raise ValueError(f"The {split} split has no answerable retrieval cases")
    if not prompt_cases:
        raise ValueError(f"The {split} split has no answer cases")
    return AWSPipelineOperationsBenchmarks(
        split=split,
        dataset_version=dataset_version,
        corpus_directory=root / "documents",
        document_ids=document_ids,
        retrieval=RetrievalBenchmark(
            benchmark_version=(
                f"aws-pipeline-operations-retrieval-{split}-v1"
            ),
            corpus_version=f"aws-pipeline-operations-{dataset_version}",
            license="CC-BY-4.0",
            client_id=EVALUATION_CLIENT_ID,
            environment=EVALUATION_ENVIRONMENT,
            cases=tuple(retrieval_cases),
        ),
        answers=PromptBenchmark(
            benchmark_version=f"aws-pipeline-operations-answer-{split}-v1",
            corpus_version=f"aws-pipeline-operations-{dataset_version}",
            license="CC-BY-4.0",
            client_id=EVALUATION_CLIENT_ID,
            environment=EVALUATION_ENVIRONMENT,
            cases=tuple(prompt_cases),
        ),
        unanswerable_query_ids=tuple(unanswerable),
    )


def _match_type(query_type: str) -> str:
    if query_type == "direct_fact":
        return "exact_keyword"
    if query_type == "paraphrase":
        return "paraphrase"
    if query_type in {
        "ambiguous_terminology",
        "multi_step",
        "troubleshooting",
    }:
        return "ambiguous"
    raise ValueError(f"Unsupported answerable query_type: {query_type}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _index_jsonl(path: Path, *, key: str) -> dict[str, dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Unable to load JSONL file: {path}") from error
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSONL at {path}:{line_number}"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError(f"JSONL record must be an object: {path}")
        identifier = _required_text(raw, key)
        if identifier in indexed:
            raise ValueError(f"Duplicate {key}: {identifier}")
        indexed[identifier] = raw
    return indexed


def _require_ids(
    records: dict[str, dict[str, Any]],
    identifiers: tuple[str, ...],
    label: str,
) -> None:
    missing = set(identifiers) - set(records)
    if missing:
        raise ValueError(
            f"Split references missing {label}: " + ", ".join(sorted(missing))
        )


def _required_text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _unique_strings(
    payload: dict[str, Any],
    name: str,
) -> tuple[str, ...]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    parsed = tuple(
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    )
    if len(parsed) != len(value) or len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must contain unique non-empty strings")
    return parsed
