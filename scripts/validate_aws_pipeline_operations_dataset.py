"""Validate the static AWS pipeline operations dataset without network calls."""

from __future__ import annotations

import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any, Iterable

from scripts.generate_aws_pipeline_operations_dataset import (
    DATASET_ROOT,
    REPOSITORY_ROOT,
    SYNTHETIC_NOTICE,
    check_dataset,
    render_dataset_files,
)


MINIMUM_DOCUMENTS = 30
MINIMUM_RETRIEVAL_QUERIES = 150
MINIMUM_ANSWER_CASES = 75
MINIMUM_QUERIES_PER_DOCUMENT = 5
MINIMUM_DOCUMENT_WORDS = 500
MAXIMUM_DOCUMENT_WORDS = 1_500

METADATA_FIELDS = {
    "document_id",
    "filename",
    "title",
    "summary",
    "domain",
    "service",
    "document_type",
    "difficulty",
    "keywords",
    "synthetic",
    "license",
    "created_at",
    "expected_chunk_topics",
}
RETRIEVAL_FIELDS = {
    "query_id",
    "query",
    "relevant_document_ids",
    "primary_document_id",
    "expected_answer_summary",
    "answerable",
    "difficulty",
    "query_type",
    "services",
    "required_keywords",
}
ANSWER_FIELDS = {
    "case_id",
    "question",
    "expected_answer",
    "required_facts",
    "prohibited_claims",
    "supporting_document_ids",
    "citation_required",
    "answerable",
    "difficulty",
    "category",
}

PROHIBITED_PATTERNS = {
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "AWS account ID": re.compile(r"(?<!\d)\d{12}(?!\d)"),
    "JWT-like token": re.compile(
        r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\."
        r"[A-Za-z0-9_-]{20,}\b"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "assigned credential": re.compile(
        r"(?i)\b(?:api[_ -]?key|password|auth[_ -]?token|secret[_ -]?value)"
        r"\s*[:=]\s*[\"']?[A-Za-z0-9+/=_-]{12,}"
    ),
    "email address": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
    ),
}


class DatasetValidationError(ValueError):
    """Raised when one or more dataset invariants fail."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DatasetValidationError([f"cannot read {path}: {error}"]) from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetValidationError(
                [f"invalid JSON in {path}:{line_number}: {error.msg}"]
            ) from error
        if not isinstance(value, dict):
            raise DatasetValidationError(
                [f"expected JSON object in {path}:{line_number}"]
            )
        records.append(value)
    return records


def _normalized_question(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def find_near_duplicate_questions(
    records: Iterable[dict[str, Any]],
    *,
    similarity_threshold: float = 0.94,
) -> list[tuple[str, str]]:
    """Return IDs of exact or highly similar accidental question pairs."""

    prepared: list[tuple[str, str, set[str]]] = []
    for index, record in enumerate(records):
        identifier = str(
            record.get("query_id") or record.get("case_id") or f"record-{index}"
        )
        question = str(record.get("query") or record.get("question") or "")
        normalized = _normalized_question(question)
        prepared.append((identifier, normalized, set(normalized.split())))

    duplicates: list[tuple[str, str]] = []
    for index, (left_id, left_text, left_tokens) in enumerate(prepared):
        for right_id, right_text, right_tokens in prepared[index + 1 :]:
            if not left_text or not right_text:
                continue
            if left_text == right_text:
                duplicates.append((left_id, right_id))
                continue
            union = left_tokens | right_tokens
            jaccard = len(left_tokens & right_tokens) / len(union) if union else 1.0
            sequence = SequenceMatcher(None, left_text, right_text).ratio()
            if jaccard >= 0.86 and sequence >= similarity_threshold:
                duplicates.append((left_id, right_id))
    return duplicates


def _missing_fields(record: dict[str, Any], required: set[str]) -> set[str]:
    return required - set(record)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_metadata(
    root: Path, records: list[dict[str, Any]], errors: list[str]
) -> set[str]:
    if len(records) < MINIMUM_DOCUMENTS:
        errors.append(
            f"metadata has {len(records)} records; expected at least {MINIMUM_DOCUMENTS}"
        )
    document_ids: set[str] = set()
    filenames: set[str] = set()
    document_directory = root / DATASET_ROOT / "documents"
    for index, record in enumerate(records, start=1):
        missing = _missing_fields(record, METADATA_FIELDS)
        if missing:
            errors.append(f"metadata record {index} missing {sorted(missing)}")
            continue
        document_id = record["document_id"]
        filename = record["filename"]
        if not _is_nonempty_string(document_id):
            errors.append(f"metadata record {index} has empty document_id")
        elif document_id in document_ids:
            errors.append(f"duplicate document_id: {document_id}")
        else:
            document_ids.add(document_id)
        if not _is_nonempty_string(filename):
            errors.append(f"metadata record {index} has empty filename")
            continue
        if filename in filenames:
            errors.append(f"duplicate filename: {filename}")
        filenames.add(filename)
        if Path(filename).name != filename or not filename.endswith(".md"):
            errors.append(f"unsafe or unsupported filename: {filename}")
            continue
        path = document_directory / filename
        if not path.is_file():
            errors.append(f"missing document file: {filename}")
            continue
        content = path.read_text(encoding="utf-8")
        if SYNTHETIC_NOTICE not in content:
            errors.append(f"missing synthetic notice: {filename}")
        words = re.findall(r"\b[\w'-]+\b", content)
        if not MINIMUM_DOCUMENT_WORDS <= len(words) <= MAXIMUM_DOCUMENT_WORDS:
            errors.append(
                f"{filename} has {len(words)} words; expected "
                f"{MINIMUM_DOCUMENT_WORDS}-{MAXIMUM_DOCUMENT_WORDS}"
            )
        for field in (
            "title",
            "summary",
            "domain",
            "service",
            "document_type",
            "difficulty",
            "license",
            "created_at",
        ):
            if not _is_nonempty_string(record[field]):
                errors.append(f"metadata {document_id} has empty {field}")
        for field in ("keywords", "expected_chunk_topics"):
            if not isinstance(record[field], list) or not record[field]:
                errors.append(f"metadata {document_id} has empty {field}")
        if record["synthetic"] is not True:
            errors.append(f"metadata {document_id} must set synthetic=true")

    actual_files = {
        path.name for path in document_directory.glob("*.md") if path.is_file()
    }
    if actual_files != filenames:
        errors.append(
            "document directory and metadata filenames differ: "
            f"extra={sorted(actual_files - filenames)}, "
            f"missing={sorted(filenames - actual_files)}"
        )
    return document_ids


def _validate_retrieval(
    records: list[dict[str, Any]], document_ids: set[str], errors: list[str]
) -> set[str]:
    if len(records) < MINIMUM_RETRIEVAL_QUERIES:
        errors.append(
            f"retrieval set has {len(records)} records; expected at least "
            f"{MINIMUM_RETRIEVAL_QUERIES}"
        )
    query_ids: set[str] = set()
    answerable_counts: defaultdict[str, int] = defaultdict(int)
    for index, record in enumerate(records, start=1):
        missing = _missing_fields(record, RETRIEVAL_FIELDS)
        if missing:
            errors.append(f"retrieval record {index} missing {sorted(missing)}")
            continue
        query_id = record["query_id"]
        if not _is_nonempty_string(query_id):
            errors.append(f"retrieval record {index} has empty query_id")
        elif query_id in query_ids:
            errors.append(f"duplicate query_id: {query_id}")
        else:
            query_ids.add(query_id)
        if not _is_nonempty_string(record["query"]):
            errors.append(f"retrieval {query_id} has empty query")
        if not _is_nonempty_string(record["expected_answer_summary"]):
            errors.append(f"retrieval {query_id} has empty expected answer")
        relevant = record["relevant_document_ids"]
        if not isinstance(relevant, list):
            errors.append(f"retrieval {query_id} relevant_document_ids is not a list")
            continue
        unknown = set(relevant) - document_ids
        if unknown:
            errors.append(f"retrieval {query_id} references unknown documents {sorted(unknown)}")
        forbidden = record.get("forbidden_document_ids", [])
        if not isinstance(forbidden, list):
            errors.append(f"retrieval {query_id} forbidden_document_ids is not a list")
        elif set(forbidden) - document_ids:
            errors.append(f"retrieval {query_id} has unknown forbidden document")
        if set(relevant) & set(forbidden):
            errors.append(f"retrieval {query_id} marks a document relevant and forbidden")
        if record["answerable"] is True:
            if not relevant:
                errors.append(f"answerable retrieval {query_id} has no support")
            if record["primary_document_id"] not in relevant:
                errors.append(f"retrieval {query_id} primary document is not relevant")
            for document_id in relevant:
                answerable_counts[document_id] += 1
        elif record["answerable"] is False:
            if relevant or record["primary_document_id"] is not None:
                errors.append(f"unanswerable retrieval {query_id} contains support labels")
            summary = str(record["expected_answer_summary"]).lower()
            if not any(
                marker in summary
                for marker in ("cannot be answered", "does not contain", "no real")
            ):
                errors.append(
                    f"unanswerable retrieval {query_id} asserts unsupported expected claims"
                )
            if record["required_keywords"]:
                errors.append(
                    f"unanswerable retrieval {query_id} must not require answer keywords"
                )
        else:
            errors.append(f"retrieval {query_id} answerable must be boolean")

    for document_id in document_ids:
        total = sum(
            1
            for record in records
            if record.get("primary_document_id") == document_id
            or document_id in record.get("forbidden_document_ids", [])
        )
        if total < MINIMUM_QUERIES_PER_DOCUMENT:
            errors.append(
                f"{document_id} has {total} retrieval queries; expected at least "
                f"{MINIMUM_QUERIES_PER_DOCUMENT}"
            )
        if answerable_counts[document_id] == 0:
            errors.append(f"{document_id} has no answerable retrieval query")

    duplicates = find_near_duplicate_questions(records)
    if duplicates:
        errors.append(f"duplicate or near-duplicate retrieval questions: {duplicates}")
    return query_ids


def _validate_answers(
    records: list[dict[str, Any]], document_ids: set[str], errors: list[str]
) -> set[str]:
    if len(records) < MINIMUM_ANSWER_CASES:
        errors.append(
            f"answer set has {len(records)} records; expected at least "
            f"{MINIMUM_ANSWER_CASES}"
        )
    case_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        missing = _missing_fields(record, ANSWER_FIELDS)
        if missing:
            errors.append(f"answer record {index} missing {sorted(missing)}")
            continue
        case_id = record["case_id"]
        if not _is_nonempty_string(case_id):
            errors.append(f"answer record {index} has empty case_id")
        elif case_id in case_ids:
            errors.append(f"duplicate case_id: {case_id}")
        else:
            case_ids.add(case_id)
        for field in ("question", "expected_answer"):
            if not _is_nonempty_string(record[field]):
                errors.append(f"answer {case_id} has empty {field}")
        supporting = record["supporting_document_ids"]
        if not isinstance(supporting, list):
            errors.append(f"answer {case_id} supporting_document_ids is not a list")
            continue
        unknown = set(supporting) - document_ids
        if unknown:
            errors.append(f"answer {case_id} references unknown documents {sorted(unknown)}")
        if record["answerable"] is True and not supporting:
            errors.append(f"answerable case {case_id} has no supporting document")
        if record["answerable"] is False and supporting:
            errors.append(f"unanswerable case {case_id} has supporting documents")
        for field in ("required_facts", "prohibited_claims"):
            if not isinstance(record[field], list) or not record[field]:
                errors.append(f"answer {case_id} has empty {field}")
    duplicates = find_near_duplicate_questions(records)
    if duplicates:
        errors.append(f"duplicate or near-duplicate answer questions: {duplicates}")
    return case_ids


def _validate_splits(
    root: Path,
    document_ids: set[str],
    query_ids: set[str],
    case_ids: set[str],
    queries: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    errors: list[str],
) -> None:
    seen: dict[str, set[str]] = {
        "document_ids": set(),
        "retrieval_query_ids": set(),
        "answer_case_ids": set(),
    }
    scenario_splits: defaultdict[str, set[str]] = defaultdict(set)
    query_by_id = {record["query_id"]: record for record in queries}
    answer_by_id = {record["case_id"]: record for record in answers}
    for name in ("train", "validation", "test"):
        path = root / DATASET_ROOT / "splits" / f"{name}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid split file {path}: {error}")
            continue
        if not isinstance(payload, dict) or payload.get("split") != name:
            errors.append(f"split file {name} has invalid split label")
            continue
        for field, allowed in (
            ("document_ids", document_ids),
            ("retrieval_query_ids", query_ids),
            ("answer_case_ids", case_ids),
        ):
            values = payload.get(field)
            if not isinstance(values, list) or len(values) != len(set(values)):
                errors.append(f"split {name} has invalid or duplicate {field}")
                continue
            overlap = seen[field] & set(values)
            if overlap:
                errors.append(f"split overlap in {field}: {sorted(overlap)}")
            unknown = set(values) - allowed
            if unknown:
                errors.append(f"split {name} has unknown {field}: {sorted(unknown)}")
            seen[field].update(values)
        counts = payload.get("counts", {})
        expected_counts = {
            "documents": len(payload.get("document_ids", [])),
            "retrieval_queries": len(payload.get("retrieval_query_ids", [])),
            "answer_cases": len(payload.get("answer_case_ids", [])),
        }
        if counts != expected_counts:
            errors.append(f"split {name} count summary is inaccurate")
        for query_id in payload.get("retrieval_query_ids", []):
            record = query_by_id.get(query_id)
            if record and record.get("scenario_id"):
                scenario_splits[str(record["scenario_id"])].add(name)
        for case_id in payload.get("answer_case_ids", []):
            record = answer_by_id.get(case_id)
            if record and record.get("scenario_id"):
                scenario_splits[str(record["scenario_id"])].add(name)

    expected_sets = {
        "document_ids": document_ids,
        "retrieval_query_ids": query_ids,
        "answer_case_ids": case_ids,
    }
    for field, expected in expected_sets.items():
        if seen[field] != expected:
            errors.append(
                f"splits do not cover all {field}: missing={sorted(expected - seen[field])}"
            )
    leaking = {
        scenario: sorted(splits)
        for scenario, splits in scenario_splits.items()
        if len(splits) > 1
    }
    if leaking:
        errors.append(f"scenario leakage across splits: {leaking}")


def _validate_prohibited_content(root: Path, errors: list[str]) -> None:
    for relative_path in render_dataset_files():
        content = (root / relative_path).read_text(encoding="utf-8")
        for label, pattern in PROHIBITED_PATTERNS.items():
            if pattern.search(content):
                errors.append(
                    f"prohibited {label} pattern in {relative_path.as_posix()}"
                )


def validate_dataset(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Validate the dataset and return exact record counts."""

    root = repository_root.resolve()
    errors = check_dataset(root)
    first_render = render_dataset_files()
    if first_render != render_dataset_files():
        errors.append("generator returned non-deterministic content in one process")
    dataset = root / DATASET_ROOT
    metadata = _load_jsonl(dataset / "metadata/documents.jsonl")
    queries = _load_jsonl(dataset / "evaluation/retrieval_queries.jsonl")
    answers = _load_jsonl(dataset / "evaluation/answer_evaluation.jsonl")
    document_ids = _validate_metadata(root, metadata, errors)
    query_ids = _validate_retrieval(queries, document_ids, errors)
    case_ids = _validate_answers(answers, document_ids, errors)
    _validate_splits(
        root,
        document_ids,
        query_ids,
        case_ids,
        queries,
        answers,
        errors,
    )
    _validate_prohibited_content(root, errors)
    if errors:
        raise DatasetValidationError(errors)

    split_counts: dict[str, dict[str, int]] = {}
    for name in ("train", "validation", "test"):
        payload = json.loads(
            (dataset / "splits" / f"{name}.json").read_text(encoding="utf-8")
        )
        split_counts[name] = payload["counts"]
    return {
        "documents": len(document_ids),
        "metadata_records": len(metadata),
        "retrieval_queries": len(query_ids),
        "answer_cases": len(case_ids),
        "splits": split_counts,
        "generated_files": len(first_render),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root", type=Path, default=REPOSITORY_ROOT
    )
    arguments = parser.parse_args()
    try:
        summary = validate_dataset(arguments.repository_root)
    except DatasetValidationError as error:
        for item in error.errors:
            print(f"ERROR: {item}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("AWS pipeline operations dataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
