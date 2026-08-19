"""Deterministic JSON, Markdown, and CSV retrieval comparison reports."""

import csv
import io
import json
from pathlib import Path
from typing import Any

from evaluation.comparison import RetrievalComparison, STRATEGIES


JSON_FILENAME = "retrieval_comparison.json"
MARKDOWN_FILENAME = "retrieval_comparison.md"
CSV_FILENAME = "retrieval_query_results.csv"


def write_reports(
    comparison: RetrievalComparison,
    output_directory: Path | str,
) -> tuple[Path, Path, Path]:
    """Write all reviewer and machine-readable output artifacts."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / JSON_FILENAME
    markdown_path = destination / MARKDOWN_FILENAME
    csv_path = destination / CSV_FILENAME
    json_path.write_text(
        json.dumps(
            comparison.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown(comparison),
        encoding="utf-8",
    )
    csv_path.write_text(
        render_csv(comparison),
        encoding="utf-8",
        newline="",
    )
    return json_path, markdown_path, csv_path


def render_markdown(comparison: RetrievalComparison) -> str:
    """Create the reviewed, human-readable comparison report."""

    metadata = comparison.metadata
    settings = comparison.settings
    lines = [
        "# Retrieval comparison results",
        "",
        "This snapshot was generated offline from a repository-owned "
        f"synthetic corpus (`{metadata['benchmark_license']}`).",
        "",
        "## Provenance",
        "",
        f"- Evaluation date: `{metadata['evaluation_date']}`",
        f"- Application version: `{metadata['application_version']}`",
        f"- Benchmark version: `{metadata['benchmark_version']}`",
        f"- Corpus version: `{metadata['corpus_version']}`",
        f"- Corpus checksum: `{metadata['corpus_checksum']}`",
        f"- Corpus documents/chunks: "
        f"`{metadata['corpus_document_count']}` / "
        f"`{metadata['corpus_chunk_count']}`",
        f"- Embedding provider: `{metadata['embedding_provider']}`",
        f"- Embedding model: `{metadata['embedding_model_id']}`",
        f"- Python version: `{metadata['python_version']}`",
        f"- Scope: `{settings['scope']['client_id']}` / "
        f"`{settings['scope']['environment']}`",
        "",
        "## Overall metrics",
        "",
        "| Strategy | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | Hit rate | Exact doc | No result | Avg ms | P50 ms | P95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in STRATEGIES:
        metric = comparison.metrics[strategy]
        lines.append(
            f"| {strategy} "
            f"| {_metric(metric['precision_at_1'])} "
            f"| {_metric(metric['precision_at_3'])} "
            f"| {_metric(metric['precision_at_5'])} "
            f"| {_metric(metric['recall_at_1'])} "
            f"| {_metric(metric['recall_at_3'])} "
            f"| {_metric(metric['recall_at_5'])} "
            f"| {_metric(metric['mrr'])} "
            f"| {_metric(metric['hit_rate'])} "
            f"| {_metric(metric['exact_document_success'])} "
            f"| {_metric(metric['no_result_rate'])} "
            f"| {_latency(metric['average_latency_ms'])} "
            f"| {_latency(metric['p50_latency_ms'])} "
            f"| {_latency(metric['p95_latency_ms'])} |"
        )

    selection = comparison.selection
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Recommended default: "
            f"**{selection['recommended_default']}**",
            f"- Best exact-keyword strategy: "
            f"**{selection['best_exact_keyword']}**",
            f"- Best paraphrase strategy: "
            f"**{selection['best_paraphrase']}**",
            f"- Formula: {selection['selection_formula']}",
            f"- Tie policy: {selection['tie_policy']}",
            "",
            "This recommendation does not silently change the Streamlit "
            "application default. The existing semantic path remains in place "
            "until a separate backward-compatible integration decision.",
            "",
            "## Performance by match type",
            "",
            "| Strategy | Match type | Cases | Hit rate | MRR | Recall@5 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy in STRATEGIES:
        groups = comparison.metrics[strategy]["by_match_type"]
        for match_type, group in groups.items():
            lines.append(
                f"| {strategy} | {match_type} "
                f"| {group['case_count']} "
                f"| {_metric(group['hit_rate'])} "
                f"| {_metric(group['mrr'])} "
                f"| {_metric(group['recall_at_5'])} |"
            )

    lines.extend(
        [
            "",
            "## Performance by category",
            "",
            "| Strategy | Category | Cases | Hit rate | MRR | Recall@5 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy in STRATEGIES:
        groups = comparison.metrics[strategy]["by_category"]
        for category, group in groups.items():
            lines.append(
                f"| {strategy} | {category} "
                f"| {group['case_count']} "
                f"| {_metric(group['hit_rate'])} "
                f"| {_metric(group['mrr'])} "
                f"| {_metric(group['recall_at_5'])} |"
            )

    lines.extend(
        [
            "",
            "## Performance by difficulty",
            "",
            "| Strategy | Difficulty | Cases | Hit rate | MRR | Recall@5 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy in STRATEGIES:
        groups = comparison.metrics[strategy]["by_difficulty"]
        for difficulty, group in groups.items():
            lines.append(
                f"| {strategy} | {difficulty} "
                f"| {group['case_count']} "
                f"| {_metric(group['hit_rate'])} "
                f"| {_metric(group['mrr'])} "
                f"| {_metric(group['recall_at_5'])} |"
            )

    lines.extend(
        [
            "",
            "## Failure summary",
            "",
        ]
    )
    for strategy in STRATEGIES:
        failures = comparison.failure_analysis[strategy]
        missed = ", ".join(failures["missed_at_5"]) or "none"
        no_results = ", ".join(failures["no_results"]) or "none"
        lines.extend(
            [
                f"### {strategy}",
                "",
                f"- Missed expected targets at k=5: {missed}",
                f"- Returned no results: {no_results}",
                "",
            ]
        )

    lines.extend(
        [
            "## Settings",
            "",
            "```json",
            json.dumps(settings, indent=2, sort_keys=True),
            "```",
            "",
            "Latency is local wall-clock time and is environment-dependent. "
            "Each reported query latency is the median of the configured "
            "repetitions. Ranking and quality metrics are deterministic for "
            "the versioned corpus, benchmark, and settings.",
            "",
        ]
    )
    return "\n".join(lines)


def render_csv(comparison: RetrievalComparison) -> str:
    """Create one flat row per strategy/query pair."""

    output = io.StringIO(newline="")
    fieldnames = [
        "strategy",
        "query_id",
        "query",
        "category",
        "difficulty",
        "match_type",
        "expected_document_ids",
        "returned_document_ids",
        "returned_chunk_ids",
        "precision_at_1",
        "precision_at_3",
        "precision_at_5",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "reciprocal_rank",
        "hit",
        "exact_document_success",
        "latency_ms",
    ]
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    for outcome in comparison.outcomes:
        writer.writerow(
            {
                "strategy": outcome.strategy,
                "query_id": outcome.case.query_id,
                "query": outcome.case.query,
                "category": outcome.case.category,
                "difficulty": outcome.case.difficulty,
                "match_type": outcome.case.match_type,
                "expected_document_ids": ";".join(
                    outcome.case.expected_document_ids
                ),
                "returned_document_ids": ";".join(
                    result.document_id for result in outcome.results
                ),
                "returned_chunk_ids": ";".join(
                    result.chunk_id for result in outcome.results
                ),
                "precision_at_1": outcome.precision_at[1],
                "precision_at_3": outcome.precision_at[3],
                "precision_at_5": outcome.precision_at[5],
                "recall_at_1": outcome.recall_at[1],
                "recall_at_3": outcome.recall_at[3],
                "recall_at_5": outcome.recall_at[5],
                "reciprocal_rank": outcome.reciprocal_rank,
                "hit": outcome.hit,
                "exact_document_success": (
                    outcome.exact_document_success
                ),
                "latency_ms": round(outcome.latency_ms, 6),
            }
        )
    return output.getvalue()


def _metric(value: Any) -> str:
    return f"{float(value):.4f}"


def _latency(value: Any) -> str:
    return f"{float(value):.3f}"
