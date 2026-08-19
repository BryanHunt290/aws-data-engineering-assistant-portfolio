"""JSON, Markdown, and CSV reports for prompt comparison evidence."""

import csv
from decimal import Decimal
import io
import json
from pathlib import Path

from evaluation.prompt_comparison import PromptComparison


JSON_FILENAME = "llm_prompt_comparison.json"
MARKDOWN_FILENAME = "llm_prompt_comparison.md"
CSV_FILENAME = "llm_prompt_case_results.csv"


def write_prompt_reports(
    comparison: PromptComparison,
    output_directory: Path | str,
) -> tuple[Path, Path, Path]:
    """Write all prompt comparison artifacts."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths = (
        destination / JSON_FILENAME,
        destination / MARKDOWN_FILENAME,
        destination / CSV_FILENAME,
    )
    paths[0].write_text(
        json.dumps(comparison.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths[1].write_text(
        render_prompt_markdown(comparison),
        encoding="utf-8",
    )
    paths[2].write_text(
        render_prompt_csv(comparison),
        encoding="utf-8",
        newline="",
    )
    return paths


def render_prompt_markdown(comparison: PromptComparison) -> str:
    """Render the reviewer-facing comparison summary."""

    metadata = comparison.metadata
    lines = [
        "# Offline LLM and prompt comparison",
        "",
        "This report evaluates deterministic fake-provider adherence to prompt "
        "contracts. It does not measure real LLM quality and incurred no "
        "provider charge.",
        "",
        "## Provenance",
        "",
        f"- Evaluation date: `{metadata['evaluation_date']}`",
        f"- Git commit at generation: `{metadata['git_commit']}`",
        f"- Benchmark: `{metadata['benchmark_version']}` "
        f"({metadata['benchmark_case_count']} cases)",
        f"- Corpus: `{metadata['corpus_version']}`",
        f"- Retrieval: `{metadata['retrieval_strategy']}`",
        f"- Fake modes: `{', '.join(metadata['fake_llm_modes'])}`",
        "- Prompt strategies: "
        + ", ".join(
            f"`{strategy_id}` (`{version}`)"
            for strategy_id, version in metadata[
                "prompt_strategy_versions"
            ].items()
        ),
        f"- Python: `{metadata['python_version']}`",
        f"- Pricing profile: `{metadata['pricing_catalog_version']}`",
        "- Cost label: **simulated comparison estimate; no Bedrock charge**",
        "",
        "## Strategy metrics",
        "",
        "| Strategy | Grounded | Citation correct | Citation complete | Unsupported | Uncertainty | Safety | Approval | Complete | Format | Avg input | Avg output | Avg total | Simulated USD | Avg ms | P50 | P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy_id, metric in comparison.metrics.items():
        lines.append(
            f"| {strategy_id} "
            f"| {_rate(metric['grounded_answer_rate'])} "
            f"| {_rate(metric['correct_citation_rate'])} "
            f"| {_rate(metric['complete_citation_rate'])} "
            f"| {_rate(metric['unsupported_claim_rate'])} "
            f"| {_rate(metric['required_uncertainty_rate'])} "
            f"| {_rate(metric['safety_compliance_rate'])} "
            f"| {_rate(metric['approval_gate_compliance_rate'])} "
            f"| {_rate(metric['answer_completeness_rate'])} "
            f"| {_rate(metric['format_compliance_rate'])} "
            f"| {metric['average_input_tokens']:.1f} "
            f"| {metric['average_output_tokens']:.1f} "
            f"| {metric['average_total_tokens']:.1f} "
            f"| {metric['average_simulated_cost_usd']} "
            f"| {metric['average_latency_ms']:.3f} "
            f"| {metric['p50_latency_ms']:.3f} "
            f"| {metric['p95_latency_ms']:.3f} |"
        )

    selection = comparison.selection
    recommended = comparison.metrics[selection["recommended_strategy"]]
    baseline = comparison.metrics["baseline-concise"]
    token_delta = (
        recommended["average_total_tokens"]
        - baseline["average_total_tokens"]
    )
    cost_delta = (
        Decimal(recommended["average_simulated_cost_usd"])
        - Decimal(baseline["average_simulated_cost_usd"])
    )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Recommended overall: "
            f"**{selection['recommended_strategy']}**",
            f"- Best troubleshooting: "
            f"**{selection['best_troubleshooting_strategy']}**",
            f"- Best concise mode: "
            f"**{selection['best_concise_strategy']}**",
            f"- Best safety-sensitive: "
            f"**{selection['best_safety_sensitive_strategy']}**",
            f"- Rule: {selection['selection_rule']}",
            f"- Versus baseline: {token_delta:.1f} more average total tokens "
            f"and USD {format(cost_delta, 'f')} more simulated average cost.",
            "",
            "The existing application prompt default was not changed.",
            "",
            "## Performance by fake LLM mode",
            "",
            "| Strategy | Mode | Quality | Grounded | Complete | Avg output tokens | Avg latency ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy_id, metric in comparison.metrics.items():
        for mode, values in metric["by_mode"].items():
            lines.append(
                f"| {strategy_id} | {mode} "
                f"| {_rate(values['overall_quality_score'])} "
                f"| {_rate(values['grounded_answer_rate'])} "
                f"| {_rate(values['answer_completeness_rate'])} "
                f"| {values['average_output_tokens']:.1f} "
                f"| {values['average_latency_ms']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Performance by category",
            "",
            "| Strategy | Category | Quality | Grounded | Complete | Safety | Avg output | Avg ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy_id, metric in comparison.metrics.items():
        for category, values in metric["by_category"].items():
            lines.append(
                f"| {strategy_id} | {category} "
                f"| {_rate(values['overall_quality_score'])} "
                f"| {_rate(values['grounded_answer_rate'])} "
                f"| {_rate(values['answer_completeness_rate'])} "
                f"| {_rate(values['safety_compliance_rate'])} "
                f"| {values['average_output_tokens']:.1f} "
                f"| {values['average_latency_ms']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Performance by difficulty",
            "",
            "| Strategy | Difficulty | Quality | Grounded | Complete | Avg output | Avg ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy_id, metric in comparison.metrics.items():
        for difficulty, values in metric["by_difficulty"].items():
            lines.append(
                f"| {strategy_id} | {difficulty} "
                f"| {_rate(values['overall_quality_score'])} "
                f"| {_rate(values['grounded_answer_rate'])} "
                f"| {_rate(values['answer_completeness_rate'])} "
                f"| {values['average_output_tokens']:.1f} "
                f"| {values['average_latency_ms']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Performance by safety sensitivity",
            "",
            "| Strategy | Group | Quality | Grounded | Complete | Safety | Avg output | Avg ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy_id, metric in comparison.metrics.items():
        for group, values in metric["by_safety"].items():
            lines.append(
                f"| {strategy_id} | {group} "
                f"| {_rate(values['overall_quality_score'])} "
                f"| {_rate(values['grounded_answer_rate'])} "
                f"| {_rate(values['answer_completeness_rate'])} "
                f"| {_rate(values['safety_compliance_rate'])} "
                f"| {values['average_output_tokens']:.1f} "
                f"| {values['average_latency_ms']:.3f} |"
            )

    lines.extend(["", "## Failure analysis", ""])
    for strategy_id, failures in comparison.failure_analysis.items():
        lines.extend([f"### {strategy_id}", ""])
        for failure_type, case_ids in failures.items():
            display = ", ".join(case_ids[:12]) or "none"
            suffix = (
                f" (plus {len(case_ids) - 12} more)"
                if len(case_ids) > 12
                else ""
            )
            lines.append(
                f"- {failure_type.replace('_', ' ')}: "
                f"{display}{suffix}"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "Scores are exact string, citation, flag, section, and token-limit "
            "checks against synthetic labels. They can verify deterministic "
            "orchestration, formatting, grounding markers, uncertainty, and "
            "safety gates. They cannot establish factual fluency, nuanced "
            "reasoning, naturalness, robustness to unseen prompts, or real "
            "model quality.",
            "",
        ]
    )
    return "\n".join(lines)


def render_prompt_csv(comparison: PromptComparison) -> str:
    """Render one flat row per case, strategy, and fake mode."""

    output = io.StringIO(newline="")
    fields = [
        "strategy_id",
        "strategy_version",
        "fake_llm_mode",
        "case_id",
        "category",
        "difficulty",
        "safety_sensitive",
        "context_checksum",
        "required_source_ids",
        "cited_source_ids",
        "groundedness",
        "citation_correctness",
        "citation_completeness",
        "answer_relevance",
        "instruction_following",
        "uncertainty_handling",
        "safety_compliance",
        "approval_gate_compliance",
        "forbidden_claim_avoidance",
        "response_completeness",
        "response_conciseness",
        "structural_correctness",
        "prompt_tokens",
        "context_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "simulated_estimated_cost_usd",
        "charge_incurred",
        "latency_ms",
        "response",
    ]
    writer = csv.DictWriter(
        output,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    for outcome in comparison.outcomes:
        row = outcome.to_dict()
        score = row["scores"]
        writer.writerow(
            {
                "strategy_id": outcome.strategy_id,
                "strategy_version": outcome.strategy_version,
                "fake_llm_mode": outcome.mode.value,
                "case_id": outcome.case.case_id,
                "category": outcome.case.category,
                "difficulty": outcome.case.difficulty,
                "safety_sensitive": outcome.case.safety_sensitive,
                "context_checksum": outcome.context_checksum,
                "required_source_ids": ";".join(
                    outcome.case.required_source_ids
                ),
                "cited_source_ids": ";".join(
                    outcome.cited_source_ids
                ),
                **{
                    name: score[name]
                    for name in (
                        "groundedness",
                        "citation_correctness",
                        "citation_completeness",
                        "answer_relevance",
                        "instruction_following",
                        "uncertainty_handling",
                        "safety_compliance",
                        "approval_gate_compliance",
                        "forbidden_claim_avoidance",
                        "response_completeness",
                        "response_conciseness",
                        "structural_correctness",
                    )
                },
                "prompt_tokens": outcome.prompt_tokens,
                "context_tokens": outcome.context_tokens,
                "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens,
                "total_tokens": outcome.total_tokens,
                "simulated_estimated_cost_usd": format(
                    outcome.simulated_cost,
                    "f",
                ),
                "charge_incurred": False,
                "latency_ms": round(outcome.latency_ms, 6),
                "response": outcome.response,
            }
        )
    return output.getvalue()


def _rate(value: float) -> str:
    return f"{float(value):.4f}"
