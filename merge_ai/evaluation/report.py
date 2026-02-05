"""Generate Markdown reports from experiment results."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime

from .error_analysis import analyze_errors, ErrorAnalysis, format_error_report


@dataclass
class HypothesisSummary:
    """Summary statistics for a hypothesis."""

    name: str
    description: str
    conditions: list[str]
    total_conflicts: int
    total_cost_usd: float
    duration_seconds: float

    # Per-condition metrics
    condition_metrics: dict[str, dict]

    # Best condition
    best_condition: str
    best_gold_score: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "conditions": self.conditions,
            "total_conflicts": self.total_conflicts,
            "total_cost_usd": self.total_cost_usd,
            "duration_seconds": self.duration_seconds,
            "condition_metrics": self.condition_metrics,
            "best_condition": self.best_condition,
            "best_gold_score": self.best_gold_score,
        }


def summarize_hypothesis(results_data: dict, hypothesis: str) -> HypothesisSummary:
    """Summarize results for a single hypothesis.

    Args:
        results_data: Loaded experiment results
        hypothesis: Hypothesis name (H1, H2, H3)

    Returns:
        HypothesisSummary with aggregated metrics
    """
    descriptions = {
        "H1": "Classification improves resolution",
        "H2": "Multi-model beats single-model",
        "H3": "Constraints reduce hallucination",
    }

    results = results_data.get("results", [])
    conditions = results_data.get("conditions", [])

    # Aggregate metrics per condition
    condition_metrics = {}
    for condition in conditions:
        condition_results = [r for r in results if r.get("condition") == condition]
        if not condition_results:
            continue

        metrics_list = [r.get("metrics", {}) for r in condition_results]

        avg_gold = sum(m.get("gold_match_score", 0) for m in metrics_list) / len(
            metrics_list
        )
        syntax_valid_count = sum(1 for m in metrics_list if m.get("syntax_valid", False))
        avg_halluc_imports = sum(
            m.get("hallucinated_imports", 0) for m in metrics_list
        ) / len(metrics_list)
        avg_halluc_ids = sum(
            m.get("hallucinated_identifiers", 0) for m in metrics_list
        ) / len(metrics_list)
        total_cost = sum(m.get("cost_usd", 0) for m in metrics_list)

        condition_metrics[condition] = {
            "count": len(condition_results),
            "avg_gold_match": avg_gold,
            "syntax_valid_rate": syntax_valid_count / len(condition_results),
            "avg_hallucinated_imports": avg_halluc_imports,
            "avg_hallucinated_identifiers": avg_halluc_ids,
            "total_cost_usd": total_cost,
        }

    # Find best condition
    best_condition = max(
        condition_metrics.keys(),
        key=lambda c: condition_metrics[c]["avg_gold_match"],
    )
    best_gold_score = condition_metrics[best_condition]["avg_gold_match"]

    # Get unique conflicts
    conflict_ids = set(r.get("conflict_id") for r in results)

    return HypothesisSummary(
        name=hypothesis,
        description=descriptions.get(hypothesis, ""),
        conditions=conditions,
        total_conflicts=len(conflict_ids),
        total_cost_usd=results_data.get("total_cost_usd", 0),
        duration_seconds=results_data.get("duration_seconds", 0),
        condition_metrics=condition_metrics,
        best_condition=best_condition,
        best_gold_score=best_gold_score,
    )


def generate_hypothesis_report(
    results_data: dict, hypothesis: str, error_analysis: Optional[ErrorAnalysis] = None
) -> str:
    """Generate Markdown report for a single hypothesis.

    Args:
        results_data: Loaded experiment results
        hypothesis: Hypothesis name
        error_analysis: Optional error analysis

    Returns:
        Markdown formatted report
    """
    summary = summarize_hypothesis(results_data, hypothesis)
    lines = []

    lines.append(f"## {summary.name}: {summary.description}")
    lines.append("")

    # Overview
    lines.append("### Overview")
    lines.append("")
    lines.append(f"- **Conflicts tested:** {summary.total_conflicts}")
    lines.append(f"- **Conditions:** {', '.join(summary.conditions)}")
    lines.append(f"- **Total cost:** ${summary.total_cost_usd:.2f}")
    lines.append(f"- **Duration:** {summary.duration_seconds:.1f} seconds")
    lines.append("")

    # Results table
    lines.append("### Results by Condition")
    lines.append("")
    lines.append(
        "| Condition | Gold Match | Syntax Valid | Halluc. Imports | Halluc. IDs | Cost |"
    )
    lines.append(
        "|-----------|------------|--------------|-----------------|-------------|------|"
    )

    for condition in summary.conditions:
        if condition not in summary.condition_metrics:
            continue
        m = summary.condition_metrics[condition]
        lines.append(
            f"| {condition} | {m['avg_gold_match']:.2f} | "
            f"{m['syntax_valid_rate']:.0%} | {m['avg_hallucinated_imports']:.1f} | "
            f"{m['avg_hallucinated_identifiers']:.1f} | ${m['total_cost_usd']:.2f} |"
        )

    lines.append("")

    # Key finding
    lines.append("### Key Finding")
    lines.append("")
    lines.append(
        f"**Best condition: `{summary.best_condition}`** with average gold match "
        f"score of **{summary.best_gold_score:.2f}**"
    )
    lines.append("")

    # Error analysis if available
    if error_analysis and error_analysis.total_with_errors > 0:
        lines.append("### Error Analysis")
        lines.append("")
        lines.append(f"- Error rate: {error_analysis.error_rate:.1%}")
        lines.append(f"- Retry success rate: {error_analysis.retry_success_rate:.1%}")
        lines.append("")

        if error_analysis.error_stats:
            lines.append("**Error type breakdown:**")
            lines.append("")
            for error_type, stats in sorted(
                error_analysis.error_stats.items(), key=lambda x: -x[1].count
            ):
                lines.append(
                    f"- {error_type}: {stats.count} occurrences "
                    f"({stats.retry_success_rate:.0%} retry success)"
                )
            lines.append("")

    return "\n".join(lines)


def generate_report(results_dir: Path, output_path: Optional[Path] = None) -> str:
    """Generate comprehensive Markdown report from all experiment results.

    Args:
        results_dir: Directory containing result JSON files
        output_path: Optional path to write report to

    Returns:
        Complete Markdown report
    """
    lines = []

    # Header
    lines.append("# MERGE-AI Experiment Results")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")

    total_cost = 0.0
    total_conflicts = 0
    summaries = {}

    # Load and summarize each hypothesis
    hypothesis_files = {
        "H1": "h1_classification.json",
        "H2": "h2_multimodel.json",
        "H3": "h3_constraints.json",
    }

    for hypothesis, filename in hypothesis_files.items():
        filepath = results_dir / filename
        if not filepath.exists():
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = summarize_hypothesis(data, hypothesis)
        summaries[hypothesis] = (data, summary)
        total_cost += summary.total_cost_usd
        total_conflicts = max(total_conflicts, summary.total_conflicts)

    if not summaries:
        lines.append("*No experiment results found.*")
        return "\n".join(lines)

    lines.append(f"- **Total experiments run:** {len(summaries)}")
    lines.append(f"- **Conflicts evaluated:** {total_conflicts}")
    lines.append(f"- **Total cost:** ${total_cost:.2f}")
    lines.append("")

    # Key findings summary
    lines.append("### Key Findings")
    lines.append("")

    for hypothesis, (data, summary) in summaries.items():
        improvement = ""
        if hypothesis == "H1":
            no_classify = summary.condition_metrics.get("no_classify", {})
            with_classify = summary.condition_metrics.get("with_classify", {})
            if no_classify and with_classify:
                diff = (
                    with_classify["avg_gold_match"] - no_classify["avg_gold_match"]
                )
                if diff > 0:
                    improvement = f" (+{diff:.2f} gold match improvement)"
                else:
                    improvement = f" ({diff:.2f} gold match difference)"
        elif hypothesis == "H3":
            baseline = summary.condition_metrics.get("baseline", {})
            all_constraints = summary.condition_metrics.get("all_constraints", {})
            if baseline and all_constraints:
                import_diff = (
                    baseline["avg_hallucinated_imports"]
                    - all_constraints["avg_hallucinated_imports"]
                )
                if import_diff > 0:
                    improvement = f" (-{import_diff:.1f} hallucinated imports)"

        lines.append(
            f"- **{hypothesis}:** Best condition = `{summary.best_condition}` "
            f"(gold match: {summary.best_gold_score:.2f}){improvement}"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed hypothesis reports
    for hypothesis, (data, summary) in summaries.items():
        error_analysis = analyze_errors(data)
        report = generate_hypothesis_report(data, hypothesis, error_analysis)
        lines.append(report)
        lines.append("---")
        lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Metrics")
    lines.append("")
    lines.append("1. **Gold Match Score (0-1):** Semantic similarity to gold standard")
    lines.append("2. **Syntax Valid (%):** Percentage of outputs that parse as valid Python")
    lines.append("3. **Hallucinated Imports:** Average count of imports not in input files")
    lines.append(
        "4. **Hallucinated Identifiers:** Average count of variables/functions not in inputs"
    )
    lines.append("5. **Cost:** API cost in USD")
    lines.append("")

    lines.append("### Error Handling")
    lines.append("")
    lines.append("- Maximum 2 retries per resolution")
    lines.append("- Error-specific retry prompts")
    lines.append("- Errors tracked: syntax errors, truncation, hallucinated imports/identifiers")
    lines.append("")

    # Limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Sample size limited by budget constraints")
    lines.append("- Gold standard may not always represent optimal merge")
    lines.append("- Semantic similarity metric is approximate")
    lines.append("- Results may vary with different LLM versions")
    lines.append("")

    # Future work
    lines.append("## Future Work")
    lines.append("")
    lines.append("- Larger sample sizes for statistical significance")
    lines.append("- Additional constraint types")
    lines.append("- Human evaluation of semantic correctness")
    lines.append("- Cross-language support")
    lines.append("")

    report = "\n".join(lines)

    # Write to file if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def generate_csv_summary(results_dir: Path, output_path: Optional[Path] = None) -> str:
    """Generate CSV summary of all results.

    Args:
        results_dir: Directory containing result JSON files
        output_path: Optional path to write CSV to

    Returns:
        CSV content
    """
    lines = []

    # Header
    lines.append(
        "hypothesis,condition,conflict_id,gold_match,syntax_valid,"
        "halluc_imports,halluc_ids,cost_usd,attempts"
    )

    # Load each hypothesis
    hypothesis_files = {
        "H1": "h1_classification.json",
        "H2": "h2_multimodel.json",
        "H3": "h3_constraints.json",
    }

    for hypothesis, filename in hypothesis_files.items():
        filepath = results_dir / filename
        if not filepath.exists():
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for result in data.get("results", []):
            metrics = result.get("metrics", {})
            lines.append(
                f"{hypothesis},"
                f"{result.get('condition', '')},"
                f"{result.get('conflict_id', '')},"
                f"{metrics.get('gold_match_score', 0):.4f},"
                f"{1 if metrics.get('syntax_valid', False) else 0},"
                f"{metrics.get('hallucinated_imports', 0)},"
                f"{metrics.get('hallucinated_identifiers', 0)},"
                f"{metrics.get('cost_usd', 0):.4f},"
                f"{metrics.get('attempts', 1)}"
            )

    csv_content = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

    return csv_content
