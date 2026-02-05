"""Aggregate error statistics and analysis."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core.errors import ErrorType


@dataclass
class ErrorStats:
    """Statistics for a single error type."""

    error_type: str
    count: int
    retry_success_count: int
    retry_failure_count: int

    @property
    def total_with_retry(self) -> int:
        return self.retry_success_count + self.retry_failure_count

    @property
    def retry_success_rate(self) -> float:
        if self.total_with_retry == 0:
            return 0.0
        return self.retry_success_count / self.total_with_retry

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "count": self.count,
            "retry_success_count": self.retry_success_count,
            "retry_failure_count": self.retry_failure_count,
            "retry_success_rate": self.retry_success_rate,
        }


@dataclass
class ErrorAnalysis:
    """Complete error analysis results."""

    # Overall stats
    total_resolutions: int = 0
    total_with_errors: int = 0
    total_retries: int = 0
    total_retry_successes: int = 0

    # Per-error-type stats
    error_stats: dict[str, ErrorStats] = field(default_factory=dict)

    # Per-condition stats
    condition_error_rates: dict[str, float] = field(default_factory=dict)

    # Correlations
    error_gold_correlations: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_resolutions": self.total_resolutions,
            "total_with_errors": self.total_with_errors,
            "total_retries": self.total_retries,
            "total_retry_successes": self.total_retry_successes,
            "overall_error_rate": self.error_rate,
            "overall_retry_success_rate": self.retry_success_rate,
            "error_stats": {k: v.to_dict() for k, v in self.error_stats.items()},
            "condition_error_rates": self.condition_error_rates,
            "error_gold_correlations": self.error_gold_correlations,
        }

    @property
    def error_rate(self) -> float:
        if self.total_resolutions == 0:
            return 0.0
        return self.total_with_errors / self.total_resolutions

    @property
    def retry_success_rate(self) -> float:
        if self.total_retries == 0:
            return 0.0
        return self.total_retry_successes / self.total_retries


def analyze_errors(results_data: dict) -> ErrorAnalysis:
    """Analyze errors from experiment results.

    Args:
        results_data: Loaded experiment results (from JSON)

    Returns:
        ErrorAnalysis with aggregated statistics
    """
    analysis = ErrorAnalysis()

    # Count errors across all results
    error_counts = Counter()
    retry_successes = defaultdict(int)
    retry_failures = defaultdict(int)
    condition_errors = defaultdict(lambda: {"total": 0, "errors": 0})
    error_gold_scores = defaultdict(list)

    results = results_data.get("results", [])
    analysis.total_resolutions = len(results)

    for result in results:
        condition = result.get("condition", "unknown")
        metadata = result.get("metadata", {})
        metrics = result.get("metrics", {})
        gold_score = metrics.get("gold_match_score", 0)

        condition_errors[condition]["total"] += 1

        # Check for errors in resolution result
        resolution_result = metadata.get("resolution_result", {})
        errors_encountered = resolution_result.get("errors_encountered", [])

        if errors_encountered:
            analysis.total_with_errors += 1
            condition_errors[condition]["errors"] += 1

            for error in errors_encountered:
                error_type = error.get("error_type", "unknown")
                error_counts[error_type] += 1
                analysis.total_retries += 1

                # Track gold score correlation with this error type
                error_gold_scores[error_type].append(gold_score)

            # Check if final resolution was successful after retries
            final_success = resolution_result.get("success", False)
            last_error = errors_encountered[-1].get("error_type", "unknown")

            if final_success:
                retry_successes[last_error] += 1
                analysis.total_retry_successes += 1
            else:
                retry_failures[last_error] += 1

    # Build error stats
    for error_type, count in error_counts.items():
        analysis.error_stats[error_type] = ErrorStats(
            error_type=error_type,
            count=count,
            retry_success_count=retry_successes[error_type],
            retry_failure_count=retry_failures[error_type],
        )

    # Calculate condition error rates
    for condition, stats in condition_errors.items():
        if stats["total"] > 0:
            analysis.condition_error_rates[condition] = (
                stats["errors"] / stats["total"]
            )

    # Calculate error-gold correlations (average gold score when error occurs)
    for error_type, scores in error_gold_scores.items():
        if scores:
            analysis.error_gold_correlations[error_type] = sum(scores) / len(scores)

    return analysis


def analyze_from_file(results_path: Path) -> ErrorAnalysis:
    """Load results from file and analyze errors.

    Args:
        results_path: Path to results JSON file

    Returns:
        ErrorAnalysis with aggregated statistics
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return analyze_errors(data)


def analyze_all_hypotheses(results_dir: Path) -> dict[str, ErrorAnalysis]:
    """Analyze errors from all hypothesis result files.

    Args:
        results_dir: Directory containing result files

    Returns:
        Dict mapping hypothesis name to ErrorAnalysis
    """
    analyses = {}

    files = {
        "H1": "h1_classification.json",
        "H2": "h2_multimodel.json",
        "H3": "h3_constraints.json",
    }

    for hypothesis, filename in files.items():
        filepath = results_dir / filename
        if filepath.exists():
            analyses[hypothesis] = analyze_from_file(filepath)

    return analyses


def format_error_report(analysis: ErrorAnalysis) -> str:
    """Format error analysis as a readable report.

    Args:
        analysis: ErrorAnalysis to format

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ERROR ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Overall stats
    lines.append("OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total resolutions: {analysis.total_resolutions}")
    lines.append(f"Resolutions with errors: {analysis.total_with_errors}")
    lines.append(f"Error rate: {analysis.error_rate:.1%}")
    lines.append(f"Total retries: {analysis.total_retries}")
    lines.append(f"Retry success rate: {analysis.retry_success_rate:.1%}")
    lines.append("")

    # Per-error-type stats
    lines.append("ERROR TYPE BREAKDOWN")
    lines.append("-" * 40)
    for error_type, stats in sorted(
        analysis.error_stats.items(), key=lambda x: -x[1].count
    ):
        lines.append(f"\n{error_type}:")
        lines.append(f"  Count: {stats.count}")
        lines.append(f"  Retry success rate: {stats.retry_success_rate:.1%}")
        if error_type in analysis.error_gold_correlations:
            lines.append(
                f"  Avg gold score when error: {analysis.error_gold_correlations[error_type]:.2f}"
            )

    lines.append("")

    # Per-condition stats
    if analysis.condition_error_rates:
        lines.append("ERROR RATES BY CONDITION")
        lines.append("-" * 40)
        for condition, rate in sorted(
            analysis.condition_error_rates.items(), key=lambda x: x[1]
        ):
            lines.append(f"  {condition}: {rate:.1%}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def save_error_analysis(
    analyses: dict[str, ErrorAnalysis], output_dir: Path
) -> Path:
    """Save error analysis to JSON file.

    Args:
        analyses: Dict mapping hypothesis to ErrorAnalysis
        output_dir: Directory to save to

    Returns:
        Path to saved file
    """
    output_path = output_dir / "error_analysis.json"
    data = {name: analysis.to_dict() for name, analysis in analyses.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return output_path
