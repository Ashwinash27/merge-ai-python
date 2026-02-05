"""Evaluation framework for merge conflict resolution experiments."""

from .experiments import run_h1_experiment, run_h2_experiment, run_h3_experiment
from .benchmark import Benchmark, run_benchmark
from .compare import compare_to_gold, semantic_compare
from .error_analysis import analyze_errors, ErrorAnalysis
from .report import generate_report

__all__ = [
    # Experiments
    "run_h1_experiment",
    "run_h2_experiment",
    "run_h3_experiment",
    # Benchmark
    "Benchmark",
    "run_benchmark",
    # Compare
    "compare_to_gold",
    "semantic_compare",
    # Error Analysis
    "analyze_errors",
    "ErrorAnalysis",
    # Report
    "generate_report",
]
