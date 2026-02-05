"""Benchmark orchestration with cost tracking and checkpointing."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..config import ExperimentConfig, RESULTS_DIR
from .experiments import (
    ConflictData,
    H1Results,
    H2Results,
    H3Results,
    run_h1_experiment,
    run_h2_experiment,
    run_h3_experiment,
    load_conflicts_from_dir,
    save_results,
)


@dataclass
class BenchmarkProgress:
    """Track progress of a benchmark run."""

    hypothesis: str
    total_conflicts: int
    completed_conflicts: int = 0
    current_cost_usd: float = 0.0
    start_time: Optional[float] = None
    checkpoint_file: Optional[Path] = None

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def progress_percent(self) -> float:
        if self.total_conflicts == 0:
            return 100.0
        return (self.completed_conflicts / self.total_conflicts) * 100

    def to_dict(self) -> dict:
        return {
            "hypothesis": self.hypothesis,
            "total_conflicts": self.total_conflicts,
            "completed_conflicts": self.completed_conflicts,
            "current_cost_usd": self.current_cost_usd,
            "elapsed_seconds": self.elapsed_seconds,
            "progress_percent": self.progress_percent,
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""

    h1_results: Optional[H1Results] = None
    h2_results: Optional[H2Results] = None
    h3_results: Optional[H3Results] = None
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "h1_results": self.h1_results.to_dict() if self.h1_results else None,
            "h2_results": self.h2_results.to_dict() if self.h2_results else None,
            "h3_results": self.h3_results.to_dict() if self.h3_results else None,
            "total_cost_usd": self.total_cost_usd,
            "total_duration_seconds": self.total_duration_seconds,
            "completed_at": self.completed_at,
        }


class Benchmark:
    """Orchestrates running experiments with cost tracking and checkpointing."""

    def __init__(
        self,
        conflicts_dir: Path,
        output_dir: Optional[Path] = None,
        config: Optional[ExperimentConfig] = None,
    ):
        """Initialize the benchmark.

        Args:
            conflicts_dir: Directory containing conflict data
            output_dir: Directory to save results (defaults to RESULTS_DIR)
            config: Experiment configuration
        """
        self.conflicts_dir = conflicts_dir
        self.output_dir = output_dir or RESULTS_DIR
        self.config = config or ExperimentConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        hypotheses: Optional[list[str]] = None,
        sample_size: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """Run the benchmark for specified hypotheses.

        Args:
            hypotheses: List of hypotheses to run ("H1", "H2", "H3", or "all")
            sample_size: Number of conflicts to process per hypothesis
            max_cost_usd: Maximum cost before aborting
            verbose: Whether to print progress

        Returns:
            BenchmarkResult with all experiment data
        """
        hypotheses = hypotheses or ["all"]
        if "all" in hypotheses:
            hypotheses = ["H1", "H2", "H3"]

        sample_size = sample_size or self.config.sample_size
        max_cost_usd = max_cost_usd or self.config.max_cost_usd

        # Load conflicts
        conflicts = load_conflicts_from_dir(self.conflicts_dir, sample_size)
        if not conflicts:
            raise ValueError(f"No conflicts found in {self.conflicts_dir}")

        if verbose:
            print(f"Loaded {len(conflicts)} conflicts from {self.conflicts_dir}")
            print(f"Running hypotheses: {', '.join(hypotheses)}")
            print(f"Max cost budget: ${max_cost_usd:.2f}")
            print("-" * 50)

        result = BenchmarkResult()
        start_time = time.time()
        total_cost = 0.0

        # Run each hypothesis
        if "H1" in hypotheses:
            if verbose:
                print("\n=== Running H1: Classification Experiment ===")

            h1_results = run_h1_experiment(
                conflicts=conflicts, config=self.config, verbose=verbose
            )
            result.h1_results = h1_results
            total_cost += h1_results.total_cost_usd

            # Save intermediate results
            save_results(h1_results, self.output_dir)

            if verbose:
                print(f"H1 complete. Cost: ${h1_results.total_cost_usd:.2f}")

            if total_cost > max_cost_usd:
                print(f"WARNING: Cost ${total_cost:.2f} exceeds budget ${max_cost_usd:.2f}")
                print("Stopping early.")
                result.total_cost_usd = total_cost
                result.total_duration_seconds = time.time() - start_time
                result.completed_at = datetime.now().isoformat()
                return result

        if "H2" in hypotheses:
            if verbose:
                print("\n=== Running H2: Multi-Model Experiment ===")

            h2_results = run_h2_experiment(
                conflicts=conflicts, config=self.config, verbose=verbose
            )
            result.h2_results = h2_results
            total_cost += h2_results.total_cost_usd

            # Save intermediate results
            save_results(h2_results, self.output_dir)

            if verbose:
                print(f"H2 complete. Cost: ${h2_results.total_cost_usd:.2f}")

            if total_cost > max_cost_usd:
                print(f"WARNING: Cost ${total_cost:.2f} exceeds budget ${max_cost_usd:.2f}")
                print("Stopping early.")
                result.total_cost_usd = total_cost
                result.total_duration_seconds = time.time() - start_time
                result.completed_at = datetime.now().isoformat()
                return result

        if "H3" in hypotheses:
            if verbose:
                print("\n=== Running H3: Constraints Experiment ===")

            h3_results = run_h3_experiment(
                conflicts=conflicts, config=self.config, verbose=verbose
            )
            result.h3_results = h3_results
            total_cost += h3_results.total_cost_usd

            # Save intermediate results
            save_results(h3_results, self.output_dir)

            if verbose:
                print(f"H3 complete. Cost: ${h3_results.total_cost_usd:.2f}")

        result.total_cost_usd = total_cost
        result.total_duration_seconds = time.time() - start_time
        result.completed_at = datetime.now().isoformat()

        # Save complete results
        self._save_complete_results(result)

        if verbose:
            print("\n" + "=" * 50)
            print("BENCHMARK COMPLETE")
            print(f"Total cost: ${total_cost:.2f}")
            print(f"Total time: {result.total_duration_seconds:.1f} seconds")
            print(f"Results saved to: {self.output_dir}")

        return result

    def _save_complete_results(self, result: BenchmarkResult) -> Path:
        """Save complete benchmark results."""
        output_path = self.output_dir / "benchmark_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        return output_path

    def run_pilot(
        self, sample_size: int = 5, verbose: bool = True
    ) -> BenchmarkResult:
        """Run a pilot test with a small sample.

        Args:
            sample_size: Number of conflicts for pilot (default 5)
            verbose: Whether to print progress

        Returns:
            BenchmarkResult from pilot run
        """
        if verbose:
            print(f"Running PILOT with {sample_size} conflicts...")
            print("This will test all hypotheses and conditions.")

        return self.run(
            hypotheses=["all"],
            sample_size=sample_size,
            max_cost_usd=20.0,  # Lower budget for pilot
            verbose=verbose,
        )


def run_benchmark(
    conflicts_dir: Path,
    output_dir: Optional[Path] = None,
    hypotheses: Optional[list[str]] = None,
    sample_size: int = 30,
    max_cost_usd: float = 100.0,
    verbose: bool = True,
) -> BenchmarkResult:
    """Convenience function to run the benchmark.

    Args:
        conflicts_dir: Directory containing conflict data
        output_dir: Directory to save results
        hypotheses: List of hypotheses to run
        sample_size: Number of conflicts per hypothesis
        max_cost_usd: Maximum cost budget
        verbose: Whether to print progress

    Returns:
        BenchmarkResult with all experiment data
    """
    benchmark = Benchmark(conflicts_dir=conflicts_dir, output_dir=output_dir)
    return benchmark.run(
        hypotheses=hypotheses,
        sample_size=sample_size,
        max_cost_usd=max_cost_usd,
        verbose=verbose,
    )
