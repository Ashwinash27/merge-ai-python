"""Experiment runners for H1, H2, H3 hypotheses."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..config import (
    ExperimentConfig,
    GPT_4O,
    CLAUDE_SONNET,
    CLAUDE_HAIKU,
    OPENROUTER_SONNET,
    OPENROUTER_HAIKU,
    RESULTS_DIR,
)
from ..core.resolver import resolve_conflict, Resolver, ResolutionResult
from ..core.classifier import classify_conflict, Classification, ConflictType
from ..core.validator import validate_resolution, ensemble_select, ValidationResult
from ..core.constraints import Constraint
from ..core.metrics import compute_metrics, ResolutionMetrics


@dataclass
class ConflictData:
    """Data for a single merge conflict."""

    id: str
    base: str
    ours: str
    theirs: str
    gold: Optional[str] = None
    path: Optional[Path] = None


@dataclass
class ExperimentResult:
    """Result of running an experiment on a single conflict."""

    conflict_id: str
    condition: str
    resolution: str
    metrics: ResolutionMetrics
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "conflict_id": self.conflict_id,
            "condition": self.condition,
            "resolution": self.resolution,
            "metrics": self.metrics.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class H1Results:
    """Results from H1 (Classification) experiment."""

    hypothesis: str = "H1: Classification improves resolution"
    conditions: list[str] = field(
        default_factory=lambda: ["no_classify", "with_classify"]
    )
    results: list[ExperimentResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "hypothesis": self.hypothesis,
            "conditions": self.conditions,
            "results": [r.to_dict() for r in self.results],
            "total_cost_usd": self.total_cost_usd,
            "duration_seconds": self.duration_seconds,
            "timestamp": datetime.now().isoformat(),
        }


@dataclass
class H2Results:
    """Results from H2 (Multi-model) experiment."""

    hypothesis: str = "H2: Multi-model beats single-model"
    conditions: list[str] = field(
        default_factory=lambda: [
            "sonnet_only",
            "gpt4o_only",
            "sonnet_gpt4o_validate",
            "ensemble_vote",
        ]
    )
    results: list[ExperimentResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "hypothesis": self.hypothesis,
            "conditions": self.conditions,
            "results": [r.to_dict() for r in self.results],
            "total_cost_usd": self.total_cost_usd,
            "duration_seconds": self.duration_seconds,
            "timestamp": datetime.now().isoformat(),
        }


@dataclass
class H3Results:
    """Results from H3 (Constraints) experiment."""

    hypothesis: str = "H3: Constraints reduce hallucination"
    conditions: list[str] = field(
        default_factory=lambda: [
            "baseline",
            "ast_only",
            "imports_only",
            "functions_only",
            "all_constraints",
        ]
    )
    results: list[ExperimentResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "hypothesis": self.hypothesis,
            "conditions": self.conditions,
            "results": [r.to_dict() for r in self.results],
            "total_cost_usd": self.total_cost_usd,
            "duration_seconds": self.duration_seconds,
            "timestamp": datetime.now().isoformat(),
        }


def run_h1_experiment(
    conflicts: list[ConflictData],
    config: Optional[ExperimentConfig] = None,
    verbose: bool = True,
) -> H1Results:
    """Run H1 experiment: Classification improves resolution.

    Tests whether classifying conflicts before resolution leads to better results.

    Conditions:
    - no_classify: Generic resolution prompt
    - with_classify: Classify first, then use type-specific prompt

    Args:
        conflicts: List of conflict data to process
        config: Experiment configuration
        verbose: Whether to print progress

    Returns:
        H1Results with all experiment data
    """
    config = config or ExperimentConfig()
    results = H1Results()
    start_time = time.time()
    total_cost = 0.0

    for i, conflict in enumerate(conflicts):
        if verbose:
            print(f"H1 [{i + 1}/{len(conflicts)}] Processing {conflict.id}...")

        # Condition 1: No classification
        result_no_classify = resolve_conflict(
            base=conflict.base,
            ours=conflict.ours,
            theirs=conflict.theirs,
            model=config.primary_model,
            max_retries=config.max_retries,
        )

        metrics_no_classify = compute_metrics(
            resolved=result_no_classify.code,
            base=conflict.base,
            ours=conflict.ours,
            theirs=conflict.theirs,
            gold=conflict.gold,
            input_tokens=result_no_classify.input_tokens,
            output_tokens=result_no_classify.output_tokens,
            cost_usd=result_no_classify.cost_usd,
            attempts=result_no_classify.attempts,
        )

        results.results.append(
            ExperimentResult(
                conflict_id=conflict.id,
                condition="no_classify",
                resolution=result_no_classify.code,
                metrics=metrics_no_classify,
                metadata={"resolution_result": result_no_classify.to_dict()},
            )
        )
        total_cost += result_no_classify.cost_usd

        # Condition 2: With classification
        classification = classify_conflict(
            base=conflict.base,
            ours=conflict.ours,
            theirs=conflict.theirs,
            model=config.primary_model,
        )
        classify_cost = (
            classification.input_tokens / 1000 * config.primary_model.cost_per_1k_input
            + classification.output_tokens
            / 1000
            * config.primary_model.cost_per_1k_output
        )
        total_cost += classify_cost

        result_with_classify = resolve_conflict(
            base=conflict.base,
            ours=conflict.ours,
            theirs=conflict.theirs,
            model=config.primary_model,
            conflict_type=classification.conflict_type,
            max_retries=config.max_retries,
        )

        metrics_with_classify = compute_metrics(
            resolved=result_with_classify.code,
            base=conflict.base,
            ours=conflict.ours,
            theirs=conflict.theirs,
            gold=conflict.gold,
            input_tokens=result_with_classify.input_tokens
            + classification.input_tokens,
            output_tokens=result_with_classify.output_tokens
            + classification.output_tokens,
            cost_usd=result_with_classify.cost_usd + classify_cost,
            attempts=result_with_classify.attempts,
        )

        results.results.append(
            ExperimentResult(
                conflict_id=conflict.id,
                condition="with_classify",
                resolution=result_with_classify.code,
                metrics=metrics_with_classify,
                metadata={
                    "classification": classification.to_dict(),
                    "resolution_result": result_with_classify.to_dict(),
                },
            )
        )
        total_cost += result_with_classify.cost_usd

        if verbose:
            print(
                f"  no_classify: gold_match={metrics_no_classify.gold_match_score:.2f}, "
                f"syntax_valid={metrics_no_classify.syntax_valid}"
            )
            print(
                f"  with_classify ({classification.conflict_type.value}): "
                f"gold_match={metrics_with_classify.gold_match_score:.2f}, "
                f"syntax_valid={metrics_with_classify.syntax_valid}"
            )

    results.total_cost_usd = total_cost
    results.duration_seconds = time.time() - start_time

    return results


def run_h2_experiment(
    conflicts: list[ConflictData],
    config: Optional[ExperimentConfig] = None,
    verbose: bool = True,
) -> H2Results:
    """Run H2 experiment: Multi-model beats single-model.

    Tests whether using multiple LLMs improves results.

    Conditions:
    - sonnet_only: Claude Sonnet resolves alone
    - gpt4o_only: GPT-4o resolves alone
    - sonnet_gpt4o_validate: Sonnet resolves, GPT-4o validates
    - ensemble_vote: Both resolve, pick better by heuristic

    Args:
        conflicts: List of conflict data to process
        config: Experiment configuration
        verbose: Whether to print progress

    Returns:
        H2Results with all experiment data
    """
    config = config or ExperimentConfig()
    results = H2Results()
    start_time = time.time()
    total_cost = 0.0

    # Use config's primary and secondary models (Sonnet and GPT-4o via OpenRouter)
    sonnet_resolver = Resolver(model=config.primary_model, max_retries=config.max_retries)
    gpt4o_resolver = Resolver(model=config.secondary_model, max_retries=config.max_retries)

    for i, conflict in enumerate(conflicts):
        if verbose:
            print(f"H2 [{i + 1}/{len(conflicts)}] Processing {conflict.id}...")

        # Condition 1: Sonnet only
        result_sonnet = sonnet_resolver.resolve(
            base=conflict.base, ours=conflict.ours, theirs=conflict.theirs
        )
        metrics_sonnet = compute_metrics(
            resolved=result_sonnet.code,
            base=conflict.base,
            ours=conflict.ours,
            theirs=conflict.theirs,
            gold=conflict.gold,
            input_tokens=result_sonnet.input_tokens,
            output_tokens=result_sonnet.output_tokens,
            cost_usd=result_sonnet.cost_usd,
            attempts=result_sonnet.attempts,
        )
        results.results.append(
            ExperimentResult(
                conflict_id=conflict.id,
                condition="sonnet_only",
                resolution=result_sonnet.code,
                metrics=metrics_sonnet,
            )
        )
        total_cost += result_sonnet.cost_usd

        # Condition 2: Haiku only
        result_gpt4o = gpt4o_resolver.resolve(
            base=conflict.base, ours=conflict.ours, theirs=conflict.theirs
        )
        metrics_gpt4o = compute_metrics(
            resolved=result_gpt4o.code,
            base=conflict.base,
            ours=conflict.ours,
            theirs=conflict.theirs,
            gold=conflict.gold,
            input_tokens=result_gpt4o.input_tokens,
            output_tokens=result_gpt4o.output_tokens,
            cost_usd=result_gpt4o.cost_usd,
            attempts=result_gpt4o.attempts,
        )
        results.results.append(
            ExperimentResult(
                conflict_id=conflict.id,
                condition="gpt4o_only",
                resolution=result_gpt4o.code,
                metrics=metrics_gpt4o,
            )
        )
        total_cost += result_gpt4o.cost_usd

        # Condition 3: Sonnet + Haiku validation
        validation = validate_resolution(
            resolved=result_sonnet.code,
            base=conflict.base,
            ours=conflict.ours,
            theirs=conflict.theirs,
            model=config.secondary_model,
        )
        validate_cost = (
            validation.input_tokens / 1000 * config.secondary_model.cost_per_1k_input
            + validation.output_tokens / 1000 * config.secondary_model.cost_per_1k_output
        )

        # If validation fails and suggests revision, retry with feedback
        if not validation.passed and validation.suggestions:
            result_validated = sonnet_resolver.resolve(
                base=conflict.base, ours=conflict.ours, theirs=conflict.theirs
            )
            total_validated_cost = (
                result_sonnet.cost_usd + validate_cost + result_validated.cost_usd
            )
        else:
            result_validated = result_sonnet
            total_validated_cost = result_sonnet.cost_usd + validate_cost

        metrics_validated = compute_metrics(
            resolved=result_validated.code,
            base=conflict.base,
            ours=conflict.ours,
            theirs=conflict.theirs,
            gold=conflict.gold,
            input_tokens=result_sonnet.input_tokens + validation.input_tokens,
            output_tokens=result_sonnet.output_tokens + validation.output_tokens,
            cost_usd=total_validated_cost,
            attempts=result_sonnet.attempts,
        )
        results.results.append(
            ExperimentResult(
                conflict_id=conflict.id,
                condition="sonnet_gpt4o_validate",
                resolution=result_validated.code,
                metrics=metrics_validated,
                metadata={"validation": validation.to_dict()},
            )
        )
        total_cost += validate_cost

        # Condition 4: Ensemble vote
        selected, reason = ensemble_select(
            resolution_a=result_sonnet.code,
            resolution_b=result_gpt4o.code,
            base=conflict.base,
            ours=conflict.ours,
            theirs=conflict.theirs,
        )
        metrics_ensemble = compute_metrics(
            resolved=selected,
            base=conflict.base,
            ours=conflict.ours,
            theirs=conflict.theirs,
            gold=conflict.gold,
            input_tokens=result_sonnet.input_tokens + result_gpt4o.input_tokens,
            output_tokens=result_sonnet.output_tokens + result_gpt4o.output_tokens,
            cost_usd=result_sonnet.cost_usd + result_gpt4o.cost_usd,
            attempts=1,
        )
        results.results.append(
            ExperimentResult(
                conflict_id=conflict.id,
                condition="ensemble_vote",
                resolution=selected,
                metrics=metrics_ensemble,
                metadata={"selection_reason": reason},
            )
        )

        if verbose:
            print(f"  sonnet_only: gold_match={metrics_sonnet.gold_match_score:.2f}")
            print(f"  gpt4o_only: gold_match={metrics_gpt4o.gold_match_score:.2f}")
            print(
                f"  sonnet_gpt4o_validate: gold_match={metrics_validated.gold_match_score:.2f}"
            )
            print(f"  ensemble_vote: gold_match={metrics_ensemble.gold_match_score:.2f}")

    results.total_cost_usd = total_cost
    results.duration_seconds = time.time() - start_time

    return results


def run_h3_experiment(
    conflicts: list[ConflictData],
    config: Optional[ExperimentConfig] = None,
    verbose: bool = True,
) -> H3Results:
    """Run H3 experiment: Constraints reduce hallucination.

    Tests whether structural constraints reduce LLM hallucination.

    Conditions:
    - baseline: No constraints
    - ast_only: Must be valid Python
    - imports_only: No new imports
    - functions_only: Preserve function signatures
    - all_constraints: All of the above

    Args:
        conflicts: List of conflict data to process
        config: Experiment configuration
        verbose: Whether to print progress

    Returns:
        H3Results with all experiment data
    """
    config = config or ExperimentConfig()
    results = H3Results()
    start_time = time.time()
    total_cost = 0.0

    conditions = [
        ("baseline", []),
        ("ast_only", [Constraint.AST]),
        ("imports_only", [Constraint.IMPORTS]),
        ("functions_only", [Constraint.FUNCTIONS]),
        ("all_constraints", [Constraint.ALL]),
    ]

    for i, conflict in enumerate(conflicts):
        if verbose:
            print(f"H3 [{i + 1}/{len(conflicts)}] Processing {conflict.id}...")

        for condition_name, constraints in conditions:
            resolver = Resolver(
                model=config.primary_model,
                constraints=constraints,
                max_retries=config.max_retries,
            )

            result = resolver.resolve(
                base=conflict.base, ours=conflict.ours, theirs=conflict.theirs
            )

            metrics = compute_metrics(
                resolved=result.code,
                base=conflict.base,
                ours=conflict.ours,
                theirs=conflict.theirs,
                gold=conflict.gold,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                attempts=result.attempts,
            )

            results.results.append(
                ExperimentResult(
                    conflict_id=conflict.id,
                    condition=condition_name,
                    resolution=result.code,
                    metrics=metrics,
                    metadata={"resolution_result": result.to_dict()},
                )
            )
            total_cost += result.cost_usd

            if verbose:
                print(
                    f"  {condition_name}: gold_match={metrics.gold_match_score:.2f}, "
                    f"halluc_imports={metrics.hallucinated_imports}, "
                    f"halluc_ids={metrics.hallucinated_identifiers}"
                )

    results.total_cost_usd = total_cost
    results.duration_seconds = time.time() - start_time

    return results


def load_conflicts_from_dir(
    conflicts_dir: Path, sample_size: Optional[int] = None
) -> list[ConflictData]:
    """Load conflict data from a directory structure.

    Expected structure:
    conflicts_dir/
        conflict_1/
            base.py
            ours.py
            theirs.py
            gold.py (optional)
        conflict_2/
            ...

    Args:
        conflicts_dir: Path to conflicts directory
        sample_size: Maximum number of conflicts to load

    Returns:
        List of ConflictData objects
    """
    conflicts = []

    for conflict_path in sorted(conflicts_dir.iterdir()):
        if not conflict_path.is_dir():
            continue

        base_file = conflict_path / "base.py"
        ours_file = conflict_path / "ours.py"
        theirs_file = conflict_path / "theirs.py"
        gold_file = conflict_path / "gold.py"

        if not all(f.exists() for f in [base_file, ours_file, theirs_file]):
            continue

        conflict = ConflictData(
            id=conflict_path.name,
            base=base_file.read_text(encoding="utf-8"),
            ours=ours_file.read_text(encoding="utf-8"),
            theirs=theirs_file.read_text(encoding="utf-8"),
            gold=gold_file.read_text(encoding="utf-8") if gold_file.exists() else None,
            path=conflict_path,
        )
        conflicts.append(conflict)

        if sample_size and len(conflicts) >= sample_size:
            break

    return conflicts


def save_results(results: H1Results | H2Results | H3Results, output_dir: Path) -> Path:
    """Save experiment results to JSON file.

    Args:
        results: Experiment results to save
        output_dir: Directory to save results in

    Returns:
        Path to the saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(results, H1Results):
        filename = "h1_classification.json"
    elif isinstance(results, H2Results):
        filename = "h2_multimodel.json"
    else:
        filename = "h3_constraints.json"

    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2)

    return output_path
