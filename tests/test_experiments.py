"""Tests for experiment data structures."""

import pytest
from pathlib import Path
from merge_ai.evaluation.experiments import (
    ConflictData,
    ExperimentResult,
    H1Results,
    H2Results,
    H3Results,
    load_conflicts_from_dir,
)
from merge_ai.core.metrics import ResolutionMetrics


class TestConflictData:
    """Tests for ConflictData."""

    def test_basic_creation(self):
        """Should create ConflictData with required fields."""
        conflict = ConflictData(
            id="test1",
            base="base code",
            ours="our code",
            theirs="their code",
        )
        assert conflict.id == "test1"
        assert conflict.base == "base code"
        assert conflict.ours == "our code"
        assert conflict.theirs == "their code"
        assert conflict.gold is None

    def test_with_gold(self):
        """Should support optional gold field."""
        conflict = ConflictData(
            id="test1",
            base="base code",
            ours="our code",
            theirs="their code",
            gold="gold code",
        )
        assert conflict.gold == "gold code"


class TestExperimentResult:
    """Tests for ExperimentResult."""

    def test_to_dict(self):
        """Should convert to dict properly."""
        metrics = ResolutionMetrics(
            syntax_valid=True,
            hallucinated_imports=0,
            hallucinated_identifiers=1,
            hallucinated_import_names=[],
            hallucinated_identifier_names=["new_var"],
            gold_match_score=0.85,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.01,
            attempts=1,
            final_success=True,
        )

        result = ExperimentResult(
            conflict_id="test1",
            condition="baseline",
            resolution="x = 1",
            metrics=metrics,
            metadata={"extra": "info"},
        )

        d = result.to_dict()
        assert d["conflict_id"] == "test1"
        assert d["condition"] == "baseline"
        assert d["resolution"] == "x = 1"
        assert d["metrics"]["syntax_valid"] is True
        assert d["metadata"]["extra"] == "info"


class TestH1Results:
    """Tests for H1Results."""

    def test_default_conditions(self):
        """H1 should have correct default conditions."""
        results = H1Results()
        assert "no_classify" in results.conditions
        assert "with_classify" in results.conditions

    def test_hypothesis_description(self):
        """H1 should have correct hypothesis."""
        results = H1Results()
        assert "Classification" in results.hypothesis

    def test_to_dict(self):
        """Should convert to dict with all fields."""
        results = H1Results()
        results.total_cost_usd = 5.0
        results.duration_seconds = 120.0

        d = results.to_dict()
        assert "hypothesis" in d
        assert "conditions" in d
        assert "results" in d
        assert d["total_cost_usd"] == 5.0
        assert d["duration_seconds"] == 120.0
        assert "timestamp" in d


class TestH2Results:
    """Tests for H2Results."""

    def test_default_conditions(self):
        """H2 should have correct default conditions."""
        results = H2Results()
        assert "sonnet_only" in results.conditions
        assert "gpt4o_only" in results.conditions
        assert "sonnet_gpt4o_validate" in results.conditions
        assert "ensemble_vote" in results.conditions

    def test_hypothesis_description(self):
        """H2 should have correct hypothesis."""
        results = H2Results()
        assert "Multi-model" in results.hypothesis


class TestH3Results:
    """Tests for H3Results."""

    def test_default_conditions(self):
        """H3 should have correct default conditions."""
        results = H3Results()
        assert "baseline" in results.conditions
        assert "ast_only" in results.conditions
        assert "imports_only" in results.conditions
        assert "functions_only" in results.conditions
        assert "all_constraints" in results.conditions

    def test_hypothesis_description(self):
        """H3 should have correct hypothesis."""
        results = H3Results()
        assert "Constraints" in results.hypothesis or "hallucination" in results.hypothesis.lower()


class TestLoadConflictsFromDir:
    """Tests for loading conflicts from directory."""

    def test_load_existing_conflicts(self, tmp_path):
        """Should load conflicts from properly structured directory."""
        # Create test conflict
        conflict_dir = tmp_path / "test_conflict"
        conflict_dir.mkdir()
        (conflict_dir / "base.py").write_text("x = 0")
        (conflict_dir / "ours.py").write_text("x = 1")
        (conflict_dir / "theirs.py").write_text("x = 2")
        (conflict_dir / "gold.py").write_text("x = 1")

        conflicts = load_conflicts_from_dir(tmp_path)
        assert len(conflicts) == 1
        assert conflicts[0].id == "test_conflict"
        assert conflicts[0].base == "x = 0"
        assert conflicts[0].gold == "x = 1"

    def test_skip_incomplete_conflicts(self, tmp_path):
        """Should skip conflicts missing required files."""
        # Create incomplete conflict (missing ours.py)
        conflict_dir = tmp_path / "incomplete"
        conflict_dir.mkdir()
        (conflict_dir / "base.py").write_text("x = 0")
        (conflict_dir / "theirs.py").write_text("x = 2")

        conflicts = load_conflicts_from_dir(tmp_path)
        assert len(conflicts) == 0

    def test_sample_size_limit(self, tmp_path):
        """Should respect sample_size limit."""
        # Create 5 conflicts
        for i in range(5):
            conflict_dir = tmp_path / f"conflict_{i}"
            conflict_dir.mkdir()
            (conflict_dir / "base.py").write_text(f"x = {i}")
            (conflict_dir / "ours.py").write_text(f"x = {i + 1}")
            (conflict_dir / "theirs.py").write_text(f"x = {i + 2}")

        conflicts = load_conflicts_from_dir(tmp_path, sample_size=3)
        assert len(conflicts) == 3
