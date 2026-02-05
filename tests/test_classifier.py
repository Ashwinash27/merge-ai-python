"""Tests for conflict classifier."""

import pytest
from merge_ai.core.classifier import (
    ConflictType,
    Classification,
    get_type_specific_prompt,
)


class TestConflictType:
    """Tests for ConflictType enum."""

    def test_syntactic_value(self):
        """SYNTACTIC should have correct value."""
        assert ConflictType.SYNTACTIC.value == "syntactic"

    def test_semantic_value(self):
        """SEMANTIC should have correct value."""
        assert ConflictType.SEMANTIC.value == "semantic"

    def test_structural_value(self):
        """STRUCTURAL should have correct value."""
        assert ConflictType.STRUCTURAL.value == "structural"

    def test_unknown_value(self):
        """UNKNOWN should have correct value."""
        assert ConflictType.UNKNOWN.value == "unknown"


class TestClassification:
    """Tests for Classification dataclass."""

    def test_to_dict(self):
        """Classification should convert to dict properly."""
        classification = Classification(
            conflict_type=ConflictType.SYNTACTIC,
            confidence=0.9,
            rationale="Import changes only",
            strategy="Combine imports",
            input_tokens=100,
            output_tokens=50,
        )

        d = classification.to_dict()
        assert d["conflict_type"] == "syntactic"
        assert d["confidence"] == 0.9
        assert d["rationale"] == "Import changes only"
        assert d["strategy"] == "Combine imports"
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50


class TestTypeSpecificPrompts:
    """Tests for type-specific resolution prompts."""

    def test_syntactic_prompt(self):
        """SYNTACTIC prompt should mention imports and formatting."""
        prompt = get_type_specific_prompt(ConflictType.SYNTACTIC)
        assert "SYNTACTIC" in prompt
        assert "import" in prompt.lower() or "format" in prompt.lower()

    def test_semantic_prompt(self):
        """SEMANTIC prompt should mention behavior preservation."""
        prompt = get_type_specific_prompt(ConflictType.SEMANTIC)
        assert "SEMANTIC" in prompt
        assert "behavior" in prompt.lower() or "conservative" in prompt.lower()

    def test_structural_prompt(self):
        """STRUCTURAL prompt should mention refactoring."""
        prompt = get_type_specific_prompt(ConflictType.STRUCTURAL)
        assert "STRUCTURAL" in prompt
        assert "rename" in prompt.lower() or "refactor" in prompt.lower()

    def test_unknown_prompt(self):
        """UNKNOWN prompt should be conservative."""
        prompt = get_type_specific_prompt(ConflictType.UNKNOWN)
        assert "conservative" in prompt.lower() or "careful" in prompt.lower()
