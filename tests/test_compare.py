"""Tests for comparison utilities."""

import pytest
from merge_ai.evaluation.compare import (
    compare_to_gold,
    semantic_compare,
    format_diff,
    partial_credit_score,
    get_functions,
    get_imports,
    normalize_code,
)


class TestCompareToGold:
    """Tests for gold standard comparison."""

    def test_identical_match(self):
        """Identical code should have perfect score."""
        code = """
def hello():
    print("Hello, world!")
"""
        result = compare_to_gold(code, code)
        assert result.overall_score == 1.0
        assert result.missing_functions == []
        assert result.extra_functions == []

    def test_missing_function(self):
        """Missing function should be detected."""
        gold = """
def hello():
    pass

def goodbye():
    pass
"""
        resolved = """
def hello():
    pass
"""
        result = compare_to_gold(resolved, gold)
        assert "goodbye" in result.missing_functions
        assert result.overall_score < 1.0

    def test_extra_function(self):
        """Extra function should be detected."""
        gold = """
def hello():
    pass
"""
        resolved = """
def hello():
    pass

def extra():
    pass
"""
        result = compare_to_gold(resolved, gold)
        assert "extra" in result.extra_functions


class TestSemanticCompare:
    """Tests for semantic comparison."""

    def test_exact_match(self):
        """Exact match should return 1.0."""
        code = "x = 1"
        assert semantic_compare(code, code) == 1.0

    def test_whitespace_differences(self):
        """Whitespace differences should still yield high score."""
        code1 = "x = 1\ny = 2"
        code2 = "x=1\ny=2"
        # After AST normalization, these should be equivalent
        score = semantic_compare(code1, code2)
        assert score > 0.9

    def test_completely_different(self):
        """Completely different code should have low score."""
        code1 = "x = 1"
        code2 = "class Foo: pass"
        score = semantic_compare(code1, code2)
        assert score < 0.5


class TestDiffFormatting:
    """Tests for diff formatting."""

    def test_no_diff_for_identical(self):
        """Identical code should produce minimal/no diff."""
        code = "x = 1"
        diff = format_diff(code, code)
        # Diff should be empty or just headers
        assert "+x = 1" not in diff
        assert "-x = 1" not in diff

    def test_diff_shows_changes(self):
        """Diff should show actual changes."""
        gold = "x = 1"
        resolved = "x = 2"
        diff = format_diff(resolved, gold)
        assert "-x = 1" in diff or "+x = 2" in diff


class TestPartialCredit:
    """Tests for partial credit scoring."""

    def test_perfect_match(self):
        """Perfect match should get full credit."""
        base = "x = 0"
        ours = "x = 1"
        theirs = "x = 2"
        gold = "x = 1"
        resolved = "x = 1"

        score = partial_credit_score(resolved, gold, base, ours, theirs)
        assert score > 0.9

    def test_syntax_error_low_score(self):
        """Syntax error should result in lower score than perfect match."""
        base = "x = 0"
        ours = "x = 1"
        theirs = "x = 2"
        gold = "x = 1"
        resolved = "x = ("  # Syntax error

        score = partial_credit_score(resolved, gold, base, ours, theirs)
        # Syntax error gets 0 for syntax validity component (0.2 weight)
        # but may still get partial credit for other components
        assert score < 1.0  # Should not be perfect
        # The syntax component should definitely be penalized
        assert score <= 0.8  # At most 80% (since 20% is syntax)


class TestHelpers:
    """Tests for helper functions."""

    def test_get_functions(self):
        """Should extract function names."""
        code = """
def foo():
    pass

def bar():
    pass
"""
        funcs = get_functions(code)
        assert "foo" in funcs
        assert "bar" in funcs

    def test_get_imports(self):
        """Should extract import names."""
        code = """
import os
from pathlib import Path
"""
        imports = get_imports(code)
        assert "os" in imports
        assert "pathlib" in imports

    def test_normalize_code(self):
        """Should normalize code to consistent format."""
        code = """
def foo():
    x=1
    y=2
"""
        normalized = normalize_code(code)
        # Should have consistent formatting
        assert "def foo():" in normalized
