"""Tests for metrics computation."""

import pytest
from merge_ai.core.metrics import (
    count_hallucinated_imports,
    count_hallucinated_identifiers,
    semantic_match_score,
    normalize_code,
    get_ast_structure,
    compute_metrics,
)


class TestHallucinationCounts:
    """Tests for hallucination counting."""

    def test_no_hallucinated_imports(self):
        """Code using only input imports should have zero count."""
        base = "import os"
        ours = "import sys"
        theirs = "from pathlib import Path"
        resolved = """
import os
import sys
from pathlib import Path
"""
        count, names = count_hallucinated_imports(resolved, base, ours, theirs)
        assert count == 0
        assert names == []

    def test_no_hallucinated_identifiers(self):
        """Code using only input identifiers should have low count."""
        base = "x = 1"
        ours = "y = 2"
        theirs = "z = 3"
        resolved = """
x = 1
y = 2
z = 3
"""
        count, names = count_hallucinated_identifiers(resolved, base, ours, theirs)
        assert count == 0


class TestSemanticMatch:
    """Tests for semantic matching."""

    def test_identical_code_score_one(self):
        """Identical code should have score of 1.0."""
        code = """
def hello():
    print("Hello")
"""
        score = semantic_match_score(code, code)
        assert score == 1.0

    def test_different_code_lower_score(self):
        """Completely different code should have low score."""
        code1 = """
def hello():
    print("Hello")
"""
        code2 = """
class Goodbye:
    def farewell(self):
        return "Bye"
"""
        score = semantic_match_score(code1, code2)
        assert score < 0.5

    def test_similar_code_high_score(self):
        """Similar code should have higher score."""
        code1 = """
def hello(name):
    print(f"Hello, {name}!")
"""
        code2 = """
def hello(name):
    print(f"Hi, {name}!")
"""
        score = semantic_match_score(code1, code2)
        assert score > 0.7

    def test_empty_code_zero_score(self):
        """Empty code comparison should return zero."""
        score = semantic_match_score("", "def foo(): pass")
        assert score == 0.0


class TestCodeNormalization:
    """Tests for code normalization."""

    def test_whitespace_normalization(self):
        """Code with different whitespace should normalize similarly."""
        code1 = """
def hello():
    x = 1
    y = 2
"""
        code2 = """def hello():
    x=1
    y=2"""

        norm1 = normalize_code(code1)
        norm2 = normalize_code(code2)
        # After AST normalization, they should be identical
        assert norm1.strip() == norm2.strip()


class TestAstStructure:
    """Tests for AST structure extraction."""

    def test_function_extraction(self):
        """Functions should be extracted from AST."""
        code = """
def hello(name):
    pass

def goodbye():
    pass
"""
        structure = get_ast_structure(code)
        assert structure is not None
        function_names = [s[1] for s in structure if s[0] == "function"]
        assert "hello" in function_names
        assert "goodbye" in function_names

    def test_class_extraction(self):
        """Classes should be extracted from AST."""
        code = """
class MyClass:
    pass
"""
        structure = get_ast_structure(code)
        assert structure is not None
        class_names = [s[1] for s in structure if s[0] == "class"]
        assert "MyClass" in class_names

    def test_import_extraction(self):
        """Imports should be extracted from AST."""
        code = """
import os
from pathlib import Path
"""
        structure = get_ast_structure(code)
        assert structure is not None
        imports = [s for s in structure if s[0] in ("import", "from_import")]
        assert len(imports) >= 2


class TestComputeMetrics:
    """Tests for comprehensive metrics computation."""

    def test_valid_code_metrics(self):
        """Valid code should have syntax_valid=True."""
        resolved = "x = 1"
        base = "x = 0"
        ours = "x = 1"
        theirs = "x = 2"
        gold = "x = 1"

        metrics = compute_metrics(resolved, base, ours, theirs, gold)
        assert metrics.syntax_valid is True
        assert metrics.gold_match_score > 0.9  # Should be high match

    def test_invalid_code_metrics(self):
        """Invalid code should have syntax_valid=False."""
        resolved = "def foo(:"  # Invalid syntax
        base = "x = 0"
        ours = "x = 1"
        theirs = "x = 2"
        gold = "x = 1"

        metrics = compute_metrics(resolved, base, ours, theirs, gold)
        assert metrics.syntax_valid is False

    def test_metrics_without_gold(self):
        """Metrics should work without gold standard."""
        resolved = "x = 1"
        base = "x = 0"
        ours = "x = 1"
        theirs = "x = 2"

        metrics = compute_metrics(resolved, base, ours, theirs, gold=None)
        assert metrics.syntax_valid is True
        assert metrics.gold_match_score == 0.0  # No gold to compare against
