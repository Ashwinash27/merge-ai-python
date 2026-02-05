"""Comparison utilities for evaluating merge resolutions against gold standard."""

import ast
import difflib
from dataclasses import dataclass
from typing import Optional


@dataclass
class ComparisonResult:
    """Result of comparing a resolution to gold standard."""

    # Scores (0-1)
    overall_score: float
    ast_structure_score: float
    text_similarity_score: float
    line_match_score: float

    # Details
    missing_functions: list[str]
    extra_functions: list[str]
    missing_imports: list[str]
    extra_imports: list[str]

    # Diff
    diff_lines: list[str]

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "ast_structure_score": self.ast_structure_score,
            "text_similarity_score": self.text_similarity_score,
            "line_match_score": self.line_match_score,
            "missing_functions": self.missing_functions,
            "extra_functions": self.extra_functions,
            "missing_imports": self.missing_imports,
            "extra_imports": self.extra_imports,
            "diff_lines": self.diff_lines[:50],  # Limit for readability
        }


def get_functions(code: str) -> set[str]:
    """Extract function names from code."""
    try:
        tree = ast.parse(code)
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    except SyntaxError:
        return set()


def get_classes(code: str) -> set[str]:
    """Extract class names from code."""
    try:
        tree = ast.parse(code)
        return {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }
    except SyntaxError:
        return set()


def get_imports(code: str) -> set[str]:
    """Extract import names from code."""
    try:
        tree = ast.parse(code)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        return imports
    except SyntaxError:
        return set()


def normalize_code(code: str) -> str:
    """Normalize code for comparison."""
    try:
        tree = ast.parse(code)
        return ast.unparse(tree)
    except SyntaxError:
        # Fallback: basic normalization
        lines = []
        for line in code.split("\n"):
            stripped = line.rstrip()
            if stripped and not stripped.lstrip().startswith("#"):
                lines.append(stripped)
        return "\n".join(lines)


def compare_to_gold(resolved: str, gold: str) -> ComparisonResult:
    """Compare a resolution to the gold standard.

    Args:
        resolved: The resolved/merged code
        gold: The gold standard (expected) code

    Returns:
        ComparisonResult with detailed comparison metrics
    """
    # Get structural elements
    resolved_funcs = get_functions(resolved)
    gold_funcs = get_functions(gold)
    resolved_imports = get_imports(resolved)
    gold_imports = get_imports(gold)

    missing_functions = list(gold_funcs - resolved_funcs)
    extra_functions = list(resolved_funcs - gold_funcs)
    missing_imports = list(gold_imports - resolved_imports)
    extra_imports = list(resolved_imports - gold_imports)

    # AST structure score
    total_expected = len(gold_funcs) + len(gold_imports)
    if total_expected > 0:
        matching = (
            len(gold_funcs & resolved_funcs) + len(gold_imports & resolved_imports)
        )
        total_actual = len(resolved_funcs) + len(resolved_imports)
        precision = matching / total_actual if total_actual > 0 else 0
        recall = matching / total_expected
        ast_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        ast_score = 1.0 if not resolved_funcs and not resolved_imports else 0.5

    # Text similarity score
    try:
        normalized_resolved = normalize_code(resolved)
        normalized_gold = normalize_code(gold)
        matcher = difflib.SequenceMatcher(None, normalized_resolved, normalized_gold)
        text_score = matcher.ratio()
    except Exception:
        text_score = 0.0

    # Line match score
    resolved_lines = set(line.strip() for line in resolved.split("\n") if line.strip())
    gold_lines = set(line.strip() for line in gold.split("\n") if line.strip())

    if gold_lines:
        intersection = len(resolved_lines & gold_lines)
        union = len(resolved_lines | gold_lines)
        line_score = intersection / union if union > 0 else 0
    else:
        line_score = 1.0 if not resolved_lines else 0.0

    # Generate diff
    diff = list(
        difflib.unified_diff(
            gold.splitlines(keepends=True),
            resolved.splitlines(keepends=True),
            fromfile="gold.py",
            tofile="resolved.py",
            lineterm="",
        )
    )

    # Overall score (weighted average)
    overall_score = (ast_score * 0.4 + text_score * 0.3 + line_score * 0.3)

    return ComparisonResult(
        overall_score=overall_score,
        ast_structure_score=ast_score,
        text_similarity_score=text_score,
        line_match_score=line_score,
        missing_functions=missing_functions,
        extra_functions=extra_functions,
        missing_imports=missing_imports,
        extra_imports=extra_imports,
        diff_lines=diff,
    )


def semantic_compare(code_a: str, code_b: str) -> float:
    """Calculate semantic similarity between two code snippets.

    This compares the structure and content of two Python files,
    ignoring formatting and comments.

    Args:
        code_a: First code snippet
        code_b: Second code snippet

    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Quick check for exact match
    if code_a.strip() == code_b.strip():
        return 1.0

    try:
        # Parse both
        tree_a = ast.parse(code_a)
        tree_b = ast.parse(code_b)

        # Compare unparsed (normalized) versions
        normalized_a = ast.unparse(tree_a)
        normalized_b = ast.unparse(tree_b)

        if normalized_a == normalized_b:
            return 1.0

        # Use sequence matcher for similarity
        matcher = difflib.SequenceMatcher(None, normalized_a, normalized_b)
        return matcher.ratio()

    except SyntaxError:
        # Fall back to text comparison
        matcher = difflib.SequenceMatcher(None, code_a, code_b)
        return matcher.ratio()


def format_diff(resolved: str, gold: str, context: int = 3) -> str:
    """Generate a readable diff between resolved and gold.

    Args:
        resolved: The resolved code
        gold: The gold standard code
        context: Number of context lines

    Returns:
        Formatted diff string
    """
    diff = difflib.unified_diff(
        gold.splitlines(keepends=True),
        resolved.splitlines(keepends=True),
        fromfile="gold.py",
        tofile="resolved.py",
        n=context,
    )
    return "".join(diff)


def partial_credit_score(
    resolved: str, gold: str, base: str, ours: str, theirs: str
) -> float:
    """Calculate partial credit score for a resolution.

    This gives credit for:
    - Preserving changes from ours
    - Preserving changes from theirs
    - Maintaining valid syntax
    - Matching structural elements

    Args:
        resolved: The resolved code
        gold: The gold standard
        base: Base version
        ours: Our branch version
        theirs: Their branch version

    Returns:
        Score between 0.0 and 1.0
    """
    scores = []

    # 1. Syntax validity (0.2 weight)
    try:
        ast.parse(resolved)
        scores.append((1.0, 0.2))
    except SyntaxError:
        scores.append((0.0, 0.2))

    # 2. Gold match (0.4 weight)
    comparison = compare_to_gold(resolved, gold)
    scores.append((comparison.overall_score, 0.4))

    # 3. Ours preservation (0.2 weight)
    ours_funcs = get_functions(ours) - get_functions(base)
    resolved_funcs = get_functions(resolved)
    if ours_funcs:
        ours_preserved = len(ours_funcs & resolved_funcs) / len(ours_funcs)
    else:
        ours_preserved = 1.0
    scores.append((ours_preserved, 0.2))

    # 4. Theirs preservation (0.2 weight)
    theirs_funcs = get_functions(theirs) - get_functions(base)
    if theirs_funcs:
        theirs_preserved = len(theirs_funcs & resolved_funcs) / len(theirs_funcs)
    else:
        theirs_preserved = 1.0
    scores.append((theirs_preserved, 0.2))

    # Weighted average
    total_score = sum(score * weight for score, weight in scores)
    return total_score
