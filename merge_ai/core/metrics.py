"""Metrics for measuring merge resolution quality."""

import ast
import difflib
from dataclasses import dataclass
from typing import Optional

from .errors import (
    get_imports_from_code,
    get_identifiers_from_code,
    find_hallucinated_imports,
    find_hallucinated_identifiers,
)


@dataclass
class ResolutionMetrics:
    """Comprehensive metrics for a merge resolution."""

    # Syntax validity (0 or 1)
    syntax_valid: bool

    # Hallucination counts
    hallucinated_imports: int
    hallucinated_identifiers: int
    hallucinated_import_names: list[str]
    hallucinated_identifier_names: list[str]

    # Gold match score (0-1)
    gold_match_score: float

    # Cost tracking
    input_tokens: int
    output_tokens: int
    cost_usd: float

    # Attempt info
    attempts: int
    final_success: bool

    def to_dict(self) -> dict:
        return {
            "syntax_valid": self.syntax_valid,
            "hallucinated_imports": self.hallucinated_imports,
            "hallucinated_identifiers": self.hallucinated_identifiers,
            "hallucinated_import_names": self.hallucinated_import_names,
            "hallucinated_identifier_names": self.hallucinated_identifier_names,
            "gold_match_score": self.gold_match_score,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "attempts": self.attempts,
            "final_success": self.final_success,
        }


def count_hallucinated_imports(
    resolved: str, base: str, ours: str, theirs: str
) -> tuple[int, list[str]]:
    """Count imports in resolved code that don't exist in any input file.

    Args:
        resolved: The resolved/merged code
        base: Base version of the file
        ours: Our branch version
        theirs: Their branch version

    Returns:
        Tuple of (count, list of hallucinated import names)
    """
    hallucinated = find_hallucinated_imports(resolved, base, ours, theirs)
    return len(hallucinated), hallucinated


def count_hallucinated_identifiers(
    resolved: str, base: str, ours: str, theirs: str
) -> tuple[int, list[str]]:
    """Count identifiers in resolved code that don't exist in any input file.

    Args:
        resolved: The resolved/merged code
        base: Base version of the file
        ours: Our branch version
        theirs: Their branch version

    Returns:
        Tuple of (count, list of hallucinated identifier names)
    """
    hallucinated = find_hallucinated_identifiers(resolved, base, ours, theirs)
    return len(hallucinated), hallucinated


def normalize_code(code: str) -> str:
    """Normalize code for comparison by removing comments and normalizing whitespace."""
    try:
        tree = ast.parse(code)
        # Use ast.unparse for consistent formatting (Python 3.9+)
        return ast.unparse(tree)
    except SyntaxError:
        # Fallback to basic normalization
        lines = []
        for line in code.split("\n"):
            line = line.rstrip()
            # Remove comments (simple approach)
            if "#" in line:
                # Don't remove # in strings
                in_string = False
                for i, c in enumerate(line):
                    if c in "\"'" and (i == 0 or line[i - 1] != "\\"):
                        in_string = not in_string
                    elif c == "#" and not in_string:
                        line = line[:i].rstrip()
                        break
            if line.strip():
                lines.append(line)
        return "\n".join(lines)


def get_ast_structure(code: str) -> Optional[list]:
    """Extract structural elements from code's AST for comparison."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    structure = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            structure.append(("function", node.name, tuple(args)))
        elif isinstance(node, ast.AsyncFunctionDef):
            args = [arg.arg for arg in node.args.args]
            structure.append(("async_function", node.name, tuple(args)))
        elif isinstance(node, ast.ClassDef):
            bases = [
                ast.unparse(b) if hasattr(ast, "unparse") else str(b)
                for b in node.bases
            ]
            structure.append(("class", node.name, tuple(bases)))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                structure.append(("import", alias.name, alias.asname))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                structure.append(
                    ("from_import", node.module, alias.name, alias.asname)
                )

    return sorted(structure, key=str)


def semantic_match_score(resolved: str, gold: str) -> float:
    """Calculate semantic similarity between resolved code and gold standard.

    This uses a combination of:
    1. AST structure comparison (functions, classes, imports)
    2. Normalized text similarity
    3. Line-by-line matching

    Args:
        resolved: The resolved/merged code
        gold: The gold standard (expected) code

    Returns:
        Score between 0.0 and 1.0, where 1.0 is perfect match
    """
    if not resolved or not gold:
        return 0.0

    scores = []
    weights = []

    # 1. AST Structure comparison (weight: 0.4)
    resolved_structure = get_ast_structure(resolved)
    gold_structure = get_ast_structure(gold)

    if resolved_structure is not None and gold_structure is not None:
        if not gold_structure:
            # Empty gold structure
            ast_score = 1.0 if not resolved_structure else 0.5
        else:
            # Count matching structural elements
            resolved_set = set(str(s) for s in resolved_structure)
            gold_set = set(str(s) for s in gold_structure)

            intersection = len(resolved_set & gold_set)
            union = len(resolved_set | gold_set)

            ast_score = intersection / union if union > 0 else 1.0
        scores.append(ast_score)
        weights.append(0.4)

    # 2. Normalized text similarity (weight: 0.3)
    try:
        normalized_resolved = normalize_code(resolved)
        normalized_gold = normalize_code(gold)

        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, normalized_resolved, normalized_gold)
        text_score = matcher.ratio()
        scores.append(text_score)
        weights.append(0.3)
    except Exception:
        pass

    # 3. Line-by-line matching (weight: 0.3)
    resolved_lines = set(line.strip() for line in resolved.split("\n") if line.strip())
    gold_lines = set(line.strip() for line in gold.split("\n") if line.strip())

    if gold_lines:
        intersection = len(resolved_lines & gold_lines)
        union = len(resolved_lines | gold_lines)
        line_score = intersection / union if union > 0 else 1.0
        scores.append(line_score)
        weights.append(0.3)

    # Calculate weighted average
    if not scores:
        return 0.0

    total_weight = sum(weights)
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return weighted_sum / total_weight


def compute_metrics(
    resolved: str,
    base: str,
    ours: str,
    theirs: str,
    gold: Optional[str],
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    attempts: int = 1,
) -> ResolutionMetrics:
    """Compute all metrics for a resolution.

    Args:
        resolved: The resolved/merged code
        base: Base version of the file
        ours: Our branch version
        theirs: Their branch version
        gold: Gold standard (expected) code, if available
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        cost_usd: Total cost in USD
        attempts: Number of resolution attempts

    Returns:
        ResolutionMetrics with all computed metrics
    """
    # Syntax validity
    try:
        ast.parse(resolved)
        syntax_valid = True
    except SyntaxError:
        syntax_valid = False

    # Hallucination counts
    import_count, import_names = count_hallucinated_imports(
        resolved, base, ours, theirs
    )
    id_count, id_names = count_hallucinated_identifiers(resolved, base, ours, theirs)

    # Gold match score
    gold_score = semantic_match_score(resolved, gold) if gold else 0.0

    return ResolutionMetrics(
        syntax_valid=syntax_valid,
        hallucinated_imports=import_count,
        hallucinated_identifiers=id_count,
        hallucinated_import_names=import_names,
        hallucinated_identifier_names=id_names,
        gold_match_score=gold_score,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        attempts=attempts,
        final_success=syntax_valid,
    )
