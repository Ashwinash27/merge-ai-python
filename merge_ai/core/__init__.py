"""Core components for merge conflict resolution."""

from .resolver import resolve_conflict, Resolver
from .classifier import classify_conflict, ConflictType
from .validator import validate_resolution, ValidationResult
from .constraints import (
    check_ast_valid,
    check_imports_preserved,
    check_functions_preserved,
    Constraint,
)
from .errors import (
    ErrorType,
    ErrorClassification,
    classify_error,
    get_retry_prompt,
)
from .metrics import (
    count_hallucinated_imports,
    count_hallucinated_identifiers,
    semantic_match_score,
)

__all__ = [
    # Resolver
    "resolve_conflict",
    "Resolver",
    # Classifier
    "classify_conflict",
    "ConflictType",
    # Validator
    "validate_resolution",
    "ValidationResult",
    # Constraints
    "check_ast_valid",
    "check_imports_preserved",
    "check_functions_preserved",
    "Constraint",
    # Errors
    "ErrorType",
    "ErrorClassification",
    "classify_error",
    "get_retry_prompt",
    # Metrics
    "count_hallucinated_imports",
    "count_hallucinated_identifiers",
    "semantic_match_score",
]
