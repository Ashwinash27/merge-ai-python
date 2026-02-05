"""Error taxonomy and intelligent retry system for merge resolution."""

import ast
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ErrorType(Enum):
    """Types of errors that can occur during merge resolution."""

    SYNTAX_ERROR = "syntax_error"
    HALLUCINATION_IMPORT = "hallucination_import"
    HALLUCINATION_IDENTIFIER = "hallucination_identifier"
    TRUNCATION = "truncation"
    SEMANTIC_DRIFT = "semantic_drift"
    EMPTY_OUTPUT = "empty_output"
    MARKDOWN_WRAPPED = "markdown_wrapped"


@dataclass
class ErrorClassification:
    """Result of classifying an error in LLM output."""

    error_type: Optional[ErrorType]
    details: Optional[str]
    is_valid: bool

    @property
    def has_error(self) -> bool:
        return self.error_type is not None


def extract_code_from_markdown(code: str) -> str:
    """Extract Python code from markdown code blocks if present."""
    # Pattern for ```python ... ``` or ``` ... ```
    pattern = r"```(?:python)?\s*\n?(.*?)```"
    matches = re.findall(pattern, code, re.DOTALL)
    if matches:
        return matches[0].strip()
    return code.strip()


def looks_truncated(code: str) -> bool:
    """Check if code appears to be truncated."""
    if not code or len(code.strip()) < 50:
        return True

    code = code.strip()

    # Check for obvious truncation markers
    truncation_markers = ["...", "# ...", "# continued", "# etc", "[truncated]"]
    for marker in truncation_markers:
        if code.endswith(marker):
            return True

    # If code parses as valid Python, it's likely not truncated
    try:
        ast.parse(code)
        # Valid Python - only check for very short output relative to typical code
        if len(code) > 200:
            return False
    except SyntaxError:
        pass

    # Check for unbalanced brackets/parens (only for unparseable code)
    open_parens = code.count("(") - code.count(")")
    open_brackets = code.count("[") - code.count("]")
    open_braces = code.count("{") - code.count("}")

    if open_parens > 2 or open_brackets > 2 or open_braces > 2:
        return True

    return False


def get_imports_from_code(code: str) -> set[str]:
    """Extract all imported module/name combinations from code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def get_identifiers_from_code(code: str) -> set[str]:
    """Extract all identifiers (variable names, function names, etc.) from code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.FunctionDef):
            identifiers.add(node.name)
        elif isinstance(node, ast.ClassDef):
            identifiers.add(node.name)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
    return identifiers


def find_hallucinated_imports(
    resolved: str, base: str, ours: str, theirs: str
) -> list[str]:
    """Find imports in resolved code that don't exist in any input file."""
    allowed = get_imports_from_code(base) | get_imports_from_code(ours) | get_imports_from_code(theirs)
    actual = get_imports_from_code(resolved)

    # Also allow standard library and common imports
    common_imports = {
        "os", "sys", "re", "json", "typing", "collections", "itertools",
        "functools", "pathlib", "datetime", "time", "logging", "copy",
        "abc", "enum", "dataclasses", "ast", "inspect", "warnings",
    }
    allowed = allowed | common_imports

    return list(actual - allowed)


def find_hallucinated_identifiers(
    resolved: str, base: str, ours: str, theirs: str
) -> list[str]:
    """Find identifiers in resolved code that don't exist in any input file."""
    allowed = (
        get_identifiers_from_code(base)
        | get_identifiers_from_code(ours)
        | get_identifiers_from_code(theirs)
    )

    # Add Python builtins
    builtins = {
        "True", "False", "None", "print", "len", "range", "str", "int",
        "float", "list", "dict", "set", "tuple", "bool", "type", "isinstance",
        "hasattr", "getattr", "setattr", "open", "file", "input", "output",
        "self", "cls", "super", "property", "staticmethod", "classmethod",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "AttributeError", "RuntimeError", "NotImplementedError", "StopIteration",
        "zip", "map", "filter", "sorted", "reversed", "enumerate", "all", "any",
        "min", "max", "sum", "abs", "round", "format", "repr", "id", "hash",
        "callable", "iter", "next", "slice", "object", "bytes", "bytearray",
        "memoryview", "complex", "frozenset", "ord", "chr", "hex", "oct", "bin",
        "pow", "divmod", "exec", "eval", "compile", "globals", "locals", "vars",
        "dir", "help", "breakpoint", "ascii", "exit", "quit", "__name__",
        "__doc__", "__file__", "__package__", "__spec__", "__annotations__",
        "__dict__", "__class__", "__init__", "__new__", "__del__", "__repr__",
        "__str__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
        "__hash__", "__bool__", "__call__", "__len__", "__iter__", "__next__",
        "__getitem__", "__setitem__", "__delitem__", "__contains__", "__add__",
        "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__",
        "__enter__", "__exit__", "__await__", "__aiter__", "__anext__",
    }
    allowed = allowed | builtins

    actual = get_identifiers_from_code(resolved)
    hallucinated = actual - allowed

    # Filter out likely false positives (single letter vars, common names)
    common_names = {"i", "j", "k", "n", "x", "y", "z", "f", "e", "v", "key", "value", "item", "result", "data", "args", "kwargs", "_"}
    hallucinated = hallucinated - common_names

    return list(hallucinated)


def classify_error(
    code: str, base: str, ours: str, theirs: str
) -> ErrorClassification:
    """Classify the type of error in LLM output.

    Returns:
        ErrorClassification with error_type, details, and is_valid flag
    """
    if not code or not code.strip():
        return ErrorClassification(
            error_type=ErrorType.EMPTY_OUTPUT,
            details="Output is empty",
            is_valid=False,
        )

    # Check for markdown wrapping
    original_code = code
    if "```" in code:
        code = extract_code_from_markdown(code)
        if code != original_code.strip():
            # Code was wrapped in markdown - extract and continue checking
            pass

    # Check syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        return ErrorClassification(
            error_type=ErrorType.SYNTAX_ERROR,
            details=f"Line {e.lineno}: {e.msg}",
            is_valid=False,
        )

    # Check truncation
    if looks_truncated(code):
        return ErrorClassification(
            error_type=ErrorType.TRUNCATION,
            details="Output appears incomplete or truncated",
            is_valid=False,
        )

    # Check hallucinated imports
    invalid_imports = find_hallucinated_imports(code, base, ours, theirs)
    if invalid_imports:
        return ErrorClassification(
            error_type=ErrorType.HALLUCINATION_IMPORT,
            details=f"Invalid imports: {', '.join(invalid_imports)}",
            is_valid=False,
        )

    # Check hallucinated identifiers (less strict - only flag if many)
    invalid_ids = find_hallucinated_identifiers(code, base, ours, theirs)
    if len(invalid_ids) > 5:  # Only flag if many new identifiers
        return ErrorClassification(
            error_type=ErrorType.HALLUCINATION_IDENTIFIER,
            details=f"Potentially hallucinated identifiers: {', '.join(list(invalid_ids)[:10])}",
            is_valid=False,
        )

    # No errors detected
    return ErrorClassification(
        error_type=None,
        details=None,
        is_valid=True,
    )


def get_retry_prompt(error_type: ErrorType, details: str) -> str:
    """Generate an error-specific retry instruction."""
    prompts = {
        ErrorType.SYNTAX_ERROR: f"""CRITICAL: Your previous output had a syntax error: {details}

Fix the syntax error and output valid Python code. Make sure all brackets, parentheses, and quotes are properly balanced.""",
        ErrorType.TRUNCATION: """CRITICAL: Your previous output was incomplete or truncated.

Output the COMPLETE merged file from start to finish. Do not abbreviate or skip any sections.""",
        ErrorType.HALLUCINATION_IMPORT: f"""CRITICAL: Your previous output contained invalid imports that don't exist in the input files: {details}

Remove these imports and only use imports that appear in BASE, OURS, or THEIRS.""",
        ErrorType.HALLUCINATION_IDENTIFIER: f"""CRITICAL: Your previous output introduced new variables or functions not present in the input files: {details}

Use ONLY the variables, functions, and class names that exist in BASE, OURS, or THEIRS.""",
        ErrorType.EMPTY_OUTPUT: """CRITICAL: Your previous output was empty.

Provide the complete merged Python file.""",
        ErrorType.MARKDOWN_WRAPPED: """CRITICAL: Do not wrap your output in markdown code blocks.

Output ONLY the raw Python code, with no ``` markers or language tags.""",
        ErrorType.SEMANTIC_DRIFT: """CRITICAL: Your resolution changed the code behavior too much.

Be more conservative. Preserve the original behavior from the input files as much as possible.""",
    }
    return prompts.get(error_type, f"Fix the error and try again: {details}")


@dataclass
class RetryResult:
    """Result of a resolution attempt with retry information."""

    code: str
    success: bool
    attempts: int
    errors_encountered: list[dict]
    final_error: Optional[ErrorType] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "attempts": self.attempts,
            "errors_encountered": self.errors_encountered,
            "final_error": self.final_error.value if self.final_error else None,
        }
