"""Structural constraints for merge resolution (H3 experiment)."""

import ast
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Constraint(Enum):
    """Types of constraints that can be applied to merge resolution."""

    AST = "ast"  # Must be valid Python
    IMPORTS = "imports"  # No new imports
    FUNCTIONS = "functions"  # Preserve function signatures
    ALL = "all"  # All constraints


@dataclass
class ConstraintResult:
    """Result of checking a constraint."""

    constraint: Constraint
    passed: bool
    violations: list[str]

    def __bool__(self) -> bool:
        return self.passed


def check_ast_valid(code: str) -> ConstraintResult:
    """Check if code is valid Python that parses without errors.

    Args:
        code: Python source code to validate

    Returns:
        ConstraintResult with passed=True if valid, violations list if not
    """
    try:
        ast.parse(code)
        return ConstraintResult(
            constraint=Constraint.AST, passed=True, violations=[]
        )
    except SyntaxError as e:
        return ConstraintResult(
            constraint=Constraint.AST,
            passed=False,
            violations=[f"Line {e.lineno}: {e.msg}"],
        )


def get_imports(code: str) -> set[str]:
    """Extract all imports from Python code.

    Returns set of import strings in format:
    - "module" for `import module`
    - "module.submodule" for `import module.submodule`
    - "module.name" for `from module import name`
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if module:
                    imports.add(f"{module}.{alias.name}")
                else:
                    imports.add(alias.name)
    return imports


def check_imports_preserved(
    resolved: str, base: str, ours: str, theirs: str
) -> ConstraintResult:
    """Check that resolved code only uses imports from input files.

    Args:
        resolved: The resolved/merged code
        base: Base version of the file
        ours: Our branch version
        theirs: Their branch version

    Returns:
        ConstraintResult with violations list of any new imports
    """
    allowed = get_imports(base) | get_imports(ours) | get_imports(theirs)

    # Also get base module names (without submodules) for flexibility
    allowed_base = set()
    for imp in allowed:
        allowed_base.add(imp.split(".")[0])

    try:
        actual = get_imports(resolved)
    except SyntaxError:
        return ConstraintResult(
            constraint=Constraint.IMPORTS,
            passed=False,
            violations=["Cannot parse resolved code to check imports"],
        )

    violations = []
    for imp in actual:
        base_module = imp.split(".")[0]
        # Check if import or its base module is allowed
        if imp not in allowed and base_module not in allowed_base:
            violations.append(f"New import: {imp}")

    return ConstraintResult(
        constraint=Constraint.IMPORTS,
        passed=len(violations) == 0,
        violations=violations,
    )


def get_function_signatures(code: str) -> dict[str, tuple[list[str], bool]]:
    """Extract function signatures from Python code.

    Returns:
        Dict mapping function name to (arg_names, has_return_annotation)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    signatures = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = []
            # Regular args
            for arg in node.args.args:
                args.append(arg.arg)
            # *args
            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")
            # Keyword-only args
            for arg in node.args.kwonlyargs:
                args.append(arg.arg)
            # **kwargs
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")

            has_return = node.returns is not None
            signatures[node.name] = (args, has_return)

    return signatures


def check_functions_preserved(
    resolved: str, base: str, ours: str, theirs: str
) -> ConstraintResult:
    """Check that resolved code preserves all function names and signatures.

    Args:
        resolved: The resolved/merged code
        base: Base version of the file
        ours: Our branch version
        theirs: Their branch version

    Returns:
        ConstraintResult with violations for missing or changed functions
    """
    # Collect expected functions from all inputs
    expected = {}
    for code in [base, ours, theirs]:
        sigs = get_function_signatures(code)
        for name, sig in sigs.items():
            if name not in expected:
                expected[name] = sig
            # If function exists in multiple versions, keep any signature
            # (we just want to ensure the function exists with compatible sig)

    try:
        actual = get_function_signatures(resolved)
    except SyntaxError:
        return ConstraintResult(
            constraint=Constraint.FUNCTIONS,
            passed=False,
            violations=["Cannot parse resolved code to check functions"],
        )

    violations = []
    for name, (expected_args, _) in expected.items():
        if name not in actual:
            violations.append(f"Missing function: {name}")
        else:
            actual_args, _ = actual[name]
            # Check if args match (allowing for some flexibility)
            # We compare arg names, ignoring self/cls
            expected_filtered = [a for a in expected_args if a not in ("self", "cls")]
            actual_filtered = [a for a in actual_args if a not in ("self", "cls")]

            if expected_filtered != actual_filtered:
                violations.append(
                    f"Changed signature: {name}({', '.join(expected_args)}) -> "
                    f"{name}({', '.join(actual_args)})"
                )

    return ConstraintResult(
        constraint=Constraint.FUNCTIONS,
        passed=len(violations) == 0,
        violations=violations,
    )


def check_all_constraints(
    resolved: str, base: str, ours: str, theirs: str
) -> list[ConstraintResult]:
    """Check all constraints against resolved code.

    Returns:
        List of ConstraintResult for each constraint type
    """
    return [
        check_ast_valid(resolved),
        check_imports_preserved(resolved, base, ours, theirs),
        check_functions_preserved(resolved, base, ours, theirs),
    ]


def get_constraint_prompt(constraints: list[Constraint]) -> str:
    """Generate prompt text for the specified constraints.

    Args:
        constraints: List of constraints to include in prompt

    Returns:
        Prompt text describing the constraints
    """
    if not constraints:
        return ""

    if Constraint.ALL in constraints:
        constraints = [Constraint.AST, Constraint.IMPORTS, Constraint.FUNCTIONS]

    prompts = []

    if Constraint.AST in constraints:
        prompts.append(
            """CONSTRAINT - VALID PYTHON: Your output MUST be valid Python code that parses without errors.
Do not include markdown, explanations, or code fences. Output ONLY the merged Python file."""
        )

    if Constraint.IMPORTS in constraints:
        prompts.append(
            """CONSTRAINT - IMPORTS: You may ONLY use imports that appear in BASE, OURS, or THEIRS.
Do not add any new imports that don't exist in the input files."""
        )

    if Constraint.FUNCTIONS in constraints:
        prompts.append(
            """CONSTRAINT - FUNCTIONS: You must preserve all function names and signatures.
Do not rename functions or change their parameters. Only modify function bodies if necessary for the merge."""
        )

    return "\n\n".join(prompts)
