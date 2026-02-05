"""Tests for constraints module."""

import pytest
from merge_ai.core.constraints import (
    check_ast_valid,
    check_imports_preserved,
    check_functions_preserved,
    get_imports,
    get_function_signatures,
    get_constraint_prompt,
    Constraint,
)


class TestAstValidity:
    """Tests for AST validity checking."""

    def test_valid_python_code(self):
        """Valid Python code should pass."""
        code = """
def hello():
    print("Hello, world!")
"""
        result = check_ast_valid(code)
        assert result.passed
        assert result.violations == []

    def test_invalid_python_code(self):
        """Invalid Python code should fail with syntax error."""
        code = """
def hello(
    print("Missing closing parenthesis")
"""
        result = check_ast_valid(code)
        assert not result.passed
        assert len(result.violations) > 0
        assert "syntax" in result.violations[0].lower() or "Line" in result.violations[0]

    def test_empty_code(self):
        """Empty code should be valid Python."""
        code = ""
        result = check_ast_valid(code)
        assert result.passed

    def test_complex_valid_code(self):
        """Complex but valid Python should pass."""
        code = """
import os
from typing import Optional

class MyClass:
    def __init__(self, value: int) -> None:
        self.value = value

    async def async_method(self) -> str:
        return f"Value: {self.value}"

def main() -> None:
    obj = MyClass(42)
    print(obj.value)
"""
        result = check_ast_valid(code)
        assert result.passed


class TestImportExtraction:
    """Tests for import extraction."""

    def test_simple_import(self):
        """Extract simple imports."""
        code = "import os"
        imports = get_imports(code)
        assert "os" in imports

    def test_from_import(self):
        """Extract from imports."""
        code = "from typing import Optional"
        imports = get_imports(code)
        assert "typing.Optional" in imports

    def test_multiple_imports(self):
        """Extract multiple imports."""
        code = """
import os
import sys
from typing import Optional, List
from pathlib import Path
"""
        imports = get_imports(code)
        assert "os" in imports
        assert "sys" in imports
        assert "typing.Optional" in imports
        assert "typing.List" in imports
        assert "pathlib.Path" in imports


class TestImportPreservation:
    """Tests for import preservation checking."""

    def test_preserved_imports(self):
        """All imports from inputs should be allowed."""
        base = "import os"
        ours = "import sys"
        theirs = "from typing import Optional"
        resolved = """
import os
import sys
from typing import Optional
"""
        result = check_imports_preserved(resolved, base, ours, theirs)
        assert result.passed
        assert result.violations == []

    def test_new_import_violation(self):
        """New imports not in inputs should be flagged."""
        base = "import os"
        ours = "import os"
        theirs = "import os"
        resolved = """
import os
import requests  # Not in any input!
"""
        result = check_imports_preserved(resolved, base, ours, theirs)
        assert not result.passed
        assert any("requests" in v for v in result.violations)


class TestFunctionSignatures:
    """Tests for function signature extraction."""

    def test_simple_function(self):
        """Extract simple function signature."""
        code = """
def hello(name):
    print(f"Hello, {name}!")
"""
        sigs = get_function_signatures(code)
        assert "hello" in sigs
        assert sigs["hello"][0] == ["name"]

    def test_function_with_defaults(self):
        """Extract function with default arguments."""
        code = """
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")
"""
        sigs = get_function_signatures(code)
        assert "greet" in sigs
        assert sigs["greet"][0] == ["name", "greeting"]

    def test_function_with_varargs(self):
        """Extract function with *args and **kwargs."""
        code = """
def flexible(*args, **kwargs):
    pass
"""
        sigs = get_function_signatures(code)
        assert "flexible" in sigs
        assert "*args" in sigs["flexible"][0]
        assert "**kwargs" in sigs["flexible"][0]


class TestFunctionPreservation:
    """Tests for function preservation checking."""

    def test_preserved_functions(self):
        """All functions from inputs should be preserved."""
        base = """
def base_func():
    pass
"""
        ours = """
def base_func():
    pass
def ours_func():
    pass
"""
        theirs = """
def base_func():
    pass
def theirs_func():
    pass
"""
        resolved = """
def base_func():
    pass
def ours_func():
    pass
def theirs_func():
    pass
"""
        result = check_functions_preserved(resolved, base, ours, theirs)
        assert result.passed
        assert result.violations == []

    def test_missing_function_violation(self):
        """Missing function should be flagged."""
        base = """
def important_func():
    pass
"""
        ours = base
        theirs = base
        resolved = """
# Oops, forgot important_func!
def some_other_func():
    pass
"""
        result = check_functions_preserved(resolved, base, ours, theirs)
        assert not result.passed
        assert any("important_func" in v for v in result.violations)


class TestConstraintPrompts:
    """Tests for constraint prompt generation."""

    def test_ast_constraint_prompt(self):
        """AST constraint should generate appropriate prompt."""
        prompt = get_constraint_prompt([Constraint.AST])
        assert "valid Python" in prompt.lower() or "parse" in prompt.lower()

    def test_imports_constraint_prompt(self):
        """Imports constraint should generate appropriate prompt."""
        prompt = get_constraint_prompt([Constraint.IMPORTS])
        assert "import" in prompt.lower()

    def test_functions_constraint_prompt(self):
        """Functions constraint should generate appropriate prompt."""
        prompt = get_constraint_prompt([Constraint.FUNCTIONS])
        assert "function" in prompt.lower()

    def test_all_constraints_prompt(self):
        """ALL constraint should include all individual prompts."""
        prompt = get_constraint_prompt([Constraint.ALL])
        assert "python" in prompt.lower() or "parse" in prompt.lower()
        assert "import" in prompt.lower()
        assert "function" in prompt.lower()

    def test_empty_constraints(self):
        """Empty constraints should return empty string."""
        prompt = get_constraint_prompt([])
        assert prompt == ""
