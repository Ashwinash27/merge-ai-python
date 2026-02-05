"""Tests for error detection and classification."""

import pytest
from merge_ai.core.errors import (
    ErrorType,
    extract_code_from_markdown,
    looks_truncated,
    classify_error,
    get_retry_prompt,
    find_hallucinated_imports,
)


class TestMarkdownExtraction:
    """Tests for extracting code from markdown."""

    def test_plain_code(self):
        """Plain code without markdown should be unchanged."""
        code = "def hello():\n    print('Hello')"
        result = extract_code_from_markdown(code)
        assert result == code

    def test_python_markdown_block(self):
        """Code in ```python block should be extracted."""
        markdown = """```python
def hello():
    print('Hello')
```"""
        result = extract_code_from_markdown(markdown)
        assert "def hello():" in result
        assert "```" not in result

    def test_generic_markdown_block(self):
        """Code in generic ``` block should be extracted."""
        markdown = """```
def hello():
    print('Hello')
```"""
        result = extract_code_from_markdown(markdown)
        assert "def hello():" in result
        assert "```" not in result

    def test_mixed_content(self):
        """Mixed content should extract code block."""
        markdown = """Here's the code:

```python
def hello():
    print('Hello')
```

That's it!"""
        result = extract_code_from_markdown(markdown)
        assert "def hello():" in result
        assert "Here's the code" not in result


class TestTruncationDetection:
    """Tests for detecting truncated output."""

    def test_complete_code(self):
        """Complete valid code should not be truncated."""
        code = """
def hello():
    print("Hello")

if __name__ == "__main__":
    hello()
"""
        assert not looks_truncated(code)

    def test_very_short_code(self):
        """Very short code might be truncated."""
        code = "def"
        assert looks_truncated(code)

    def test_trailing_ellipsis(self):
        """Code ending with ... is truncated."""
        code = """
def hello():
    print("Hello")
...
"""
        assert looks_truncated(code)

    def test_unbalanced_parens(self):
        """Unbalanced parentheses might indicate truncation."""
        code = """
def hello(
    name,
    greeting,
"""
        # This should detect truncation due to unbalanced parens
        assert looks_truncated(code)


class TestErrorClassification:
    """Tests for error classification."""

    def test_valid_code_no_error(self):
        """Valid code of reasonable length should have no error."""
        base = """
def hello():
    return "Hello, world!"

def goodbye():
    return "Goodbye!"
"""
        ours = base
        theirs = base
        resolved = base

        result = classify_error(resolved, base, ours, theirs)
        assert not result.has_error
        assert result.is_valid

    def test_empty_output_error(self):
        """Empty output should be flagged."""
        result = classify_error("", "x = 1", "x = 2", "x = 3")
        assert result.has_error
        assert result.error_type == ErrorType.EMPTY_OUTPUT

    def test_syntax_error_detection(self):
        """Syntax errors should be detected."""
        base = "x = 1"
        ours = "x = 2"
        theirs = "x = 3"
        resolved = "def foo(:"  # Invalid syntax

        result = classify_error(resolved, base, ours, theirs)
        assert result.has_error
        assert result.error_type == ErrorType.SYNTAX_ERROR


class TestHallucinatedImports:
    """Tests for detecting hallucinated imports."""

    def test_no_hallucination(self):
        """Imports from inputs should not be flagged."""
        base = "import os"
        ours = "import sys"
        theirs = "from pathlib import Path"
        resolved = """
import os
import sys
from pathlib import Path
"""
        hallucinated = find_hallucinated_imports(resolved, base, ours, theirs)
        assert len(hallucinated) == 0

    def test_common_imports_allowed(self):
        """Common stdlib imports should be allowed."""
        base = "x = 1"
        ours = "x = 2"
        theirs = "x = 3"
        resolved = """
import json
import typing
x = 1
"""
        hallucinated = find_hallucinated_imports(resolved, base, ours, theirs)
        # json and typing are common/stdlib and should be allowed
        assert len(hallucinated) == 0


class TestRetryPrompts:
    """Tests for error-specific retry prompts."""

    def test_syntax_error_prompt(self):
        """Syntax error prompt should mention fixing syntax."""
        prompt = get_retry_prompt(ErrorType.SYNTAX_ERROR, "Line 5: invalid syntax")
        assert "syntax" in prompt.lower()
        assert "Line 5" in prompt or "invalid" in prompt.lower()

    def test_truncation_prompt(self):
        """Truncation prompt should mention completeness."""
        prompt = get_retry_prompt(ErrorType.TRUNCATION, "Output incomplete")
        assert "complete" in prompt.lower() or "truncated" in prompt.lower()

    def test_hallucination_import_prompt(self):
        """Hallucination import prompt should mention removing imports."""
        prompt = get_retry_prompt(ErrorType.HALLUCINATION_IMPORT, "requests, flask")
        assert "import" in prompt.lower()
        assert "requests" in prompt or "remove" in prompt.lower()
