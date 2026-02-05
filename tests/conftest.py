"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_base_code():
    """Sample base Python code for testing."""
    return '''"""Sample module."""

import os

def hello(name):
    """Greet someone."""
    return f"Hello, {name}!"

def goodbye(name):
    """Say goodbye to someone."""
    return f"Goodbye, {name}!"
'''


@pytest.fixture
def sample_ours_code():
    """Sample 'ours' branch code for testing."""
    return '''"""Sample module."""

import os
import sys  # New import

def hello(name):
    """Greet someone warmly."""  # Updated docstring
    return f"Hello, {name}! Welcome!"  # Updated message

def goodbye(name):
    """Say goodbye to someone."""
    return f"Goodbye, {name}!"
'''


@pytest.fixture
def sample_theirs_code():
    """Sample 'theirs' branch code for testing."""
    return '''"""Sample module."""

import os

def hello(name):
    """Greet someone."""
    return f"Hello, {name}!"

def goodbye(name):
    """Say goodbye to someone."""
    return f"Goodbye, {name}! See you later!"  # Updated message

def farewell(name):  # New function
    """Final farewell."""
    return f"Farewell, {name}!"
'''


@pytest.fixture
def sample_gold_code():
    """Sample gold (expected merge) code for testing."""
    return '''"""Sample module."""

import os
import sys  # From ours

def hello(name):
    """Greet someone warmly."""  # From ours
    return f"Hello, {name}! Welcome!"  # From ours

def goodbye(name):
    """Say goodbye to someone."""
    return f"Goodbye, {name}! See you later!"  # From theirs

def farewell(name):  # From theirs
    """Final farewell."""
    return f"Farewell, {name}!"
'''


@pytest.fixture
def conflict_dir(tmp_path, sample_base_code, sample_ours_code, sample_theirs_code, sample_gold_code):
    """Create a temporary conflict directory with test files."""
    conflict_path = tmp_path / "test_conflict"
    conflict_path.mkdir()

    (conflict_path / "base.py").write_text(sample_base_code)
    (conflict_path / "ours.py").write_text(sample_ours_code)
    (conflict_path / "theirs.py").write_text(sample_theirs_code)
    (conflict_path / "gold.py").write_text(sample_gold_code)

    return conflict_path


@pytest.fixture
def simple_codes():
    """Simple code snippets for quick tests."""
    return {
        "base": "x = 0",
        "ours": "x = 1",
        "theirs": "x = 2",
        "gold": "x = 1",
    }
