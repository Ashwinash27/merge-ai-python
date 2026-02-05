"""Cross-model validation for H2 experiment."""

import json
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from openai import OpenAI
from anthropic import Anthropic

from ..config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    ModelConfig,
    CLAUDE_SONNET,
    OPENROUTER_HAIKU,
)
from .constraints import check_ast_valid, check_imports_preserved, check_functions_preserved


class ValidationVerdict(Enum):
    """Verdict from validation."""

    APPROVE = "approve"
    REJECT = "reject"
    NEEDS_REVISION = "needs_revision"


@dataclass
class ValidationResult:
    """Result of cross-model validation."""

    verdict: ValidationVerdict
    issues: list[str]
    suggestions: list[str]
    confidence: float
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def passed(self) -> bool:
        return self.verdict == ValidationVerdict.APPROVE

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


VALIDATION_SYSTEM_PROMPT = """You are an expert code reviewer validating a merge conflict resolution. Your task is to check if the resolved code correctly merges the two divergent versions.

Review the resolution for:
1. CORRECTNESS: Does it properly combine changes from both branches?
2. COMPLETENESS: Are all changes from both versions preserved (unless intentionally excluded)?
3. SYNTAX: Is the code valid Python?
4. LOGIC: Does the merged code make logical sense?
5. NO HALLUCINATIONS: Does it avoid introducing new code/imports not present in the inputs?

Respond with a JSON object:
{
    "verdict": "APPROVE" | "REJECT" | "NEEDS_REVISION",
    "issues": ["list of specific problems found"],
    "suggestions": ["list of specific fixes if NEEDS_REVISION"],
    "confidence": 0.0-1.0
}

Only output the JSON, no other text."""


VALIDATION_USER_PROMPT = """Validate this merge conflict resolution:

=== BASE (original) ===
{base}

=== OURS (our changes) ===
{ours}

=== THEIRS (their changes) ===
{theirs}

=== RESOLUTION (proposed merge) ===
{resolved}

Validate this resolution and provide your analysis as JSON."""


def validate_resolution(
    resolved: str,
    base: str,
    ours: str,
    theirs: str,
    model: Optional[ModelConfig] = None,
) -> ValidationResult:
    """Validate a merge resolution using a secondary model.

    Args:
        resolved: The proposed resolution
        base: Base version of the file
        ours: Our branch version
        theirs: Their branch version
        model: Model to use for validation (defaults to OpenRouter Haiku)

    Returns:
        ValidationResult with verdict, issues, and suggestions
    """
    model = model or OPENROUTER_HAIKU

    # First, do programmatic checks
    programmatic_issues = []

    ast_check = check_ast_valid(resolved)
    if not ast_check.passed:
        programmatic_issues.extend(ast_check.violations)

    import_check = check_imports_preserved(resolved, base, ours, theirs)
    if not import_check.passed:
        programmatic_issues.extend(import_check.violations)

    function_check = check_functions_preserved(resolved, base, ours, theirs)
    if not function_check.passed:
        programmatic_issues.extend(function_check.violations)

    # If programmatic checks fail, return early
    if programmatic_issues:
        return ValidationResult(
            verdict=ValidationVerdict.REJECT,
            issues=programmatic_issues,
            suggestions=["Fix the above issues"],
            confidence=1.0,
            input_tokens=0,
            output_tokens=0,
        )

    # Use LLM for semantic validation
    user_prompt = VALIDATION_USER_PROMPT.format(
        base=base[:8000],
        ours=ours[:8000],
        theirs=theirs[:8000],
        resolved=resolved[:8000],
    )

    if model.provider in ("openai", "openrouter"):
        return _validate_openai(user_prompt, model)
    else:
        return _validate_anthropic(user_prompt, model)


def _validate_openai(user_prompt: str, model: ModelConfig) -> ValidationResult:
    """Validate using OpenAI-compatible API (OpenAI or OpenRouter)."""
    if model.provider == "openrouter":
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=model.model,
        temperature=model.temperature,
        messages=[
            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0

    return _parse_validation(content, input_tokens, output_tokens)


def _validate_anthropic(user_prompt: str, model: ModelConfig) -> ValidationResult:
    """Validate using Anthropic API."""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=model.model,
        max_tokens=1024,
        system=VALIDATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    content = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    return _parse_validation(content, input_tokens, output_tokens)


def _parse_validation(
    content: str, input_tokens: int, output_tokens: int
) -> ValidationResult:
    """Parse LLM response into ValidationResult."""
    try:
        # Handle potential markdown wrapping
        if "```" in content:
            import re

            match = re.search(r"```(?:json)?\s*\n?(.*?)```", content, re.DOTALL)
            if match:
                content = match.group(1)

        data = json.loads(content)

        verdict_str = data.get("verdict", "REJECT").upper()
        verdict_map = {
            "APPROVE": ValidationVerdict.APPROVE,
            "REJECT": ValidationVerdict.REJECT,
            "NEEDS_REVISION": ValidationVerdict.NEEDS_REVISION,
        }
        verdict = verdict_map.get(verdict_str, ValidationVerdict.REJECT)

        return ValidationResult(
            verdict=verdict,
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            confidence=float(data.get("confidence", 0.5)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return ValidationResult(
            verdict=ValidationVerdict.REJECT,
            issues=[f"Failed to parse validation: {e}"],
            suggestions=[],
            confidence=0.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def ensemble_select(
    resolution_a: str,
    resolution_b: str,
    base: str,
    ours: str,
    theirs: str,
) -> tuple[str, str]:
    """Select the better resolution from two candidates using heuristics.

    Args:
        resolution_a: First resolution candidate
        resolution_b: Second resolution candidate
        base, ours, theirs: Input files for validation

    Returns:
        Tuple of (selected_resolution, reason)
    """
    # Check syntax validity
    a_valid = check_ast_valid(resolution_a).passed
    b_valid = check_ast_valid(resolution_b).passed

    if a_valid and not b_valid:
        return resolution_a, "Resolution A has valid syntax, B does not"
    if b_valid and not a_valid:
        return resolution_b, "Resolution B has valid syntax, A does not"
    if not a_valid and not b_valid:
        return resolution_a, "Neither has valid syntax, defaulting to A"

    # Check import preservation
    a_imports = check_imports_preserved(resolution_a, base, ours, theirs)
    b_imports = check_imports_preserved(resolution_b, base, ours, theirs)

    if a_imports.passed and not b_imports.passed:
        return resolution_a, "Resolution A preserves imports correctly"
    if b_imports.passed and not a_imports.passed:
        return resolution_b, "Resolution B preserves imports correctly"

    # Check function preservation
    a_funcs = check_functions_preserved(resolution_a, base, ours, theirs)
    b_funcs = check_functions_preserved(resolution_b, base, ours, theirs)

    if a_funcs.passed and not b_funcs.passed:
        return resolution_a, "Resolution A preserves functions correctly"
    if b_funcs.passed and not a_funcs.passed:
        return resolution_b, "Resolution B preserves functions correctly"

    # If equal on all checks, prefer the longer one (likely more complete)
    if len(resolution_a) >= len(resolution_b):
        return resolution_a, "Both equal on checks, A is longer/equal"
    else:
        return resolution_b, "Both equal on checks, B is longer"
