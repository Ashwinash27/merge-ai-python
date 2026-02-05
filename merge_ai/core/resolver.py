"""Main LLM resolution logic for merge conflicts."""

import json
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from anthropic import Anthropic

from ..config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    ModelConfig,
    GPT_4O,
    CLAUDE_SONNET,
    OPENROUTER_SONNET,
)
from .constraints import Constraint, get_constraint_prompt, check_all_constraints
from .errors import (
    ErrorType,
    ErrorClassification,
    RetryResult,
    classify_error,
    get_retry_prompt,
    extract_code_from_markdown,
)
from .classifier import ConflictType, get_type_specific_prompt


@dataclass
class ResolutionResult:
    """Result of a merge conflict resolution."""

    code: str
    success: bool
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    attempts: int
    errors_encountered: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "attempts": self.attempts,
            "errors_encountered": self.errors_encountered,
        }


BASE_SYSTEM_PROMPT = """You are an expert software engineer resolving a Python merge conflict. Your task is to merge two divergent versions of code that both stem from a common base.

CRITICAL RULES:
1. Output ONLY valid Python code - no markdown, no explanations, no code fences
2. Preserve all functionality from both OURS and THEIRS branches
3. Do not add any new code, imports, or functionality not present in the inputs
4. Maintain consistent style and formatting
5. If changes are in different parts of the file, include BOTH
6. If changes conflict in the same location, intelligently combine them

Output the complete merged Python file, nothing else."""


BASE_USER_PROMPT = """Merge these two versions of Python code:

=== BASE (original version before divergence) ===
{base}

=== OURS (our branch's changes) ===
{ours}

=== THEIRS (their branch's changes) ===
{theirs}

Output ONLY the merged Python code:"""


class Resolver:
    """Resolver for merge conflicts with configurable model and constraints."""

    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        constraints: Optional[list[Constraint]] = None,
        max_retries: int = 2,
    ):
        """Initialize the resolver.

        Args:
            model: Model configuration to use (defaults to OpenRouter Sonnet)
            constraints: List of constraints to apply
            max_retries: Maximum retry attempts on error
        """
        self.model = model or OPENROUTER_SONNET
        self.constraints = constraints or []
        self.max_retries = max_retries

        if self.model.provider == "openrouter":
            self.client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
            )
        elif self.model.provider == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

    def resolve(
        self,
        base: str,
        ours: str,
        theirs: str,
        conflict_type: Optional[ConflictType] = None,
    ) -> ResolutionResult:
        """Resolve a merge conflict.

        Args:
            base: Base version of the file
            ours: Our branch version
            theirs: Their branch version
            conflict_type: Optional classified conflict type for type-specific prompts

        Returns:
            ResolutionResult with the merged code and metadata
        """
        # Build the prompt
        system_prompt = BASE_SYSTEM_PROMPT

        # Add type-specific guidance if classified
        if conflict_type:
            system_prompt += "\n\n" + get_type_specific_prompt(conflict_type)

        # Add constraint prompts
        constraint_prompt = get_constraint_prompt(self.constraints)
        if constraint_prompt:
            system_prompt += "\n\n" + constraint_prompt

        user_prompt = BASE_USER_PROMPT.format(
            base=base,
            ours=ours,
            theirs=theirs,
        )

        # Resolve with retry logic
        return self._resolve_with_retry(
            system_prompt, user_prompt, base, ours, theirs
        )

    def _resolve_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        base: str,
        ours: str,
        theirs: str,
    ) -> ResolutionResult:
        """Resolve with intelligent retry on errors."""
        errors_log = []
        total_input_tokens = 0
        total_output_tokens = 0
        messages = [{"role": "user", "content": user_prompt}]

        for attempt in range(self.max_retries + 1):
            # Call the LLM
            if self.model.provider in ("openai", "openrouter"):
                code, input_tokens, output_tokens = self._call_openai(
                    system_prompt, messages
                )
            else:
                code, input_tokens, output_tokens = self._call_anthropic(
                    system_prompt, messages
                )

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            # Clean the output
            code = extract_code_from_markdown(code)

            # Classify any errors
            error_result = classify_error(code, base, ours, theirs)

            if not error_result.has_error:
                # Success!
                cost = self._calculate_cost(total_input_tokens, total_output_tokens)
                return ResolutionResult(
                    code=code,
                    success=True,
                    model=self.model.model,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cost_usd=cost,
                    attempts=attempt + 1,
                    errors_encountered=errors_log,
                )

            # Log the error
            errors_log.append({
                "attempt": attempt + 1,
                "error_type": error_result.error_type.value,
                "details": error_result.details,
            })

            # If we have retries left, add retry prompt
            if attempt < self.max_retries:
                retry_prompt = get_retry_prompt(
                    error_result.error_type, error_result.details
                )
                messages.append({"role": "assistant", "content": code})
                messages.append({"role": "user", "content": retry_prompt})

        # All retries exhausted
        cost = self._calculate_cost(total_input_tokens, total_output_tokens)
        return ResolutionResult(
            code=code,
            success=False,
            model=self.model.model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost_usd=cost,
            attempts=self.max_retries + 1,
            errors_encountered=errors_log,
        )

    def _call_openai(
        self, system_prompt: str, messages: list[dict]
    ) -> tuple[str, int, int]:
        """Make an OpenAI API call."""
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        response = self.client.chat.completions.create(
            model=self.model.model,
            temperature=self.model.temperature,
            max_tokens=self.model.max_tokens,
            messages=full_messages,
        )

        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return content, input_tokens, output_tokens

    def _call_anthropic(
        self, system_prompt: str, messages: list[dict]
    ) -> tuple[str, int, int]:
        """Make an Anthropic API call."""
        response = self.client.messages.create(
            model=self.model.model,
            max_tokens=self.model.max_tokens,
            system=system_prompt,
            messages=messages,
        )

        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return content, input_tokens, output_tokens

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of the API calls."""
        input_cost = (input_tokens / 1000) * self.model.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.model.cost_per_1k_output
        return input_cost + output_cost


def resolve_conflict(
    base: str,
    ours: str,
    theirs: str,
    model: Optional[ModelConfig] = None,
    constraints: Optional[list[Constraint]] = None,
    conflict_type: Optional[ConflictType] = None,
    max_retries: int = 2,
) -> ResolutionResult:
    """Convenience function to resolve a merge conflict.

    Args:
        base: Base version of the file
        ours: Our branch version
        theirs: Their branch version
        model: Model configuration (defaults to GPT-4o)
        constraints: Constraints to apply
        conflict_type: Optional classified conflict type
        max_retries: Maximum retry attempts

    Returns:
        ResolutionResult with the merged code and metadata
    """
    resolver = Resolver(model=model, constraints=constraints, max_retries=max_retries)
    return resolver.resolve(base, ours, theirs, conflict_type=conflict_type)
