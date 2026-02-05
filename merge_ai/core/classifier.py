"""Conflict type classification for H1 experiment."""

import json
from enum import Enum
from dataclasses import dataclass
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
    OPENROUTER_SONNET,
)


class ConflictType(Enum):
    """High-level conflict types for classification."""

    SYNTACTIC = "syntactic"  # Imports, formatting, comments
    SEMANTIC = "semantic"  # Logic changes, feature changes
    STRUCTURAL = "structural"  # Refactoring, renames
    UNKNOWN = "unknown"


@dataclass
class Classification:
    """Result of conflict classification."""

    conflict_type: ConflictType
    confidence: float
    rationale: str
    strategy: str
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "conflict_type": self.conflict_type.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "strategy": self.strategy,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


CLASSIFICATION_SYSTEM_PROMPT = """You are an expert at analyzing Python merge conflicts. Your task is to classify the type of conflict between two versions of code that have diverged from a common base.

Classify the conflict into ONE of these types:

1. SYNTACTIC - Changes to imports, formatting, comments, docstrings, or other surface-level syntax
   - Import reordering or additions
   - Whitespace/formatting changes
   - Comment modifications
   - Docstring updates

2. SEMANTIC - Changes to program logic, behavior, or functionality
   - Logic/algorithm changes
   - New features or feature modifications
   - Parameter or return value changes
   - Error handling changes
   - API behavior changes

3. STRUCTURAL - Changes to code organization without changing behavior
   - Function/class renaming
   - Moving code between files
   - Refactoring patterns
   - Reorganizing class hierarchies

Respond with a JSON object containing:
{
    "type": "SYNTACTIC" | "SEMANTIC" | "STRUCTURAL",
    "confidence": 0.0-1.0,
    "rationale": "Brief explanation of why this classification",
    "strategy": "Suggested resolution approach"
}

Only output the JSON, no other text."""


CLASSIFICATION_USER_PROMPT = """Analyze this merge conflict and classify it:

=== BASE (original version) ===
{base}

=== OURS (our changes) ===
{ours}

=== THEIRS (their changes) ===
{theirs}

Classify this conflict and provide your analysis as JSON."""


def classify_conflict(
    base: str,
    ours: str,
    theirs: str,
    model: Optional[ModelConfig] = None,
) -> Classification:
    """Classify a merge conflict into SYNTACTIC, SEMANTIC, or STRUCTURAL.

    Args:
        base: Base version of the file
        ours: Our branch version
        theirs: Their branch version
        model: Model configuration to use (defaults to OpenRouter Sonnet)

    Returns:
        Classification with type, confidence, rationale, and strategy
    """
    model = model or OPENROUTER_SONNET

    user_prompt = CLASSIFICATION_USER_PROMPT.format(
        base=base[:10000],  # Truncate to avoid token limits
        ours=ours[:10000],
        theirs=theirs[:10000],
    )

    if model.provider in ("openai", "openrouter"):
        return _classify_openai(user_prompt, model)
    else:
        return _classify_anthropic(user_prompt, model)


def _classify_openai(user_prompt: str, model: ModelConfig) -> Classification:
    """Classify using OpenAI-compatible API (OpenAI or OpenRouter)."""
    if model.provider == "openrouter":
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=model.model,
        temperature=model.temperature,
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0

    return _parse_classification(content, input_tokens, output_tokens)


def _classify_anthropic(user_prompt: str, model: ModelConfig) -> Classification:
    """Classify using Anthropic API."""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=model.model,
        max_tokens=1024,
        system=CLASSIFICATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    content = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    return _parse_classification(content, input_tokens, output_tokens)


def _parse_classification(
    content: str, input_tokens: int, output_tokens: int
) -> Classification:
    """Parse LLM response into Classification object."""
    try:
        # Handle potential markdown wrapping
        if "```" in content:
            import re

            match = re.search(r"```(?:json)?\s*\n?(.*?)```", content, re.DOTALL)
            if match:
                content = match.group(1)

        data = json.loads(content)

        type_str = data.get("type", "UNKNOWN").upper()
        conflict_type = ConflictType[type_str] if type_str in ConflictType.__members__ else ConflictType.UNKNOWN

        return Classification(
            conflict_type=conflict_type,
            confidence=float(data.get("confidence", 0.5)),
            rationale=data.get("rationale", ""),
            strategy=data.get("strategy", ""),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return Classification(
            conflict_type=ConflictType.UNKNOWN,
            confidence=0.0,
            rationale=f"Failed to parse classification: {e}",
            strategy="Use generic resolution",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def get_type_specific_prompt(conflict_type: ConflictType) -> str:
    """Get resolution prompt tailored to the conflict type.

    Args:
        conflict_type: The classified conflict type

    Returns:
        Additional prompt text for type-specific resolution
    """
    prompts = {
        ConflictType.SYNTACTIC: """This is a SYNTACTIC conflict (imports, formatting, comments).

RESOLUTION STRATEGY:
- Preserve consistent formatting throughout
- Combine imports intelligently (alphabetize, group by type)
- Keep comments from both versions if they add value
- Prefer the more complete/updated docstrings
- Maintain code style consistency""",
        ConflictType.SEMANTIC: """This is a SEMANTIC conflict (logic/behavior changes).

RESOLUTION STRATEGY:
- Be CONSERVATIVE - preserve both behaviors if possible
- If behaviors conflict, prefer the safer/more defensive approach
- Do not lose any feature functionality from either branch
- Ensure error handling covers all cases from both versions
- Test edge cases in your mind before deciding""",
        ConflictType.STRUCTURAL: """This is a STRUCTURAL conflict (refactoring/reorganization).

RESOLUTION STRATEGY:
- Track all renames carefully to ensure consistency
- Maintain code organization and modularity
- Prefer the cleaner/more organized structure
- Ensure all references are updated correctly
- Preserve the intent of the refactoring""",
        ConflictType.UNKNOWN: """The conflict type is unclear.

RESOLUTION STRATEGY:
- Analyze carefully before making changes
- Be conservative and preserve functionality
- When in doubt, keep both versions' contributions""",
    }
    return prompts.get(conflict_type, prompts[ConflictType.UNKNOWN])
