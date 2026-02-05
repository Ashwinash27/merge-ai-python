# MERGE-AI: Empirical Study of LLM Merge Resolution

## Project Context

This is a research project testing 3 hypotheses about LLM merge conflict resolution:
- **H1:** Classification improves resolution (knowing conflict type leads to better prompts)
- **H2:** Multi-model beats single-model (GPT + Claude validation/ensemble)
- **H3:** Constraints reduce hallucination (AST, import, function preservation)

**Primary deliverable:** Benchmark results + analysis, not a production tool.
**Language:** Python only.

## Architecture Overview

```
merge_ai/
├── core/                   # Core resolution components
│   ├── resolver.py         # Main LLM resolution logic (GPT + Claude)
│   ├── classifier.py       # Conflict type classification (H1)
│   ├── validator.py        # Cross-model validation (H2)
│   ├── constraints.py      # AST, import, function constraints (H3)
│   ├── errors.py           # Error taxonomy + intelligent retry
│   └── metrics.py          # Hallucination measurement
├── evaluation/             # Experiment framework
│   ├── experiments.py      # H1, H2, H3 experiment runners
│   ├── benchmark.py        # Orchestration + cost tracking
│   ├── compare.py          # Compare to gold standard
│   ├── error_analysis.py   # Aggregate error statistics
│   └── report.py           # Generate Markdown reports
├── data/
│   ├── conflicts/          # Test cases (base.py, ours.py, theirs.py, gold.py)
│   └── results/            # Experiment outputs (JSON + CSV)
├── cli.py                  # Command-line interface
└── config.py               # API keys, model settings
```

## Key Files

- `core/resolver.py` - Main LLM resolution logic with retry
- `core/errors.py` - Error taxonomy system (SYNTAX_ERROR, HALLUCINATION_IMPORT, etc.)
- `core/constraints.py` - Constraint checking (AST validity, import preservation, function signatures)
- `evaluation/experiments.py` - H1/H2/H3 experiment implementations
- `evaluation/benchmark.py` - Experiment orchestration with cost tracking
- `cli.py` - Command-line interface

## Running Experiments

```bash
# Validate setup
python -m merge_ai validate

# List available conflicts
python -m merge_ai list

# Resolve single conflict
python -m merge_ai resolve --folder data/conflicts/sample1/

# Run pilot (5 conflicts, all hypotheses)
python -m merge_ai benchmark --pilot

# Run specific hypothesis
python -m merge_ai benchmark --hypothesis H3 --sample 30

# Run all hypotheses
python -m merge_ai benchmark --hypothesis all --sample 30

# Generate reports from results
python -m merge_ai report --results data/results/
```

## LLM Prompt Templates

### Classification Prompt (H1)

```
You are an expert at analyzing Python merge conflicts. Classify into:
1. SYNTACTIC - imports, formatting, comments
2. SEMANTIC - logic changes, features
3. STRUCTURAL - refactoring, renames

Respond with JSON: {type, confidence, rationale, strategy}
```

### Resolution Prompt (Base)

```
You are an expert software engineer resolving a Python merge conflict.

CRITICAL RULES:
1. Output ONLY valid Python code - no markdown, no explanations
2. Preserve all functionality from both OURS and THEIRS
3. Do not add new imports not in inputs
4. Maintain consistent style
```

### Constraint Prompts (H3)

**AST Constraint:**
```
CONSTRAINT: Your output MUST be valid Python that parses without errors.
Do not include markdown or code fences. Output ONLY the merged Python file.
```

**Import Constraint:**
```
CONSTRAINT: You may ONLY use imports from BASE, OURS, or THEIRS.
Do not add any new imports.
```

**Function Constraint:**
```
CONSTRAINT: Preserve all function names and signatures.
Do not rename functions or change parameters.
```

### Error-Specific Retry Prompts

```python
{
    "SYNTAX_ERROR": "Fix syntax error: {details}. Output valid Python.",
    "TRUNCATION": "Output was incomplete. Provide COMPLETE merged file.",
    "HALLUCINATION_IMPORT": "Remove invalid imports: {details}",
    "HALLUCINATION_IDENTIFIER": "Use ONLY identifiers from input files.",
}
```

## Data Format

### Conflict Input Structure

```
conflict_folder/
├── base.py     # Original code (before divergence)
├── ours.py     # Our branch changes
├── theirs.py   # Their branch changes
└── gold.py     # Ground truth (expected merge result)
```

### Result Output (JSON)

```json
{
  "hypothesis": "H3: Constraints reduce hallucination",
  "conditions": ["baseline", "ast_only", "imports_only", "functions_only", "all_constraints"],
  "results": [
    {
      "conflict_id": "sample1",
      "condition": "all_constraints",
      "metrics": {
        "syntax_valid": true,
        "gold_match_score": 0.85,
        "hallucinated_imports": 0,
        "hallucinated_identifiers": 2,
        "cost_usd": 0.02,
        "attempts": 1
      }
    }
  ],
  "total_cost_usd": 25.50,
  "duration_seconds": 1234.5
}
```

## Error Taxonomy

| Error Type | Detection | Retry Strategy |
|------------|-----------|----------------|
| SYNTAX_ERROR | `ast.parse()` fails | "Fix syntax: {error}" |
| HALLUCINATION_IMPORT | Import not in inputs | "Remove: {imports}" |
| HALLUCINATION_IDENTIFIER | Variable not in inputs | "Use only existing identifiers" |
| TRUNCATION | Incomplete output | "Complete the full file" |
| EMPTY_OUTPUT | No content | "Provide complete code" |

## Metrics

1. **Syntax validity:** Does output parse as Python? (0/1)
2. **Import hallucination:** # of imports not in base/ours/theirs
3. **Identifier hallucination:** # of variables/functions not in inputs
4. **Gold match:** Semantic equivalence to gold.py (0-1 score)
5. **Cost:** API cost per resolution in USD

## Testing

```bash
# Run tests
pytest tests/ -v

# Test specific module
pytest tests/test_constraints.py -v
```

## Configuration

API keys in `.env`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Model settings in `config.py`:
- GPT-4o: Primary model for resolution
- Claude Sonnet: Secondary model for validation
- Temperature: 0.1 (for deterministic outputs)
- Max tokens: 4096

## Important Decisions

- **Sample size:** 30 conflicts per hypothesis (budget-constrained)
- **Max retries:** 2 per resolution with error-specific prompts
- **Comparison:** AST-based semantic comparison for gold match scoring
- **Cost tracking:** Per-resolution and aggregate tracking
- **Checkpointing:** Results saved after each hypothesis

## Budget

| Hypothesis | Est. LLM Calls | Est. Cost |
|------------|----------------|-----------|
| H1 Classification | 90 | $10-15 |
| H2 Multi-model | 180 | $25-35 |
| H3 Constraints | 150 | $20-25 |
| Pilot + debug | ~50 | $10-15 |
| **Total** | **~470** | **$65-90** |
