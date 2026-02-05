# MERGE-AI: Can AI Resolve Code Merge Conflicts?

An empirical study exploring whether Large Language Models (LLMs) can automatically resolve merge conflicts in Python code - and what techniques actually help.

## The Problem

When developers work on the same codebase, their changes sometimes conflict. Resolving these "merge conflicts" is tedious, error-prone, and time-consuming. What if AI could do it automatically?

## What We Built

A complete evaluation framework to test whether LLMs can resolve Python merge conflicts, and specifically whether certain techniques improve their performance:

- **Classification-guided resolution** - Does telling the AI what *type* of conflict it's looking at help?
- **Multi-model validation** - Do two AI models working together beat one alone?
- **Structural constraints** - Does enforcing rules (valid syntax, no new imports) reduce AI "hallucinations"?

## Key Findings

We tested Claude Sonnet 4 on real merge conflicts from the Theano machine learning library.

### Experiment Results

| Approach | Gold Match Score | Syntax Valid |
|----------|------------------|--------------|
| Basic resolution | 48.4% | 40% |
| With conflict classification | 48.5% | 40% |

**Key insight:** Classification alone doesn't significantly improve resolution quality. The challenge isn't understanding the conflict type - it's the fundamental difficulty of merging divergent code changes correctly.

### What This Means

1. **LLMs can resolve simple conflicts** - When changes are in different parts of a file, AI does well
2. **Complex semantic conflicts remain hard** - When both branches change the same logic, AI struggles
3. **40% syntax validity shows room for improvement** - Constraint-based approaches may help here

## Architecture

```
merge_ai/
├── core/                   # Core resolution engine
│   ├── resolver.py         # LLM-based conflict resolution
│   ├── classifier.py       # Conflict type detection
│   ├── validator.py        # Multi-model validation
│   ├── constraints.py      # AST/import/function constraints
│   ├── errors.py           # Error detection & smart retry
│   └── metrics.py          # Quality measurement
├── evaluation/             # Experiment framework
│   ├── experiments.py      # H1, H2, H3 experiment runners
│   ├── benchmark.py        # Cost tracking & orchestration
│   ├── compare.py          # Gold standard comparison
│   └── report.py           # Results reporting
├── data/
│   ├── conflicts/          # 31 real-world test cases
│   └── results/            # Experiment outputs
└── cli.py                  # Command-line interface
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Ashwinash27/merge-ai-python.git
cd merge-ai-python

# Install dependencies
pip install -r requirements.txt

# Set up API keys in .env
echo "OPENAI_API_KEY=your-key" > .env
echo "ANTHROPIC_API_KEY=your-key" >> .env
```

### Usage

```bash
# Validate your setup
python -m merge_ai validate

# List available test conflicts
python -m merge_ai list

# Resolve a single conflict
python -m merge_ai resolve --folder merge_ai/data/conflicts/sample1/

# Run experiments
python -m merge_ai benchmark --pilot  # Quick test (5 conflicts)
python -m merge_ai benchmark --hypothesis H1 --sample 30  # Full experiment

# Generate reports
python -m merge_ai report
```

## How It Works

### 1. Conflict Input
Each merge conflict has four files:
- `base.py` - Original code before the branches diverged
- `ours.py` - Changes from one branch
- `theirs.py` - Changes from another branch
- `gold.py` - The correct merged result (for evaluation)

### 2. Resolution Process
The AI receives all three versions and must produce a merged result that:
- Preserves functionality from both branches
- Compiles as valid Python
- Doesn't introduce new code that wasn't in either branch

### 3. Quality Metrics
We measure:
- **Syntax validity** - Does the output parse as Python?
- **Gold match score** - How similar is it to the correct merge?
- **Hallucination count** - Did the AI invent code that wasn't there?

## Dataset

31 real merge conflicts from the [Theano](https://github.com/Theano/Theano) deep learning library, ranging from simple import conflicts to complex algorithmic changes.

## Technical Details

- **Models tested:** Claude Sonnet 4, GPT-4o (via OpenRouter)
- **Evaluation metric:** AST-based semantic similarity + line matching
- **Error handling:** Automatic retry with error-specific prompts
- **Cost tracking:** Per-resolution and aggregate API cost monitoring

## Limitations

- Python-only (no other languages)
- Sample size limited by API costs
- Gold standard may not always be the optimal merge
- Results may vary with different model versions

## Future Work

- Test with larger sample sizes
- Explore fine-tuning on merge conflict data
- Add support for other programming languages
- Investigate chain-of-thought prompting approaches

## Project Structure

```
.
├── merge_ai/           # Main package
├── tests/              # Unit tests (83 tests)
├── requirements.txt    # Dependencies
├── CLAUDE.md          # Development documentation
└── README.md          # This file
```

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT License - See LICENSE file for details.

---

*Built as part of research into AI-assisted software development.*
