# MERGE-AI 🔀

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-83%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Python framework for automated merge conflict resolution using Large Language Models.**

---

## What It Does

MERGE-AI takes three versions of a Python file (base, ours, theirs) and uses LLMs to automatically produce a merged result.

```bash
python -m merge_ai resolve --folder path/to/conflict/
```

---

## Features

- **Multi-Model Support** - Claude Sonnet 4, GPT-4o via OpenRouter
- **Conflict Classification** - Detects syntactic, semantic, and structural conflicts
- **Validation Pipeline** - AST checking, import verification, function preservation
- **Intelligent Retry** - Error-specific prompts for automatic recovery
- **Benchmarking** - Compare different approaches with cost tracking

---

## Installation

```bash
git clone https://github.com/Ashwinash27/merge-ai-python.git
cd merge-ai-python
pip install -r requirements.txt
```

Create `.env` with your API keys:
```
OPENROUTER_API_KEY=sk-or-...
```

---

## Usage

```bash
# Validate setup
python -m merge_ai validate

# List available test conflicts
python -m merge_ai list

# Resolve a conflict
python -m merge_ai resolve --folder merge_ai/data/conflicts/sample1/

# Resolve with constraints (AST validation, import checking)
python -m merge_ai resolve --folder merge_ai/data/conflicts/sample1/ --constraints all

# Run benchmark
python -m merge_ai benchmark --pilot
```

---

## Project Structure

```
merge_ai/
├── core/
│   ├── resolver.py         # LLM resolution logic
│   ├── classifier.py       # Conflict type classification
│   ├── validator.py        # Multi-model validation
│   ├── constraints.py      # AST/import/function constraints
│   ├── errors.py           # Error detection & retry
│   └── metrics.py          # Quality metrics
├── evaluation/
│   ├── experiments.py      # Experiment runners
│   ├── benchmark.py        # Benchmarking
│   └── report.py           # Report generation
├── data/
│   └── conflicts/          # 31 test cases from Theano
└── cli.py
```

---

## Dataset

31 real merge conflicts from [Theano](https://github.com/Theano/Theano) (deep learning library), each with:
- `base.py` - Original code
- `ours.py` - One branch's changes
- `theirs.py` - Other branch's changes
- `gold.py` - Correct merged result

---

## Tests

```bash
pytest tests/ -v
# 83 tests covering constraints, errors, metrics, comparison
```

---

## License

MIT
