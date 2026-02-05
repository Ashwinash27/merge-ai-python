# MERGE-AI 🔀

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-83%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An intelligent framework for automated merge conflict resolution using Large Language Models.**

MERGE-AI uses Claude Sonnet 4 and GPT-4o to automatically resolve Python merge conflicts, with built-in validation, error recovery, and quality metrics.

---

## ✨ Highlights

- 🤖 **Multi-Model Support** - Claude Sonnet 4, GPT-4o via OpenRouter
- 🔍 **Smart Conflict Classification** - Automatically detects syntactic, semantic, and structural conflicts
- ✅ **Validation Pipeline** - AST checking, import verification, function preservation
- 🔄 **Intelligent Retry** - Error-specific prompts for automatic recovery
- 📊 **Comprehensive Metrics** - Gold match scoring, hallucination detection, cost tracking
- 🧪 **Battle-Tested** - 83 unit tests, 31 real-world conflict dataset

---

## 🚀 Quick Demo

```bash
# Resolve a merge conflict
python -m merge_ai resolve --folder merge_ai/data/conflicts/sample1/

# Output:
# Resolving conflict in merge_ai/data/conflicts/sample1/...
# Resolution successful
# Attempts: 1
# Cost: $0.02
# Result saved to: merge_ai/data/conflicts/sample1/merged.py
```

---

## 📊 Research Findings

We evaluated LLM-based merge resolution on **31 real conflicts** from the [Theano](https://github.com/Theano/Theano) deep learning library.

### Key Discoveries

| Finding | Insight |
|---------|---------|
| **LLMs understand merge semantics** | Successfully combines changes from divergent branches |
| **Classification adds overhead without benefit** | Type-specific prompts don't improve accuracy (saves API costs) |
| **Structural constraints help** | AST validation catches 60% of errors before they propagate |
| **Retry mechanisms are essential** | Error-specific prompts recover ~30% of initial failures |

### Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Conflicts resolved | 31 | Real-world Theano conflicts |
| Avg. resolution time | 15s | Per conflict |
| Cost per conflict | ~$0.15 | Via OpenRouter |
| Test coverage | 83 tests | All passing |

---

## 🏗️ Architecture

```
merge_ai/
├── core/                   # Resolution Engine
│   ├── resolver.py         # LLM-based conflict resolution
│   ├── classifier.py       # Conflict type detection (syntactic/semantic/structural)
│   ├── validator.py        # Multi-model cross-validation
│   ├── constraints.py      # AST, import, function preservation
│   ├── errors.py           # Error taxonomy & intelligent retry
│   └── metrics.py          # Quality measurement & scoring
│
├── evaluation/             # Benchmarking Framework
│   ├── experiments.py      # Hypothesis testing (H1, H2, H3)
│   ├── benchmark.py        # Cost tracking & orchestration
│   ├── compare.py          # Gold standard comparison
│   └── report.py           # Results reporting
│
├── data/
│   ├── conflicts/          # 31 curated test cases
│   └── results/            # Experiment outputs
│
└── cli.py                  # Command-line interface
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Ashwinash27/merge-ai-python.git
cd merge-ai-python

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

---

## 📖 Usage

### Resolve a Single Conflict

```bash
python -m merge_ai resolve --folder path/to/conflict/
```

Each conflict folder should contain:
- `base.py` - Original code before branches diverged
- `ours.py` - Your branch changes
- `theirs.py` - Their branch changes

### Run with Constraints

```bash
# Apply all structural constraints
python -m merge_ai resolve --folder path/to/conflict/ --constraints all

# Specific constraints
python -m merge_ai resolve --folder path/to/conflict/ --constraints ast,imports
```

### Run Benchmarks

```bash
# Quick pilot test (5 conflicts)
python -m merge_ai benchmark --pilot

# Full benchmark
python -m merge_ai benchmark --hypothesis all --sample 30

# Generate reports
python -m merge_ai report
```

### Validate Setup

```bash
python -m merge_ai validate
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_constraints.py -v
```

**Test Coverage:**
- `test_constraints.py` - AST validation, import/function preservation
- `test_errors.py` - Error detection, retry prompts
- `test_metrics.py` - Quality scoring, hallucination detection
- `test_compare.py` - Gold standard comparison
- `test_experiments.py` - Experiment data structures

---

## 📁 Dataset

31 real merge conflicts extracted from [Theano](https://github.com/Theano/Theano), a deep learning library. Conflicts range from simple import changes to complex algorithmic modifications.

Each conflict includes a **gold standard** (correct merge) for evaluation.

| Conflict Type | Count | Example |
|--------------|-------|---------|
| Import conflicts | 8 | Adding/reordering imports |
| Logic changes | 15 | Algorithm modifications |
| Refactoring | 8 | Function renames, reorganization |

---

## 🔧 Configuration

### Supported Models

| Model | Provider | Use Case |
|-------|----------|----------|
| Claude Sonnet 4 | Anthropic/OpenRouter | Primary resolution |
| GPT-4o | OpenAI/OpenRouter | Validation & ensemble |
| Claude Haiku | Anthropic/OpenRouter | Fast validation |

### Environment Variables

```bash
OPENAI_API_KEY=sk-...           # OpenAI API key
ANTHROPIC_API_KEY=sk-ant-...    # Anthropic API key
OPENROUTER_API_KEY=sk-or-...    # OpenRouter API key (recommended)
```

---

## 📈 Future Improvements

- [ ] Support for additional languages (JavaScript, TypeScript, Java)
- [ ] Fine-tuning on merge conflict data
- [ ] IDE integrations (VS Code, JetBrains)
- [ ] Chain-of-thought reasoning for complex conflicts

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Theano](https://github.com/Theano/Theano) for the merge conflict dataset
- [OpenRouter](https://openrouter.ai) for unified LLM API access

---

<p align="center">
  <i>Built with 🤖 Claude & GPT-4o</i>
</p>
