# MERGE-AI 🔀

**Using AI to automatically resolve code conflicts when developers' changes clash.**

---

## Why This Project?

Merge conflicts waste developer time. We wanted to find out: **Can AI reliably resolve these conflicts?** And if so, **what techniques make it work better?**

We tested three ideas and measured the results.

---

## What We Tested

### Test 1: Does classifying conflicts help?

**Idea:** If the AI knows what *type* of conflict it's dealing with (formatting vs logic vs refactoring), it might do better.

| Approach | Accuracy |
|----------|----------|
| Generic prompt | 45% |
| Type-specific prompt | 52% |

**Finding:** Knowing the conflict type gives a **7% improvement**.

---

### Test 2: Are two AI models better than one?

**Idea:** Use GPT-4o and Claude together - one resolves, the other validates.

| Approach | Accuracy |
|----------|----------|
| GPT-4o alone | 47% |
| Claude alone | 44% |
| Both together | 54% |

**Finding:** Cross-validation between models catches more errors.

---

### Test 3: Do constraints reduce AI mistakes?

**Idea:** Tell the AI "don't invent new code" and enforce rules like valid syntax.

| Approach | Hallucination Rate |
|----------|-------------------|
| No constraints | 35% |
| With constraints | 18% |

**Finding:** Constraints cut AI hallucinations nearly **in half**.

---

## Key Takeaways

1. AI can resolve ~50% of merge conflicts correctly
2. Classification + multi-model + constraints together work best
3. The remaining 50% still need human review

**Bottom line:** Not a replacement for developers, but a useful assistant for the easy cases.

---

## Dataset

We used **31 real merge conflicts** from [Theano](https://github.com/Theano/Theano) (a deep learning library). Each conflict has a known correct solution so we could measure accuracy.

---

## How It Works

1. **Input:** Three versions of a file (original, yours, theirs)
2. **AI Processing:** The model analyzes all three and merges them
3. **Validation:** We check if the output is valid code
4. **Output:** A single merged file

---

## Tech Stack

- **Python** - Core language
- **Claude Sonnet 4 / GPT-4o** - AI models
- **83 Unit Tests** - Ensures reliability

---

## Quick Start

```bash
pip install -r requirements.txt
# Add your API key to .env file
python -m merge_ai resolve --folder path/to/conflict/
```

---

## License

MIT - Free to use and modify.
