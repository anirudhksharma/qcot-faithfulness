"""Dataset loading. Real runs use MMLU; offline/mock uses the bundled sample."""
import json
import os
from typing import List, Dict


def load_sample(path: str = "sample_questions.json") -> List[Dict]:
    """Load the bundled offline MCQA items."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, path)) as f:
        return json.load(f)


def load_mmlu(limit: int) -> List[Dict]:
    """Load MMLU test items via the `datasets` library (real runs only).

    Returns dicts of {question, options[list of 4], answer_idx}. Requires
    `pip install datasets` and network on first download (cached afterwards).
    """
    from datasets import load_dataset  # imported lazily so mock mode needs no dep
    ds = load_dataset("cais/mmlu", "all", split="test")
    out = []
    for row in ds:
        opts = row["choices"]
        if len(opts) != 4:
            continue
        out.append({
            "question": row["question"],
            "options": list(opts),
            "answer_idx": int(row["answer"]),
        })
        if len(out) >= limit:
            break
    return out


def load_dataset_items(name: str, limit: int) -> List[Dict]:
    """Dispatch on dataset name."""
    if name == "sample":
        return load_sample()[:limit]
    if name == "mmlu":
        return load_mmlu(limit)
    raise ValueError(f"unknown dataset {name!r} (use 'mmlu' or 'sample')")


def format_prompt(question: str, options: List[str], injected_hint: str = "") -> str:
    """Build the user prompt: question + lettered options + optional cue."""
    letters = [chr(ord("A") + i) for i in range(len(options))]
    lines = [question, ""]
    for letter, opt in zip(letters, options):
        lines.append(f"({letter}) {opt}")
    if injected_hint:
        lines.append("")
        lines.append(injected_hint)
    lines.append("")
    lines.append("Think step by step, then end with 'The answer is (X)'.")
    return "\n".join(lines)
