from __future__ import annotations

import random
from typing import Optional

from datasets import load_dataset

DATASET_NAME = "tatsu-lab/alpaca"


def load_alpaca_prompts(
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """Load benign prompts from the Alpaca instruction dataset.

    Returns a list of {"prompt": str, "response": str} dicts sampled from the
    Alpaca training set. These serve as a control against the PKU harmful prompts:
    if the safety direction also aligns with benign prompt shifts, it captures
    generic model drift rather than safety-specific geometry.
    """
    ds = load_dataset(DATASET_NAME, split="train")

    examples = []
    for row in ds:
        prompt = row["instruction"]
        if row["input"]:
            prompt = f"{prompt}\n\n{row['input']}"
        examples.append({"prompt": prompt, "response": row["output"]})

    rng = random.Random(seed)
    rng.shuffle(examples)

    if n_samples is not None:
        examples = examples[:n_samples]

    return examples
