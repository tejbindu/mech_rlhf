from __future__ import annotations

import random
from typing import Optional

from datasets import load_dataset

DATASET_NAME = "PKU-Alignment/PKU-SafeRLHF"
DEFAULT_SUBSET = "alpaca-7b"
DEFAULT_HARM_CATEGORY = "Physical Harm"
DEFAULT_MIN_SEVERITY = 2
DEFAULT_SAFE_MAX_SEVERITY = 0


def _matches_harm(harm_cat: dict, harm_category: str) -> bool:
    """Return True if the response matches the requested harm category.

    Pass harm_category="all" to accept any non-empty harm label.
    """
    if harm_category == "all":
        return any(harm_cat.values())
    return harm_cat.get(harm_category, False)


def _extract_pair(
    row: dict,
    harm_category: str,
    min_severity: int,
    safe_max_severity: int,
) -> Optional[dict]:
    """Try both response orderings and return a valid (safe, unsafe) pair or None.

    The unsafe side must match `harm_category` (or any harm if "all") at
    severity >= min_severity. The safe side must have severity_level <= safe_max_severity.
    """
    for unsafe_idx, safe_idx in [(0, 1), (1, 0)]:
        harm_cat = row[f"response_{unsafe_idx}_harm_category"]
        if (
            _matches_harm(harm_cat, harm_category)
            and row[f"response_{unsafe_idx}_severity_level"] >= min_severity
            and row[f"response_{safe_idx}_severity_level"] <= safe_max_severity
        ):
            return {
                "prompt": row["prompt"],
                "safe_response": row[f"response_{safe_idx}"],
                "unsafe_response": row[f"response_{unsafe_idx}"],
            }
    return None


def load_pku_prompts(
    n_samples: Optional[int] = None,
    subset: str = DEFAULT_SUBSET,
    harm_category: str = "all",
    min_severity: int = 1,
    split: str = "train",
    seed: int = 42,
) -> list[dict]:
    """Load unique harmful prompts from PKU-SafeRLHF — no safe/unsafe pair required.

    For prompt_last pooling the response is causally masked at the extraction
    position, so only the prompt matters. This lets us use every row that has
    at least one harmful response, rather than only rows with a clean safe
    counterpart (which cuts the dataset from ~20K to ~2.8K).

    Returns list of {"prompt": str, "response": " "} where the response is a
    single space — present only so the tokeniser has something to append, but
    it does not affect the prompt_last hidden state.
    """
    ds = load_dataset(DATASET_NAME, subset, split=split)

    seen: set[str] = set()
    prompts: list[dict] = []
    for row in ds:
        prompt = row["prompt"]
        if prompt in seen:
            continue
        for i in [0, 1]:
            if (
                _matches_harm(row[f"response_{i}_harm_category"], harm_category)
                and row[f"response_{i}_severity_level"] >= min_severity
            ):
                prompts.append({"prompt": prompt, "response": " "})
                seen.add(prompt)
                break

    rng = random.Random(seed)
    rng.shuffle(prompts)

    if n_samples is not None:
        prompts = prompts[:n_samples]

    return prompts


def load_pku_pairs(
    n_samples: Optional[int] = None,
    subset: str = DEFAULT_SUBSET,
    harm_category: str = DEFAULT_HARM_CATEGORY,
    min_severity: int = DEFAULT_MIN_SEVERITY,
    safe_max_severity: int = DEFAULT_SAFE_MAX_SEVERITY,
    split: str = "train",
    seed: int = 42,
) -> list[dict]:
    """Load PKU-SafeRLHF and return (safe, unsafe) pairs.

    Each returned dict has keys: prompt, safe_response, unsafe_response.
    Filtering keeps pairs where:
      - The unsafe response has `harm_category` at severity >= min_severity
      - The safe response has severity_level <= safe_max_severity (default 0)
    """
    ds = load_dataset(DATASET_NAME, subset, split=split)

    pairs = []
    for row in ds:
        pair = _extract_pair(row, harm_category, min_severity, safe_max_severity)
        if pair is not None:
            pairs.append(pair)

    rng = random.Random(seed)
    rng.shuffle(pairs)

    if n_samples is not None:
        pairs = pairs[:n_samples]

    return pairs
