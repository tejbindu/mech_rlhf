from __future__ import annotations

import numpy as np


def compute_cosine_audit(
    alpaca_acts: np.ndarray,
    beaver_acts: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Compute per-example per-layer cosine similarity between the RLHF shift and safety direction.

    For each example i and layer ℓ:
        Δh = beaver_acts[i, ℓ] - alpaca_acts[i, ℓ]
        result[i, ℓ] = cos(Δh, directions[ℓ])

    Args:
        alpaca_acts: [n, n_layers, d_model] — alpaca activations on shared prompts
        beaver_acts: [n, n_layers, d_model] — beaver activations on the same prompts
        directions:  [n_layers, d_model]    — L2-normalized safety directions

    Returns:
        cosines: [n, n_layers] float array in [-1, 1].
    """
    delta = beaver_acts - alpaca_acts  # [n, n_layers, d_model]

    # Dot product of each Δh with its layer's direction
    dots = (delta * directions[np.newaxis, :, :]).sum(axis=-1)  # [n, n_layers]

    # Norms (handle zero-norm edge case)
    delta_norms = np.linalg.norm(delta, axis=-1)                   # [n, n_layers]
    delta_norms = np.where(delta_norms == 0, 1.0, delta_norms)

    dir_norms = np.linalg.norm(directions, axis=-1)                # [n_layers]
    dir_norms = np.where(dir_norms == 0, 1.0, dir_norms)

    return dots / (delta_norms * dir_norms[np.newaxis, :])         # [n, n_layers]


def summarize(cosines: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-layer summary statistics over examples.

    Args:
        cosines: [n, n_layers]

    Returns:
        dict with keys mean, std, q25, q75 — each a [n_layers] array.
    """
    return {
        "mean": cosines.mean(axis=0),
        "std":  cosines.std(axis=0),
        "q25":  np.quantile(cosines, 0.25, axis=0),
        "q75":  np.quantile(cosines, 0.75, axis=0),
    }
