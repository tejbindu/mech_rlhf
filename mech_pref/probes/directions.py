from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mech_pref.config import Config

logger = logging.getLogger(__name__)


def compute_directions(safe_acts: np.ndarray, unsafe_acts: np.ndarray) -> np.ndarray:
    """Compute per-layer mean-difference safety directions.

    Args:
        safe_acts:   [n, n_layers, d_model]
        unsafe_acts: [n, n_layers, d_model]

    Returns:
        directions: [n_layers, d_model], each row L2-normalized.
    """
    n = min(safe_acts.shape[0], unsafe_acts.shape[0])
    diff = safe_acts[:n].mean(axis=0) - unsafe_acts[:n].mean(axis=0)  # [n_layers, d_model]
    norms = np.linalg.norm(diff, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    return diff / norms


def compute_auroc_per_layer(
    safe_acts: np.ndarray,
    unsafe_acts: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Train a logistic regression per layer and return held-out AUROC.

    Labels: safe=1, unsafe=0. The 80/20 split is fixed by cfg.run.seed.

    Args:
        safe_acts:   [n, n_layers, d_model]
        unsafe_acts: [n, n_layers, d_model]
        cfg: experiment config (probes.test_size, probes.max_iter, probes.n_jobs, run.seed)

    Returns:
        aurocs: [n_layers] float array of held-out AUROC values.
    """
    # Trim to the smaller of the two sets in case a small number of examples
    # were skipped during tokenization (e.g. zero-length responses).
    n = min(safe_acts.shape[0], unsafe_acts.shape[0])
    safe_acts  = safe_acts[:n]
    unsafe_acts = unsafe_acts[:n]
    n_layers = safe_acts.shape[1]

    y = np.concatenate([np.ones(n), np.zeros(n)], axis=0)  # [2n]
    aurocs = np.zeros(n_layers)

    for layer in range(n_layers):
        # Build X one layer at a time to avoid a giant [2n, n_layers, d_model] array.
        X_layer = np.concatenate([safe_acts[:, layer, :], unsafe_acts[:, layer, :]], axis=0)

        X_train, X_test, y_train, y_test = train_test_split(
            X_layer, y,
            test_size=cfg.probes.test_size,
            random_state=cfg.run.seed,
            stratify=y,
        )

        clf = LogisticRegression(
            max_iter=cfg.probes.max_iter,
            n_jobs=cfg.probes.n_jobs,
            random_state=cfg.run.seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)

        scores = clf.predict_proba(X_test)[:, 1]
        aurocs[layer] = roc_auc_score(y_test, scores)

    return aurocs


def save_directions(directions: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, directions=directions)
    logger.info("Saved directions %s → %s", directions.shape, output_path)


def load_directions(path: str | Path) -> np.ndarray:
    return np.load(path)["directions"]
