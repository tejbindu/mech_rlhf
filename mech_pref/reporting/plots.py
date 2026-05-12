from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cosine_by_layer(
    summary: dict[str, np.ndarray],
    save_path: str | Path,
    title: str = "RLHF shift alignment with safety direction",
) -> None:
    """Line plot of mean cosine ± std across layers, with IQR shading.

    Args:
        summary: output of geometric.summarize() — keys mean, std, q25, q75.
        save_path: where to write the PNG.
    """
    layers = np.arange(len(summary["mean"]))
    mean, std = summary["mean"], summary["std"]
    q25, q75 = summary["q25"], summary["q75"]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.fill_between(layers, q25, q75, alpha=0.25, label="IQR (Q25–Q75)")
    ax.fill_between(layers, mean - std, mean + std, alpha=0.15, label="±std")
    ax.plot(layers, mean, linewidth=1.8, label="Mean cosine")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(title)
    ax.set_xlim(layers[0], layers[-1])
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()

    _save(fig, save_path)


def plot_cosine_comparison(
    harmful_summary: dict[str, np.ndarray],
    benign_summary: dict[str, np.ndarray],
    save_path: str | Path,
) -> None:
    """Overlay harmful and benign mean cosines on the same axes.

    If the safety direction is safety-specific, harmful cosines should be
    clearly higher than benign. If both are similar, the direction captures
    generic model drift rather than safety geometry.
    """
    layers = np.arange(len(harmful_summary["mean"]))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.fill_between(
        layers,
        harmful_summary["mean"] - harmful_summary["std"],
        harmful_summary["mean"] + harmful_summary["std"],
        alpha=0.15, color="tab:blue",
    )
    ax.plot(layers, harmful_summary["mean"], linewidth=1.8, color="tab:blue", label="Harmful prompts (PKU)")

    ax.fill_between(
        layers,
        benign_summary["mean"] - benign_summary["std"],
        benign_summary["mean"] + benign_summary["std"],
        alpha=0.15, color="tab:orange",
    )
    ax.plot(layers, benign_summary["mean"], linewidth=1.8, color="tab:orange", label="Benign prompts (Alpaca)")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Safety direction alignment: harmful vs benign prompts")
    ax.set_xlim(layers[0], layers[-1])
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()

    _save(fig, save_path)


def plot_cost_vs_alpha(
    alphas: list[float],
    mean_costs: dict[str, float],
    pct_unsafe: dict[str, float],
    save_path: str | Path,
) -> None:
    """Two-panel plot: mean cost and % unsafe vs steering coefficient α.

    Args:
        alphas: list of α values (x-axis). "baseline" and "beaver" are
                handled as special labels; numeric α values are plotted in order.
        mean_costs: mapping from condition label → mean cost score.
        pct_unsafe: mapping from condition label → fraction of responses where
                    cost > threshold.
        save_path: where to write the PNG.
    """
    numeric = sorted([a for a in alphas if isinstance(a, (int, float))])
    x = list(range(len(numeric)))
    labels = [str(a) for a in numeric]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    def _val(d: dict, alpha) -> float | None:
        key = str(alpha) if not isinstance(alpha, str) else alpha
        return d.get(key)

    y_cost = [_val(mean_costs, a) for a in numeric]
    y_unsafe = [_val(pct_unsafe, a) for a in numeric]

    for ax, y, ylabel, title in [
        (ax1, y_cost, "Mean cost score", "Mean cost vs α"),
        (ax2, y_unsafe, "% unsafe (cost > threshold)", "% unsafe vs α"),
    ]:
        ax.plot(x, y, marker="o", linewidth=1.8, label="Steered (alpaca)")

        if "baseline" in mean_costs:
            bv = _val(mean_costs if ax is ax1 else pct_unsafe, "baseline")
            if bv is not None:
                ax.axhline(bv, color="tab:orange", linestyle="--", linewidth=1.2, label="Alpaca baseline")

        if "beaver" in mean_costs:
            bv = _val(mean_costs if ax is ax1 else pct_unsafe, "beaver")
            if bv is not None:
                ax.axhline(bv, color="tab:green", linestyle=":", linewidth=1.2, label="Beaver (RLHF)")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("α (steering coefficient)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)


def plot_auroc_by_layer(
    aurocs: np.ndarray,
    save_path: str | Path,
    title: str = "Linear probe AUROC per layer (safe vs unsafe)",
) -> None:
    """Line plot of probe AUROC across layers.

    Args:
        aurocs: [n_layers] array of held-out AUROC values.
        save_path: where to write the PNG.
    """
    layers = np.arange(len(aurocs))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(layers, aurocs, linewidth=1.8, marker="o", markersize=3)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="Chance (0.5)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.set_xlim(layers[0], layers[-1])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()

    _save(fig, save_path)
