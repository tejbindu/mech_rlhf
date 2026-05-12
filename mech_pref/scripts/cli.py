from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import numpy as np

from mech_pref.config import load_config
from mech_pref.data.pku_beaver import load_pku_pairs, load_pku_prompts
from mech_pref.data.alpaca import load_alpaca_prompts
from mech_pref.models.load import load_model_and_tokenizer
from mech_pref.activations.extract import extract_and_save, generate_and_extract
from mech_pref.probes.directions import (
    compute_auroc_per_layer,
    compute_directions,
    load_directions,
    save_directions,
)
from mech_pref.audit.geometric import compute_cosine_audit, summarize
from mech_pref.reporting.plots import (
    plot_auroc_by_layer,
    plot_cosine_by_layer,
    plot_cosine_comparison,
    plot_cost_vs_alpha,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--config",
    "config_path",
    default="configs/pku_alpaca.yaml",
    show_default=True,
    help="Path to YAML config file.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str) -> None:
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config(config_path)


# ---------------------------------------------------------------------------
# extract-acts
# ---------------------------------------------------------------------------

@cli.command("extract-acts")
@click.option(
    "--model",
    type=click.Choice(["alpaca", "beaver"]),
    required=True,
    help="Which model to run. 'alpaca' extracts both safe and unsafe; 'beaver' extracts safe only.",
)
@click.pass_context
def extract_acts(ctx: click.Context, model: str) -> None:
    """Extract activations and save to artifacts/cache/<pooling>/."""
    cfg = ctx.obj["cfg"]
    pooling = cfg.activations.pooling
    cache_dir = Path(cfg.run.output_dir) / "cache" / pooling

    # response_generated uses {model}_generated.npz; others use {model}_safe/unsafe.npz
    if pooling == "response_generated":
        gen_path = cache_dir / f"{model}_generated.npz"
        if gen_path.exists():
            logger.info("SKIP extract-acts %s: cache exists (%s)", model, gen_path)
            return
        logger.info("Loading PKU harmful prompts for generation (%s, harm=%s)…",
                    cfg.data.subset, cfg.data.harm_category)
        examples = load_pku_prompts(
            n_samples=cfg.data.n_samples,
            subset=cfg.data.subset,
            harm_category=cfg.data.harm_category,
            min_severity=cfg.data.min_severity,
            seed=cfg.data.seed,
        )
        logger.info("Loaded %d prompts", len(examples))
        logger.info("Loading %s model (%s)…", model, cfg.model.dtype)
        m, tok = load_model_and_tokenizer(cfg, model)
        generate_and_extract(m, tok, examples, gen_path, cfg)
        return

    safe_path = cache_dir / f"{model}_safe.npz"
    unsafe_path = cache_dir / f"{model}_unsafe.npz"
    need_safe = not safe_path.exists()
    # Extract unsafe only for response_first and only for the direction source model.
    need_unsafe = (
        pooling == "response_first"
        and model == cfg.data.direction_model
        and not unsafe_path.exists()
    )

    if not need_safe and not need_unsafe:
        logger.info("SKIP extract-acts %s: caches already exist (%s)", model, cache_dir)
        return

    logger.info("Loading %s model (%s, pooling=%s)…", model, cfg.model.dtype, pooling)
    m, tok = load_model_and_tokenizer(cfg, model)

    if pooling == "prompt_last":
        if need_safe:
            logger.info(
                "Loading PKU harmful prompts (%s, harm=%s, min_sev=%d)…",
                cfg.data.subset, cfg.data.harm_category, cfg.data.min_severity,
            )
            examples = load_pku_prompts(
                n_samples=cfg.data.n_samples,
                subset=cfg.data.subset,
                harm_category=cfg.data.harm_category,
                min_severity=cfg.data.min_severity,
                seed=cfg.data.seed,
            )
            logger.info("Loaded %d prompts", len(examples))
            extract_and_save(m, tok, examples, safe_path, cfg)
        else:
            logger.info("SKIP activations for %s: cache exists", model)
    else:
        # response_first: needs clean safe/unsafe pairs.
        logger.info(
            "Loading PKU-SafeRLHF pairs (%s, harm=%s)…",
            cfg.data.subset, cfg.data.harm_category,
        )
        pairs = load_pku_pairs(
            n_samples=cfg.data.n_samples,
            subset=cfg.data.subset,
            harm_category=cfg.data.harm_category,
            min_severity=cfg.data.min_severity,
            safe_max_severity=cfg.data.safe_max_severity,
            seed=cfg.data.seed,
        )
        logger.info("Loaded %d pairs", len(pairs))

        if need_safe:
            safe_examples = [{"prompt": p["prompt"], "response": p["safe_response"]} for p in pairs]
            logger.info("Extracting safe activations for %s…", model)
            extract_and_save(m, tok, safe_examples, safe_path, cfg)
        else:
            logger.info("SKIP safe activations for %s: cache exists", model)

        if need_unsafe:
            unsafe_examples = [{"prompt": p["prompt"], "response": p["unsafe_response"]} for p in pairs]
            logger.info("Extracting unsafe activations for %s…", model)
            extract_and_save(m, tok, unsafe_examples, unsafe_path, cfg)


# ---------------------------------------------------------------------------
# extract-dirs
# ---------------------------------------------------------------------------

@cli.command("extract-dirs")
@click.pass_context
def extract_dirs(ctx: click.Context) -> None:
    """Compute mean-difference directions and probe AUROC.

    response_first: direction = mean(alpaca_safe) - mean(alpaca_unsafe)
                    AUROC = can probe distinguish safe vs unsafe responses?
    prompt_last:    direction = mean(beaver_safe) - mean(alpaca_safe)  [RLHF shift]
                    AUROC = can probe distinguish beaver vs alpaca prompts?
    """
    cfg = ctx.obj["cfg"]
    pooling = cfg.activations.pooling
    cache_dir = Path(cfg.run.output_dir) / "cache" / pooling

    directions_path = cache_dir / "directions.npz"
    if directions_path.exists():
        logger.info("SKIP extract-dirs: directions already exist (%s)", directions_path)
        return

    if pooling == "response_generated":
        # Direction = mean(h_beaver_generated) − mean(h_alpaca_generated)
        # beaver naturally refuses → "safe"; alpaca naturally complies → "unsafe"
        pos_path, neg_path = cache_dir / "beaver_generated.npz", cache_dir / "alpaca_generated.npz"
        pos_label, neg_label = "beaver_generated", "alpaca_generated"
    elif pooling == "response_first":
        dm = cfg.data.direction_model
        pos_path, neg_path = cache_dir / f"{dm}_safe.npz", cache_dir / f"{dm}_unsafe.npz"
        pos_label, neg_label = f"{dm}_safe", f"{dm}_unsafe"
    else:
        pos_path, neg_path = cache_dir / "beaver_safe.npz", cache_dir / "alpaca_safe.npz"
        pos_label, neg_label = "beaver_safe", "alpaca_safe"

    for p in (pos_path, neg_path):
        if not p.exists():
            logger.error("Missing: %s — run extract-acts first.", p)
            sys.exit(1)

    pos_acts = np.load(pos_path)["activations"]
    neg_acts = np.load(neg_path)["activations"]
    logger.info("Loaded activations: %s=%s  %s=%s", pos_label, pos_acts.shape, neg_label, neg_acts.shape)

    logger.info("Computing mean-difference directions…")
    directions = compute_directions(pos_acts, neg_acts)
    save_directions(directions, directions_path)

    logger.info("Computing probe AUROC per layer…")
    aurocs = compute_auroc_per_layer(pos_acts, neg_acts, cfg)

    plots_dir = Path(cfg.run.output_dir) / "plots" / pooling
    plot_auroc_by_layer(aurocs, plots_dir / "auroc_by_layer.png")

    _print_layer_table("Probe AUROC", aurocs, fmt=".3f")
    logger.info("Peak AUROC layer: %d  (%.3f)", int(aurocs.argmax()), aurocs.max())

    # Save aurocs alongside directions for later reference
    np.savez(cache_dir / "aurocs.npz", aurocs=aurocs)


# ---------------------------------------------------------------------------
# run-audit
# ---------------------------------------------------------------------------

@cli.command("run-audit")
@click.pass_context
def run_audit(ctx: click.Context) -> None:
    """Compute per-layer cosine alignment between RLHF shift and safety direction."""
    cfg = ctx.obj["cfg"]
    cache_dir = Path(cfg.run.output_dir) / "cache" / cfg.activations.pooling

    pooling = cfg.activations.pooling
    if pooling == "response_generated":
        alpaca_key, beaver_key = "alpaca_generated", "beaver_generated"
    else:
        alpaca_key, beaver_key = "alpaca_safe", "beaver_safe"

    required = {
        alpaca_key: cache_dir / f"{alpaca_key}.npz",
        beaver_key: cache_dir / f"{beaver_key}.npz",
        "directions": cache_dir / "directions.npz",
    }
    for name, p in required.items():
        if not p.exists():
            logger.error("Missing: %s (%s)", name, p)
            sys.exit(1)

    alpaca_acts = np.load(required[alpaca_key])["activations"]
    beaver_acts = np.load(required[beaver_key])["activations"]
    directions = load_directions(required["directions"])
    logger.info(
        "Loaded  alpaca=%s  beaver=%s  directions=%s",
        alpaca_acts.shape, beaver_acts.shape, directions.shape,
    )

    if alpaca_acts.shape[0] != beaver_acts.shape[0]:
        logger.error(
            "Shape mismatch: alpaca has %d examples, beaver has %d. "
            "Re-run extract-acts with the same config for both models.",
            alpaca_acts.shape[0], beaver_acts.shape[0],
        )
        sys.exit(1)

    if pooling in ("prompt_last", "response_generated"):
        # For these modes the direction is mean(beaver) - mean(alpaca) on the same
        # examples as the audit — making it circular. Fix: compute direction on one
        # half and evaluate on the held-out half.
        n = alpaca_acts.shape[0]
        n_dir = n // 2
        rng = np.random.default_rng(cfg.run.seed)
        idx = rng.permutation(n)
        dir_idx, eval_idx = idx[:n_dir], idx[n_dir:]

        directions = compute_directions(beaver_acts[dir_idx], alpaca_acts[dir_idx])
        alpaca_acts = alpaca_acts[eval_idx]
        beaver_acts = beaver_acts[eval_idx]
        logger.info(
            "%s: direction from %d examples, audit on %d held-out examples",
            pooling, n_dir, len(eval_idx),
        )

    logger.info("Running geometric audit…")
    cosines = compute_cosine_audit(alpaca_acts, beaver_acts, directions)
    summary = summarize(cosines)

    plots_dir = Path(cfg.run.output_dir) / "plots" / pooling
    plot_cosine_by_layer(summary, plots_dir / "cosine_by_layer.png")

    _print_layer_table("Mean cosine", summary["mean"], fmt="+.3f")
    peak = int(summary["mean"].argmax())
    logger.info(
        "Peak alignment: layer %d  mean=%.3f  std=%.3f  IQR=[%.3f, %.3f]",
        peak,
        summary["mean"][peak],
        summary["std"][peak],
        summary["q25"][peak],
        summary["q75"][peak],
    )

    np.savez(
        Path(cfg.run.output_dir) / f"audit_results_{pooling}.npz",
        cosines=cosines,
        **{f"summary_{k}": v for k, v in summary.items()},
    )


# ---------------------------------------------------------------------------
# run-control
# ---------------------------------------------------------------------------

@cli.command("run-control")
@click.pass_context
def run_control(ctx: click.Context) -> None:
    """Control experiment: measure safety direction alignment on benign prompts.

    Loads Alpaca benign prompts, extracts prompt_last activations from both
    models, and measures cosine alignment with the safety direction extracted
    from harmful PKU prompts. Generates a comparison plot.

    If the safety direction is safety-specific, harmful cosines should be
    clearly higher than benign cosines. If both are similar, the direction
    captures generic model drift rather than safety geometry.
    """
    cfg = ctx.obj["cfg"]
    pooling = cfg.activations.pooling
    cache_dir = Path(cfg.run.output_dir) / "cache" / pooling
    plots_dir = Path(cfg.run.output_dir) / "plots" / pooling

    # Benign activation caches
    alpaca_benign_path = cache_dir / "alpaca_benign.npz"
    beaver_benign_path = cache_dir / "beaver_benign.npz"

    # Match benign count to the harmful set that was actually extracted
    harmful_cache = cache_dir / "alpaca_safe.npz"
    if harmful_cache.exists():
        n_benign = int(np.load(harmful_cache)["activations"].shape[0])
    else:
        n_benign = cfg.data.n_samples or 2811
    logger.info("Loading %d Alpaca benign prompts (matching harmful set size)…", n_benign)
    benign_examples = load_alpaca_prompts(n_samples=n_benign, seed=cfg.data.seed)
    logger.info("Loaded %d benign prompts", len(benign_examples))

    # Extract benign activations if not cached — free each model explicitly before
    # loading the next one to avoid OOM on memory-constrained GPUs (e.g. V100 32GB).
    import gc
    import torch

    for model_tag, cache_path in [("alpaca", alpaca_benign_path), ("beaver", beaver_benign_path)]:
        if cache_path.exists():
            logger.info("SKIP benign extract for %s: cache exists", model_tag)
        else:
            logger.info("Loading %s model (%s) for benign extraction…", model_tag, cfg.model.dtype)
            m, tok = load_model_and_tokenizer(cfg, model_tag)
            extract_and_save(m, tok, benign_examples, cache_path, cfg)
            del m, tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Load directions from harmful prompts (already computed by extract-dirs)
    directions_path = cache_dir / "directions.npz"
    if not directions_path.exists():
        logger.error("Missing directions — run extract-dirs first.")
        sys.exit(1)

    # Use same held-out direction split as run-audit for a fair comparison.
    # Load harmful arrays, compute direction, then free them before loading benign
    # to avoid holding all four large arrays in RAM simultaneously.
    if pooling == "response_generated":
        alpaca_key, beaver_key = "alpaca_generated", "beaver_generated"
    else:
        alpaca_key, beaver_key = "alpaca_safe", "beaver_safe"
    alpaca_harmful = np.load(cache_dir / f"{alpaca_key}.npz")["activations"]
    beaver_harmful = np.load(cache_dir / f"{beaver_key}.npz")["activations"]
    n = alpaca_harmful.shape[0]
    rng = np.random.default_rng(cfg.run.seed)
    idx = rng.permutation(n)
    dir_idx = idx[: n // 2]
    directions = compute_directions(beaver_harmful[dir_idx], alpaca_harmful[dir_idx])
    del alpaca_harmful, beaver_harmful  # free ~15GB before loading benign arrays

    # Audit on benign examples
    alpaca_benign = np.load(alpaca_benign_path)["activations"]
    beaver_benign = np.load(beaver_benign_path)["activations"]
    logger.info(
        "Benign activations  alpaca=%s  beaver=%s",
        alpaca_benign.shape, beaver_benign.shape,
    )
    cosines_benign = compute_cosine_audit(alpaca_benign, beaver_benign, directions)
    summary_benign = summarize(cosines_benign)

    # Load harmful summary from saved audit results for comparison
    harmful_results_path = Path(cfg.run.output_dir) / f"audit_results_{pooling}.npz"
    if not harmful_results_path.exists():
        logger.error("Missing harmful audit results — run run-audit first.")
        sys.exit(1)
    harmful_data = np.load(harmful_results_path)
    summary_harmful = {k.replace("summary_", ""): harmful_data[k] for k in harmful_data if k.startswith("summary_")}

    # Comparison plot
    plot_cosine_comparison(summary_harmful, summary_benign, plots_dir / "cosine_comparison.png")

    # Print side-by-side table
    click.echo(f"\n{'Layer':>6}  {'Harmful':>10}  {'Benign':>10}  {'Diff':>8}")
    click.echo("-" * 40)
    for i, (h, b) in enumerate(zip(summary_harmful["mean"], summary_benign["mean"])):
        click.echo(f"{i:>6}  {h:>+10.3f}  {b:>+10.3f}  {h - b:>+8.3f}")
    click.echo()

    logger.info(
        "Mean cosine — harmful: %.3f  benign: %.3f  diff: %.3f",
        summary_harmful["mean"].mean(),
        summary_benign["mean"].mean(),
        (summary_harmful["mean"] - summary_benign["mean"]).mean(),
    )

    np.savez(
        Path(cfg.run.output_dir) / f"control_results_{pooling}.npz",
        cosines=cosines_benign,
        **{f"summary_{k}": v for k, v in summary_benign.items()},
    )


# ---------------------------------------------------------------------------
# run-steer
# ---------------------------------------------------------------------------

@cli.command("run-steer")
@click.option(
    "--model",
    type=click.Choice(["alpaca", "beaver"]),
    default="alpaca",
    show_default=True,
    help="'alpaca' generates baseline + steered responses; 'beaver' generates the RLHF upper-bound reference.",
)
@click.option(
    "--alpha",
    type=float,
    default=None,
    help="Run a single alpha value instead of the full sweep from config. Implies baseline is also generated.",
)
@click.pass_context
def run_steer(ctx: click.Context, model: str, alpha: float | None) -> None:
    """Generate responses for the steering experiment.

    alpaca: produces one JSON per alpha value (steered) plus a baseline JSON.
    beaver: produces a single reference JSON for upper-bound comparison.

    Outputs go to <output_dir>/steering/.
    Use --alpha to override the config sweep with a single value (useful for quick tests).
    """
    from mech_pref.steering.generate import generate_responses, select_top_k_layers

    cfg = ctx.obj["cfg"]
    cache_dir = Path(cfg.run.output_dir) / "cache" / cfg.activations.pooling
    steer_dir = Path(cfg.run.output_dir) / "steering" / cfg.activations.pooling
    steer_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    logger.info("Loading PKU harmful prompts (harm=%s)…", cfg.data.harm_category)
    pairs = load_pku_pairs(
        n_samples=cfg.steering.n_prompts,
        subset=cfg.data.subset,
        harm_category=cfg.data.harm_category,
        min_severity=cfg.data.min_severity,
        safe_max_severity=cfg.data.safe_max_severity,
        seed=cfg.data.seed,
    )
    prompts = [p["prompt"] for p in pairs]
    logger.info("Loaded %d prompts", len(prompts))

    # Load directions and AUROCs
    directions_path = cache_dir / "directions.npz"
    aurocs_path = cache_dir / "aurocs.npz"
    for p in (directions_path, aurocs_path):
        if not p.exists():
            logger.error("Missing: %s — run extract-dirs first.", p)
            sys.exit(1)

    directions = load_directions(directions_path)
    aurocs = np.load(aurocs_path)["aurocs"]
    layer_indices = select_top_k_layers(aurocs, cfg.steering.top_k_layers)
    logger.info("Steering layers (top-%d by AUROC): %s", cfg.steering.top_k_layers, layer_indices)

    # Load the generation model
    logger.info("Loading %s model…", model)
    gen_model, tokenizer = load_model_and_tokenizer(cfg, model)

    if model == "beaver":
        # Single reference run (no steering)
        out_path = steer_dir / "beaver_reference.json"
        if out_path.exists():
            logger.info("SKIP beaver reference: %s already exists", out_path)
        else:
            from mech_pref.steering.generate import format_alpaca_prompt
            import torch
            from tqdm import tqdm

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            responses = []
            for instruction in tqdm(prompts, desc="Generating [beaver reference]"):
                formatted = format_alpaca_prompt(instruction)
                inputs = tokenizer(formatted, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = gen_model.generate(
                        **inputs,
                        max_new_tokens=cfg.steering.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
                responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

            out_path.write_text(json.dumps({
                "model": "beaver",
                "alpha": None,
                "layer_indices": None,
                "responses": [{"prompt": p, "response": r} for p, r in zip(prompts, responses)],
            }, indent=2))
            logger.info("Saved beaver reference → %s", out_path)

    else:
        # Alpaca: baseline + steered alphas
        # --alpha flag overrides the config sweep to a single value
        alphas_to_run = [0.0] + ([alpha] if alpha is not None else list(cfg.steering.alphas))
        for alpha in alphas_to_run:
            label = "baseline" if alpha == 0.0 else f"alpha_{alpha}"
            out_path = steer_dir / f"{label}.json"

            if out_path.exists():
                logger.info("SKIP %s: %s already exists", label, out_path)
                continue

            layers = [] if alpha == 0.0 else layer_indices
            responses = generate_responses(
                gen_model, tokenizer, prompts, directions, layers, alpha,
                max_new_tokens=cfg.steering.max_new_tokens,
            )

            out_path.write_text(json.dumps({
                "model": "alpaca",
                "alpha": alpha,
                "layer_indices": layer_indices if alpha != 0.0 else [],
                "responses": [{"prompt": p, "response": r} for p, r in zip(prompts, responses)],
            }, indent=2))
            logger.info("Saved %s → %s  (%d responses)", label, out_path, len(responses))


# ---------------------------------------------------------------------------
# run-eval
# ---------------------------------------------------------------------------

@cli.command("run-eval")
@click.pass_context
def run_eval(ctx: click.Context) -> None:
    """Score all generated responses with the cost model and report results.

    Reads JSONs from <output_dir>/steering/, scores each with beaver-7b-v1.0-cost,
    and writes eval_results.npz + a cost-vs-alpha plot.

    Expects run-steer to have been run first for both 'alpaca' and 'beaver'.
    """
    from mech_pref.evaluation.cost_model import CostModelScorer

    cfg = ctx.obj["cfg"]
    steer_dir = Path(cfg.run.output_dir) / "steering" / cfg.activations.pooling
    plots_dir = Path(cfg.run.output_dir) / "plots" / "steering" / cfg.activations.pooling
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not steer_dir.exists():
        logger.error("Steering directory not found: %s — run run-steer first.", steer_dir)
        sys.exit(1)

    # Collect all JSON files
    json_files = sorted(steer_dir.glob("*.json"))
    if not json_files:
        logger.error("No response JSONs found in %s.", steer_dir)
        sys.exit(1)
    logger.info("Found %d response files: %s", len(json_files), [f.name for f in json_files])

    # Load scorer once (expensive: 7B model)
    scorer = CostModelScorer(
        cfg.eval.cost_model_id,
        dtype=cfg.model.dtype,
        device_map=cfg.model.device_map,
    )

    results: dict[str, np.ndarray] = {}
    for jf in json_files:
        data = json.loads(jf.read_text())
        pairs = [(r["prompt"], r["response"]) for r in data["responses"]]
        alpha = data["alpha"]
        label = "baseline" if alpha == 0.0 else ("beaver" if data["model"] == "beaver" else str(alpha))

        logger.info("Scoring %s (%d pairs)…", jf.name, len(pairs))
        costs = scorer.score_batch(pairs, batch_size=cfg.eval.batch_size)
        results[label] = costs
        logger.info(
            "%s  mean_cost=%.4f  pct_unsafe=%.1f%%",
            label,
            costs.mean(),
            100 * (costs > cfg.eval.cost_threshold).mean(),
        )

    # Save raw scores
    np.savez(Path(cfg.run.output_dir) / "eval_results_steering.npz", **results)
    logger.info("Saved eval_results_steering.npz")

    # Compute summary stats
    numeric_alphas = sorted(
        [float(k) for k in results if k not in ("baseline", "beaver")]
    )
    mean_costs = {k: float(v.mean()) for k, v in results.items()}
    pct_unsafe = {
        k: float((v > cfg.eval.cost_threshold).mean())
        for k, v in results.items()
    }

    # Print table
    click.echo(f"\n{'Condition':>16}  {'Mean cost':>12}  {'% unsafe':>10}")
    click.echo("-" * 44)
    for label in (["baseline"] + [str(a) for a in numeric_alphas] + ["beaver"]):
        if label not in mean_costs:
            continue
        click.echo(
            f"{label:>16}  {mean_costs[label]:>+12.4f}  {100 * pct_unsafe[label]:>9.1f}%"
        )
    click.echo()

    # Plot
    plot_cost_vs_alpha(
        alphas=numeric_alphas,
        mean_costs=mean_costs,
        pct_unsafe=pct_unsafe,
        save_path=plots_dir / "cost_vs_alpha.png",
    )
    logger.info("Plot saved → %s", plots_dir / "cost_vs_alpha.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_layer_table(label: str, values: np.ndarray, fmt: str = ".3f") -> None:
    """Print a compact per-layer table to stdout."""
    peak = int(values.argmax())
    click.echo(f"\n{'Layer':>6}  {label}")
    click.echo("-" * 24)
    for i, v in enumerate(values):
        marker = " ◀ peak" if i == peak else ""
        click.echo(f"{i:>6}  {v:{fmt}}{marker}")
    click.echo()
