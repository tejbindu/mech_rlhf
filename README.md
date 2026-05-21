# mech-pref

Mechanistic interpretability study of geometric changes induced by RLHF safety fine-tuning.

The project measures how the RLHF update from `alpaca-7b-reproduced` → `beaver-7b-v1.0` moves residual-stream activations, characterizes this as a per-layer "safety direction", and then steers the pre-RLHF model along that direction to replicate the safety behavior.

## Motivation

A prior study on Llama HH-RLHF found near-zero cosine alignment between the RLHF shift and the safety direction. This project uses a cleaner model pair — identical SFT base, differing only in RLHF — and the training data for that RLHF step (PKU-SafeRLHF), which gives a much sharper geometric signal.

## Models

| Role | Model |
|------|-------|
| Pre-RLHF | `PKU-Alignment/alpaca-7b-reproduced` |
| Post-RLHF | `PKU-Alignment/beaver-7b-v1.0` |
| Cost evaluator | `PKU-Alignment/beaver-7b-v1.0-cost` |

The cost/reward models are loaded via a custom `LlamaForScore` class (`mech_pref/evaluation/cost_model.py`) that avoids the `safe-rlhf` package dependency.

## Key Findings (Audit Phase)

- **`response_first` pooling**: weak signal (peak cosine 0.095, AUROC 0.595). Alpaca has no refusal behavior so safe/unsafe responses look identical at the first response token.
- **`prompt_last` pooling**: strong signal (mean cosine ~0.67 harmful, ~0.36 benign, AUROC 0.99).
- **Safety specificity**: harmful prompts align ~0.306 more with the safety direction than benign Alpaca prompts, confirming the direction captures safety geometry rather than generic model drift.
- **Embedding layer** (layer 0) shows the largest safety-specific gap (harmful=0.939, benign=0.092).
- **Top-8 layers by AUROC** selected for steering: `[4, 12, 13, 14, 15, 29, 30, 31]`.

## Installation

Dependencies are managed with [uv](https://github.com/astral-sh/uv). `torch` and `transformers` are installed separately by the SLURM scripts because different hardware requires different CUDA wheel indexes.

```bash
uv sync
# Install torch/transformers for your hardware, e.g. for CUDA 12.6:
uv pip install "torch>=2.11.0" --extra-index-url https://download.pytorch.org/whl/cu126
uv pip install "transformers>=5.8.0"
source .venv/bin/activate
```

Set `HF_TOKEN` in a `.env` file at the project root to authenticate with Hugging Face.

## Usage

All commands share the same `--config` flag. The default config is `configs/pku_alpaca.yaml`.

```
python -m mech_pref [--config PATH] COMMAND [OPTIONS]
```

### Commands

| Command | Description |
|---------|-------------|
| `extract-acts --model alpaca\|beaver` | Extract and cache residual-stream activations |
| `extract-dirs` | Compute mean-difference safety directions and per-layer AUROC |
| `run-audit` | Cosine alignment between the RLHF shift and the safety direction |
| `run-control` | Same audit on benign Alpaca prompts (safety-specificity check) |
| `run-steer --model alpaca\|beaver` | Generate responses with/without activation steering |
| `run-eval` | Score all generated responses with the cost model |

### Full audit pipeline

```bash
CFG=configs/pku_alpaca_prompt_last.yaml

python -m mech_pref --config $CFG extract-acts --model alpaca
python -m mech_pref --config $CFG extract-acts --model beaver
python -m mech_pref --config $CFG extract-dirs
python -m mech_pref --config $CFG run-audit
python -m mech_pref --config $CFG run-control
```

### Full steering pipeline

```bash
CFG=configs/pku_steer.yaml

python -m mech_pref --config $CFG run-steer --model alpaca   # baseline + steered
python -m mech_pref --config $CFG run-steer --model beaver   # RLHF upper-bound reference
python -m mech_pref --config $CFG run-eval
```

Pass `--alpha VALUE` to `run-steer` to run a single steering coefficient instead of the full sweep.

## Configuration

YAML configs are layered on top of the dataclass defaults in `mech_pref/config.py`. Only the fields you want to override need to be specified.

Key config sections:

```yaml
run:
  output_dir: /path/to/artifacts   # all outputs written here

data:
  harm_category: all               # "all" or a specific PKU-SafeRLHF category
  min_severity: 2                  # minimum harm severity for unsafe examples

activations:
  pooling: prompt_last             # prompt_last | response_first | response_generated

steering:
  alphas: [5.0, 10.0, 20.0, 40.0] # steering coefficients to sweep
  top_k_layers: 8                  # number of top-AUROC layers to steer
  max_new_tokens: 256
```

See `configs/` for ready-to-use examples.

## SLURM

SLURM scripts live in `../slurm/` relative to this directory.

| Script | Hardware | Purpose |
|--------|----------|---------|
| `mech_pref_audit_a100.sh` | A100 | Full audit pipeline |
| `mech_pref_audit_h100.sh` | H100 | Full audit pipeline |
| `mech_pref_audit_v100.sh` | V100 | Full audit pipeline |
| `mech_pref_steer_a100.sh` | A100 | Full steering + eval pipeline |

```bash
sbatch ../slurm/mech_pref_audit_a100.sh
# or override the config:
sbatch ../slurm/mech_pref_audit_a100.sh configs/pku_alpaca_prompt_last.yaml
```

## Artifacts

All outputs are written under the directory set by `run.output_dir`:

```
<output_dir>/
  cache/<pooling>/
    alpaca_safe.npz          # [n, n_layers, d_model] activations
    beaver_safe.npz
    directions.npz           # [n_layers, d_model] L2-normalized safety directions
    aurocs.npz               # [n_layers] per-layer AUROC
  plots/<pooling>/
    auroc_by_layer.png
    cosine_by_layer.png
    cosine_comparison.png    # harmful vs benign control
  steering/<pooling>/
    baseline.json            # alpaca, no steering
    alpha_5.0.json           # alpaca, steered at α=5
    ...
    beaver_reference.json    # beaver upper bound
  plots/steering/<pooling>/
    cost_vs_alpha.png
  audit_results_<pooling>.npz
  control_results_<pooling>.npz
  eval_results_steering.npz
```

## Package structure

```
mech_pref/
  config.py            # RunConfig, ModelConfig, DataConfig, … → Config
  scripts/cli.py       # Click CLI entry point
  activations/         # activation extraction and caching
  audit/geometric.py   # cosine audit and summary statistics
  probes/directions.py # mean-difference directions, AUROC computation
  steering/
    hooks.py           # SteeringHook, apply_steering_hooks context manager
    generate.py        # generate_responses, select_top_k_layers
  evaluation/
    cost_model.py      # LlamaForScore (custom), CostModelScorer
  data/                # PKU-SafeRLHF and Alpaca dataset loaders
  models/load.py       # model + tokenizer loading
  reporting/plots.py   # matplotlib plotting helpers
```
