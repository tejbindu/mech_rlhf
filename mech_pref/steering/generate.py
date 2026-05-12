from __future__ import annotations

import contextlib
import logging

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from mech_pref.steering.hooks import apply_steering_hooks

logger = logging.getLogger(__name__)

# Alpaca instruction template used by alpaca-7b-reproduced
_ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def format_alpaca_prompt(instruction: str) -> str:
    return _ALPACA_TEMPLATE.format(instruction=instruction)


def select_top_k_layers(aurocs: np.ndarray, k: int) -> list[int]:
    """Return sorted activation-position indices of the top-K hookable layers by AUROC.

    Index 0 is the embedding output — there is no transformer layer to hook there,
    so it is excluded. Returned indices are in [1, n_layers] and map to
    model.model.layers[i-1] in apply_steering_hooks.
    """
    hookable = aurocs[1:]                         # drop embedding (index 0)
    top_k = hookable.argsort()[::-1][:k] + 1     # shift back to original indexing
    return sorted(top_k.tolist())


@contextlib.contextmanager
def _null_ctx():
    yield


def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    directions: np.ndarray,
    layer_indices: list[int],
    alpha: float,
    max_new_tokens: int = 256,
) -> list[str]:
    """Generate one response per prompt with activation steering applied.

    Prompts are expected to be raw instructions; they are wrapped in the Alpaca
    template internally. Pass alpha=0.0 to generate an unsteered baseline.

    Args:
        directions: [n_layers, d_model] normalized directions (from directions.npz).
        layer_indices: layers to steer (empty list → no steering).
        alpha: steering coefficient. 0.0 = unsteered baseline.

    Returns:
        List of decoded response strings (new tokens only, no prompt).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    steer = alpha != 0.0 and len(layer_indices) > 0
    desc = f"α={alpha}, layers={layer_indices}" if steer else "baseline (no steering)"
    responses: list[str] = []

    for instruction in tqdm(prompts, desc=f"Generating [{desc}]"):
        formatted = format_alpaca_prompt(instruction)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        ctx = apply_steering_hooks(model, directions, layer_indices, alpha) if steer else _null_ctx()
        with ctx, torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return responses
