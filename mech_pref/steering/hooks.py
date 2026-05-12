from __future__ import annotations

import contextlib
from typing import Generator

import numpy as np
import torch


class SteeringHook:
    """Forward hook that adds α * d_ℓ to the residual stream of a transformer layer."""

    def __init__(self, direction: torch.Tensor, alpha: float) -> None:
        self.direction = direction  # [d_model], normalized
        self.alpha = alpha

    def __call__(self, module, input, output):
        # transformers >= ~4.40: decoder layer returns hidden_states as a plain Tensor.
        # Older versions returned a tuple (hidden_states, attn_weights, past_kv).
        if isinstance(output, torch.Tensor):
            delta = self.alpha * self.direction.to(output.device, output.dtype)
            return output + delta
        else:
            hidden_states = output[0]  # [B, S, D]
            delta = self.alpha * self.direction.to(hidden_states.device, hidden_states.dtype)
            return (hidden_states + delta,) + output[1:]


@contextlib.contextmanager
def apply_steering_hooks(
    model,
    directions: np.ndarray,
    layer_indices: list[int],
    alpha: float,
) -> Generator:
    """Register activation-steering hooks at specified layers, then clean up.

    Args:
        model: LlamaForCausalLM (or compatible) — hooks target model.model.layers[i].
        directions: [n_act_positions, d_model] normalized direction array, where
            index 0 is the embedding output and index i (i>=1) is the output of
            transformer layer i-1. layer_indices must be in range [1, n_layers].
        layer_indices: activation-position indices to steer (1-indexed, matching
            the directions array). Maps to model.model.layers[i-1].
        alpha: steering coefficient.
    """
    hooks = []
    for act_idx in layer_indices:
        # act_idx is 1-based (1 = output of transformer layer 0)
        transformer_layer_idx = act_idx - 1
        direction = torch.from_numpy(directions[act_idx])
        hook = SteeringHook(direction, alpha)
        handle = model.model.layers[transformer_layer_idx].register_forward_hook(hook)
        hooks.append(handle)
    try:
        yield
    finally:
        for h in hooks:
            h.remove()
