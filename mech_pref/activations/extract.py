from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from mech_pref.config import Config

logger = logging.getLogger(__name__)


def _tokenize_pair(
    tokenizer, prompt: str, response: str, max_length: int
) -> tuple[list[int], int] | None:
    """Return (full_token_ids, prompt_len), or None if prompt alone exceeds max_length."""
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids

    prompt_len = len(prompt_ids)
    remaining = max_length - prompt_len
    if remaining < 1:
        return None

    full_ids = prompt_ids + response_ids[:remaining]
    # For response_first pooling we extract at index prompt_len, which requires
    # at least one response token to be present in the sequence.
    if len(full_ids) <= prompt_len:
        return None

    return full_ids, prompt_len


def _pad_batch(
    sequences: list[list[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of token-id lists. Returns (input_ids, attention_mask)."""
    max_len = max(len(s) for s in sequences)
    B = len(sequences)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    for i, s in enumerate(sequences):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attention_mask[i, : len(s)] = 1
    return input_ids, attention_mask


def _input_device() -> torch.device:
    """Return the device to place inputs on. cuda:0 with device_map=auto, else cpu."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_and_save(
    model,
    tokenizer,
    examples: Sequence[dict],
    output_path: str | Path,
    cfg: Config,
) -> np.ndarray:
    """Forward-pass (prompt, response) pairs and save per-layer hidden states.

    Pooling is controlled by cfg.activations.pooling:
      - "response_first": hidden state at index prompt_len (first generated token)
      - "prompt_last":    hidden state at index prompt_len - 1 (last prompt token)
        For prompt_last, the response is included in the input but is causally
        masked from the extraction position, so only the prompt content matters.

    Args:
        examples: list of {"prompt": str, "response": str}
        output_path: .npz save path (parent dirs created automatically)
        cfg: experiment config (activations.max_length, activations.batch_size,
             activations.pooling)

    Returns:
        activations array of shape [n_examples, n_layers, d_model], float32.
        Also saved to output_path as activations.npz with key "activations".
    """
    pooling = cfg.activations.pooling
    pad_id = tokenizer.pad_token_id
    device = _input_device()

    # Tokenize all pairs upfront, skip any where the prompt alone fills max_length
    tokenized: list[tuple[list[int], int]] = []
    skipped = 0
    for ex in examples:
        result = _tokenize_pair(
            tokenizer, ex["prompt"], ex["response"], cfg.activations.max_length
        )
        if result is None:
            skipped += 1
        else:
            tokenized.append(result)

    if skipped:
        logger.warning(
            "%d examples skipped: prompt length >= max_length=%d",
            skipped,
            cfg.activations.max_length,
        )

    all_acts: list[np.ndarray] = []
    batch_size = cfg.activations.batch_size

    for start in tqdm(range(0, len(tokenized), batch_size), desc="Extracting"):
        batch = tokenized[start : start + batch_size]
        seqs, prompt_lens = zip(*batch)

        input_ids, attention_mask = _pad_batch(list(seqs), pad_id)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # outputs.hidden_states: tuple of n_layers tensors, each [B, S, D]
        for i, prompt_len in enumerate(prompt_lens):
            pos = prompt_len if pooling == "response_first" else prompt_len - 1
            acts = torch.stack(
                [hs[i, pos, :] for hs in outputs.hidden_states],
                dim=0,
            )  # [n_layers, D]
            all_acts.append(acts.float().cpu().numpy())

    activations = np.stack(all_acts, axis=0)  # [N, n_layers, D]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, activations=activations)
    logger.info("Saved activations %s → %s", activations.shape, output_path)

    return activations


def generate_and_extract(
    model,
    tokenizer,
    examples: Sequence[dict],
    output_path: str | Path,
    cfg: Config,
) -> np.ndarray:
    """Generate exactly one token per prompt and extract hidden states at that position.

    Unlike teacher-forcing, each model generates its own first response token
    naturally (greedy decoding). This gives genuine in-distribution hidden states:
    beaver produces refusal tokens ("I", "Sorry"...) and alpaca produces
    compliance tokens ("Sure", "Here"...) for the same harmful prompts.

    Args:
        examples: list of {"prompt": str} — response field is ignored.
        output_path: .npz save path.
        cfg: experiment config (activations.max_length).

    Returns:
        activations: [n_examples, n_layers, d_model] float32.
    """
    device = _input_device()
    all_acts: list[np.ndarray] = []
    skipped = 0

    for ex in tqdm(examples, desc="Generating"):
        prompt_ids = tokenizer(ex["prompt"], add_special_tokens=True).input_ids
        if len(prompt_ids) >= cfg.activations.max_length:
            skipped += 1
            continue

        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids,
                max_new_tokens=1,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # gen_out.hidden_states: tuple of length n_new_tokens (=1), each element
        # is a tuple of n_layers tensors with shape [batch, 1, d_model]
        if not gen_out.hidden_states:
            skipped += 1
            continue

        step_hs = gen_out.hidden_states[0]  # first (only) generation step
        acts = torch.stack(
            [hs[0, 0, :] for hs in step_hs],
            dim=0,
        ).float().cpu().numpy()  # [n_layers, d_model]
        all_acts.append(acts)

    if skipped:
        logger.warning("%d examples skipped (prompt too long or empty generation)", skipped)

    activations = np.stack(all_acts, axis=0)  # [N, n_layers, D]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, activations=activations)
    logger.info("Saved activations %s → %s", activations.shape, output_path)

    return activations

    return activations
