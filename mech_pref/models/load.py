from __future__ import annotations

from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mech_pref.config import Config

MODEL_ALPACA = "PKU-Alignment/alpaca-7b-reproduced"
MODEL_BEAVER = "PKU-Alignment/beaver-7b-v1.0"

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def load_model_and_tokenizer(
    cfg: Config,
    model_tag: Literal["alpaca", "beaver"],
) -> tuple:
    """Load a causal LM and tokenizer according to config.

    model_tag selects which model to load: 'alpaca' (pre-RLHF) or 'beaver' (post-RLHF).
    dtype, device_map come from cfg.model.
    """
    model_id = cfg.model.alpaca_id if model_tag == "alpaca" else cfg.model.beaver_id
    torch_dtype = _DTYPE_MAP[cfg.model.dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=cfg.model.device_map,
    )
    model.eval()

    return model, tokenizer
