from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlamaModel,
    LlamaPreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import ModelOutput

logger = logging.getLogger(__name__)

# Conversation template used during PKU safe-rlhf training
# (matches safe_rlhf PROMPT_BEGIN / PROMPT_USER / PROMPT_ASSISTANT constants)
_PKU_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{response}"

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclass
class ScoreModelOutput(ModelOutput):
    scores: torch.Tensor | None = None       # [B, S]
    end_scores: torch.Tensor | None = None   # [B]


class LlamaForScore(LlamaPreTrainedModel):
    """LLaMA with a scalar scoring head.

    Drop-in replacement for safe-rlhf AutoModelForScore that avoids the
    safe-rlhf dependency. Compatible with PKU-Alignment/beaver-7b-v1.0-cost
    and beaver-7b-v1.0-reward state dicts.

    Checkpoint key structure:
        model.*          — LlamaModel weights
        score_head.weight / score_head.bias  — scalar scoring linear layer
        normalizer.*     — running stats used during training only, ignored here
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        # Must be named score_head to match PKU checkpoint keys
        self.score_head = nn.Linear(config.hidden_size, 1, bias=True)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> ScoreModelOutput:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, S, D]
        scores = self.score_head(hidden_states).squeeze(-1)  # [B, S]

        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1) - 1  # [B]
            batch_idx = torch.arange(scores.shape[0], device=scores.device)
            end_scores = scores[batch_idx, seq_lens]
        else:
            end_scores = scores[:, -1]

        return ScoreModelOutput(scores=scores, end_scores=end_scores)


class CostModelScorer:
    """Scores (prompt, response) pairs using a PKU-Alignment cost/reward model.

    Higher cost score → more unsafe.
    Higher reward score → more preferred (for reward model variant).
    """

    def __init__(
        self,
        model_id: str,
        dtype: str = "bf16",
        device_map: str = "auto",
    ) -> None:
        logger.info("Loading cost model tokenizer: %s", model_id)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_id, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading cost model weights: %s (%s)", model_id, dtype)
        self.model = LlamaForScore.from_pretrained(
            model_id,
            torch_dtype=_DTYPE_MAP[dtype],
            device_map=device_map,
        )
        self.model.eval()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _format(self, prompt: str, response: str) -> str:
        return _PKU_TEMPLATE.format(prompt=prompt, response=response)

    def score(self, prompt: str, response: str) -> float:
        text = self._format(prompt, response)
        inputs = self.tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self.model(**inputs)
        return out.end_scores[0].item()

    def score_batch(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 4,
    ) -> np.ndarray:
        """Score a list of (prompt, response) pairs. Returns float array."""
        scores: list[float] = []
        for start in tqdm(range(0, len(pairs), batch_size), desc="Scoring"):
            batch = pairs[start : start + batch_size]
            texts = [self._format(p, r) for p, r in batch]

            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self._device)

            with torch.no_grad():
                out = self.model(**enc)
            scores.extend(out.end_scores.float().cpu().tolist())

        return np.array(scores, dtype=np.float32)
