from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


DType = Literal["bf16", "fp16", "fp32"]
Pooling = Literal["response_first", "prompt_last", "response_generated"]
SteerModel = Literal["alpaca", "beaver"]


@dataclass(frozen=True)
class RunConfig:
    output_dir: str = "artifacts"
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    alpaca_id: str = "PKU-Alignment/alpaca-7b-reproduced"
    beaver_id: str = "PKU-Alignment/beaver-7b-v1.0"
    dtype: DType = "bf16"
    device_map: str = "auto"


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "PKU-Alignment/PKU-SafeRLHF"
    subset: str = "alpaca-7b"
    harm_category: str = "all"
    min_severity: int = 2
    safe_max_severity: int = 0
    n_samples: int | None = None
    seed: int = 42
    direction_model: SteerModel = "alpaca"  # which model's safe/unsafe activations define the direction


@dataclass(frozen=True)
class ActivationConfig:
    max_length: int = 2048
    batch_size: int = 4
    pooling: Pooling = "response_first"


@dataclass(frozen=True)
class ProbeConfig:
    test_size: float = 0.2
    max_iter: int = 1000
    n_jobs: int = 4


@dataclass(frozen=True)
class SteeringConfig:
    alphas: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)
    top_k_layers: int = 8
    max_new_tokens: int = 256
    n_prompts: int | None = None  # None = all qualifying prompts


@dataclass(frozen=True)
class EvalConfig:
    cost_model_id: str = "PKU-Alignment/beaver-7b-v1.0-cost"
    reward_model_id: str = "PKU-Alignment/beaver-7b-v1.0-reward"
    batch_size: int = 4
    cost_threshold: float = 0.0  # cost > threshold → unsafe


@dataclass(frozen=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    activations: ActivationConfig = field(default_factory=ActivationConfig)
    probes: ProbeConfig = field(default_factory=ProbeConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _defaults() -> dict[str, Any]:
    cfg = Config()
    return {
        "run": cfg.run.__dict__,
        "model": cfg.model.__dict__,
        "data": cfg.data.__dict__,
        "activations": cfg.activations.__dict__,
        "probes": cfg.probes.__dict__,
        "steering": cfg.steering.__dict__,
        "eval": cfg.eval.__dict__,
    }


def load_config(path: str | Path | None = None) -> Config:
    """Load config from a YAML file, merging with defaults. Pass None for pure defaults."""
    if path is None:
        return Config()

    raw = yaml.safe_load(Path(path).read_text()) or {}
    merged = _deep_update(_defaults(), raw)

    # alphas may come from YAML as a plain list; coerce to tuple
    steer_raw = merged["steering"]
    if "alphas" in steer_raw and not isinstance(steer_raw["alphas"], tuple):
        steer_raw = dict(steer_raw, alphas=tuple(steer_raw["alphas"]))

    return Config(
        run=RunConfig(**merged["run"]),
        model=ModelConfig(**merged["model"]),
        data=DataConfig(**merged["data"]),
        activations=ActivationConfig(**merged["activations"]),
        probes=ProbeConfig(**merged["probes"]),
        steering=SteeringConfig(**steer_raw),
        eval=EvalConfig(**merged["eval"]),
    )
