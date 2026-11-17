"""Chemins et tags cibles ; hyperparamètres lus depuis un fichier YAML (configs/stacking.yaml)."""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "code_classification_dataset"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
LLM_PREDICTIONS_DIR = ROOT / "data" / "llm"
MODEL_PATH = ROOT / "models" / "stacking.joblib"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "stacking.yaml"

TAGS = [
    "math",
    "graphs",
    "strings",
    "number theory",
    "trees",
    "geometry",
    "games",
    "probabilities",
]


@dataclass(frozen=True)
class CleaningConfig:
    language: str
    near_dup_threshold: float
    near_dup_min_df: int


@dataclass(frozen=True)
class SplitConfig:
    val_size: float
    test_size: float


@dataclass(frozen=True)
class EncoderConfig:
    text_model: str
    text_batch_size: int
    code_model: str
    code_max_tokens: int
    code_batch_size: int


@dataclass(frozen=True)
class TfidfConfig:
    min_df: int
    stop_words: str
    sublinear_tf: bool


@dataclass(frozen=True)
class FeatureConfig:
    difficulty_bins: int
    tfidf_desc: TfidfConfig


@dataclass(frozen=True)
class Level1Config:
    class_weight: str
    max_iter: int
    C: dict[str, float]  # un C par bloc


@dataclass(frozen=True)
class Level2Config:
    n_folds: int
    logit_eps: float
    class_weight: str
    max_iter: int
    C: float


@dataclass(frozen=True)
class ModelConfig:
    type: str
    blocks: list[str]
    level1: Level1Config
    level2: Level2Config


@dataclass(frozen=True)
class ThresholdConfig:
    start: float
    stop: float
    step: float

    @property
    def grid(self) -> np.ndarray:
        return np.round(np.arange(self.start, self.stop + self.step / 2, self.step), 4)


@dataclass(frozen=True)
class Config:
    seed: int
    cleaning: CleaningConfig
    split: SplitConfig
    encoders: EncoderConfig
    features: FeatureConfig
    model: ModelConfig
    thresholds: ThresholdConfig


def _check_keys(section: dict, expected: set[str], name: str) -> None:
    if set(section) != expected:
        raise ValueError(
            f"Clés de `{name}` attendues : {sorted(expected)}, reçues : {sorted(section)}"
        )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Lit le YAML ; une clé manquante ou inconnue lève une erreur au lieu d'être ignorée."""
    raw = yaml.safe_load(path.read_text())
    # Les sous-sections construites par `**` sont déjà vérifiées par les dataclasses
    _check_keys(raw, {f.name for f in fields(Config)}, "racine")
    features, model = raw["features"], raw["model"]
    _check_keys(features, {"difficulty", "tfidf_desc"}, "features")
    _check_keys(features["difficulty"], {"bins"}, "features.difficulty")
    _check_keys(model, {"type", "blocks", "level1", "level2"}, "model")
    config = Config(
        seed=raw["seed"],
        cleaning=CleaningConfig(**raw["cleaning"]),
        split=SplitConfig(**raw["split"]),
        encoders=EncoderConfig(**raw["encoders"]),
        features=FeatureConfig(
            difficulty_bins=features["difficulty"]["bins"],
            tfidf_desc=TfidfConfig(**features["tfidf_desc"]),
        ),
        model=ModelConfig(
            type=model["type"],
            blocks=model["blocks"],
            level1=Level1Config(**model["level1"]),
            level2=Level2Config(**model["level2"]),
        ),
        thresholds=ThresholdConfig(**raw["thresholds"]),
    )
    missing = set(config.model.blocks) - set(config.model.level1.C)
    if missing:
        raise ValueError(f"Pas de C de niveau 1 pour les blocs : {sorted(missing)}")
    return config
