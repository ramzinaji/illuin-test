"""Entraînement reproductible : nettoyage, split, features, modèle, seuils sur val, évaluation sur test."""

import logging
from pathlib import Path

from src.config import (
    DEFAULT_CONFIG_PATH,
    EMBEDDINGS_DIR,
    MODEL_PATH,
    RAW_DATA_DIR,
    TAGS,
    load_config,
)
from src.data import label_matrix, load_samples, split_labels
from src.evaluate import global_metrics
from src.features import build_blocks, load_encoders, subset
from src.model import TagClassifier, build_model, tune_thresholds
from src.preprocessing import clean_dataset

logger = logging.getLogger(__name__)


def train(
    config_path: Path = DEFAULT_CONFIG_PATH,
    raw_dir: Path = RAW_DATA_DIR,
    model_path: Path = MODEL_PATH,
) -> dict[str, float]:
    """Entraîne sur train, règle les seuils sur val, sauvegarde le modèle et renvoie les métriques test."""
    cfg = load_config(config_path)
    logger.info(
        "Config %s : modèle %s, blocs %s",
        config_path.name,
        cfg.model.type,
        cfg.model.blocks,
    )

    df = clean_dataset(load_samples(raw_dir), cfg.cleaning)
    Y = label_matrix(df["tags"])
    split = split_labels(Y, cfg.split, cfg.seed)
    train, val, test = split == "train", split == "val", split == "test"
    logger.info("Split : train=%d, val=%d, test=%d", train.sum(), val.sum(), test.sum())

    # Embeddings en cache : seul le premier entraînement encode les ~4 800 exercices
    encoders = load_encoders(cfg.encoders)
    blocks = build_blocks(df, encoders, cfg.model.blocks, cache_dir=EMBEDDINGS_DIR)

    model = build_model(cfg).fit(subset(blocks, train), Y[train])
    proba_val = model.predict_proba(subset(blocks, val))
    thresholds = tune_thresholds(Y[val], proba_val, cfg.thresholds.grid)
    classifier = TagClassifier(model, thresholds, TAGS, cfg)
    classifier.save(model_path)
    logger.info("Modèle sauvegardé dans %s", model_path)

    metrics = global_metrics(Y[test], classifier.predict(subset(blocks, test)))
    logger.info("Test : %s", ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return metrics
