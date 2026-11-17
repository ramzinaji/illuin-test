"""Entraînement reproductible : nettoyage, split, features, stacking, seuils sur val, évaluation sur test."""

import logging
from pathlib import Path

from src.config import BLOCKS, EMBEDDINGS_DIR, MODEL_PATH, RAW_DATA_DIR, TAGS
from src.data import label_matrix, load_samples, split_labels
from src.evaluate import global_metrics
from src.features import build_blocks, load_encoders, subset
from src.model import StackingModel, TagClassifier, tune_thresholds
from src.preprocessing import clean_dataset

logger = logging.getLogger(__name__)


def train(
    raw_dir: Path = RAW_DATA_DIR, model_path: Path = MODEL_PATH
) -> dict[str, float]:
    """Entraîne sur train, règle les seuils sur val, sauvegarde le modèle et renvoie les métriques test."""
    df = clean_dataset(load_samples(raw_dir))
    Y = label_matrix(df["tags"])
    split = split_labels(Y)
    train, val, test = split == "train", split == "val", split == "test"
    logger.info("Split : train=%d, val=%d, test=%d", train.sum(), val.sum(), test.sum())

    # Embeddings en cache : seul le premier entraînement encode les ~4 800 exercices
    blocks = build_blocks(df, load_encoders(), cache_dir=EMBEDDINGS_DIR)

    stacking = StackingModel(BLOCKS).fit(subset(blocks, train), Y[train])
    thresholds = tune_thresholds(Y[val], stacking.predict_proba(subset(blocks, val)))
    classifier = TagClassifier(stacking, thresholds, TAGS)
    classifier.save(model_path)
    logger.info("Modèle sauvegardé dans %s", model_path)

    metrics = global_metrics(Y[test], classifier.predict(subset(blocks, test)))
    logger.info("Test : %s", ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return metrics
