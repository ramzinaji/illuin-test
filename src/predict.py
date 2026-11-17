"""Inférence sur de nouveaux exercices, avec mesure de la latence par exemple."""

import logging
import time

import numpy as np
import pandas as pd

from src.features import Encoders, build_blocks
from src.model import TagClassifier
from src.preprocessing import add_text_columns

logger = logging.getLogger(__name__)

# Champs facultatifs d'un exercice : absents, ils sont traités comme vides / difficulté inconnue
OPTIONAL_COLUMNS = ["prob_desc_input_spec", "prob_desc_notes", "difficulty"]


def predict(
    df: pd.DataFrame, classifier: TagClassifier, encoders: Encoders
) -> tuple[np.ndarray, np.ndarray]:
    """Matrice binaire (n x 8) et latence de chaque exemple en secondes (nettoyage, encodage, modèle)."""
    df = df.reindex(columns=df.columns.union(OPTIONAL_COLUMNS))
    predictions, latencies = [], []
    for i in range(len(df)):
        start = time.perf_counter()
        row = add_text_columns(df.iloc[[i]])
        blocks = build_blocks(row, encoders, classifier.config.model.blocks)
        predictions.append(classifier.predict(blocks)[0])
        latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies)
    logger.info(
        "Latence par exemple : médiane %.0f ms, max %.0f ms",
        np.median(latencies) * 1000,
        latencies.max() * 1000,
    )
    return np.array(predictions), latencies


def to_tag_lists(predictions: np.ndarray, tags: list[str]) -> list[list[str]]:
    return [[tag for tag, on in zip(tags, row) if on] for row in predictions]
