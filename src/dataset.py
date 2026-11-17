"""Jeu de modélisation : données nettoyées, embeddings, labels et split, prêts à l'emploi."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import EMBEDDINGS_DIR, TopicConfig
from src.data import load_raw_dataset
from src.embeddings import (
    compute_embeddings,
    embeddings_cache_path,
    load_cached_embeddings,
    load_encoder,
)
from src.preprocessing import clean_dataset
from src.split import build_label_matrix, train_val_test_split

DEFAULT_TOPIC_CFG = TopicConfig()


@dataclass
class ModelingData:
    df: pd.DataFrame
    # Embeddings de la description par split ("train", "val", "test")
    x: dict[str, np.ndarray]
    y: dict[str, np.ndarray]  # labels binaires par split
    positions: dict[str, np.ndarray]  # positions des exemples de chaque split dans `df`

    def dev(self) -> tuple[np.ndarray, np.ndarray]:
        """Train + val concaténés, pour la validation croisée (le test reste à part)."""
        return np.vstack([self.x["train"], self.x["val"]]), np.vstack(
            [self.y["train"], self.y["val"]]
        )

    def split(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Découpe une matrice alignée sur `df` selon le split train/val/test."""
        return {name: features[p] for name, p in self.positions.items()}


def load_modeling_data(cfg: TopicConfig = DEFAULT_TOPIC_CFG) -> ModelingData:
    """Nettoie, charge (ou calcule) les embeddings de la description et découpe en train/val/test."""
    df = clean_dataset(load_raw_dataset())
    sample_ids = df.index.to_numpy()
    cache_path = embeddings_cache_path(
        EMBEDDINGS_DIR, cfg.embedding_model, "prob_desc_description"
    )
    embeddings = load_cached_embeddings(cache_path, sample_ids, cfg.embedding_model)
    if embeddings is None:
        encoder = load_encoder(cfg.embedding_model)
        embeddings = compute_embeddings(
            df["text_clean"].tolist(),
            sample_ids,
            encoder,
            cache_path,
            cfg.embedding_model,
            cfg.batch_size,
        )

    labels = build_label_matrix(df["tags"])
    positions = {
        name: df.index.get_indexer(idx)
        for name, idx in train_val_test_split(labels).items()
    }
    return ModelingData(
        df=df,
        x={name: embeddings[p] for name, p in positions.items()},
        y={name: labels.iloc[p].to_numpy() for name, p in positions.items()},
        positions=positions,
    )
