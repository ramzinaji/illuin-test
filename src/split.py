"""Construction des labels multi-label et split train/val/test stratifié."""

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.config import SEED, TARGET_TAGS, SplitConfig

DEFAULT_SPLIT_CFG = SplitConfig()


def build_label_matrix(
    tags: pd.Series, target_tags: list[str] = TARGET_TAGS
) -> pd.DataFrame:
    """Matrice binaire (n_exemples x n_tags) des tags cibles."""
    return pd.DataFrame(
        {tag: tags.apply(lambda ts, t=tag: t in ts) for tag in target_tags},
        index=tags.index,
    ).astype(int)


def _stratified_split(
    y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    return next(splitter.split(np.zeros(len(y)), y))


def train_val_test_split(
    labels: pd.DataFrame, cfg: SplitConfig = DEFAULT_SPLIT_CFG, seed: int = SEED
) -> dict[str, pd.Index]:
    """Index train/val/test, stratifiés sur les tags cibles (itérative, adaptée au multi-label)."""
    y = labels.to_numpy()
    rest_pos, test_pos = _stratified_split(y, cfg.test_size, seed)
    # Taille de val rapportée au reste, pour obtenir val_size du total
    val_ratio = cfg.val_size / (1 - cfg.test_size)
    train_rel, val_rel = _stratified_split(y[rest_pos], val_ratio, seed)
    return {
        "train": labels.index[rest_pos[train_rel]],
        "val": labels.index[rest_pos[val_rel]],
        "test": labels.index[test_pos],
    }
