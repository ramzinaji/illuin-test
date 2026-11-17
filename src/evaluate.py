"""Métriques multi-label : globales et par tag."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
)

from src.config import TAGS


def global_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict[str, float]:
    """Micro-F1, Macro-F1 (métrique principale : chaque tag compte autant), Hamming loss, Jaccard moyen."""
    return {
        "micro_f1": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(Y_true, Y_pred),
        # Exemple sans tag vrai ni prédit = prédiction parfaite
        "jaccard": jaccard_score(Y_true, Y_pred, average="samples", zero_division=1),
    }


def per_tag_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        Y_true, Y_pred, zero_division=0
    )
    return pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support},
        index=TAGS,
    )
