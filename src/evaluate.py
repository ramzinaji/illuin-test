"""Métriques multi-label globales et par tag, et optimisation des seuils de décision."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
)

from src.config import SEED

THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)


def global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Micro/Macro-F1, Hamming loss et Jaccard moyen par exemple."""
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        # Exemple sans tag vrai ni prédit = prédiction parfaite (jaccard 1)
        "jaccard_samples": jaccard_score(
            y_true, y_pred, average="samples", zero_division=1
        ),
    }


def per_tag_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, tags: list[str]
) -> pd.DataFrame:
    """Précision, rappel, F1 et support pour chaque tag."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support},
        index=tags,
    )


def tune_thresholds(
    y_true: np.ndarray, proba: np.ndarray, grid: np.ndarray = THRESHOLD_GRID
) -> np.ndarray:
    """Seuil par tag maximisant le F1 (à calculer sur la validation, jamais sur le test)."""
    thresholds = np.empty(y_true.shape[1])
    for k in range(y_true.shape[1]):
        scores = [
            f1_score(y_true[:, k], proba[:, k] >= t, zero_division=0) for t in grid
        ]
        thresholds[k] = grid[int(np.argmax(scores))]
    return thresholds


def apply_thresholds(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (proba >= thresholds).astype(int)


def bootstrap_macro_f1_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_bootstrap: int,
    seed: int = SEED,
) -> dict[str, float]:
    """Bootstrap apparié de Macro-F1(a) - Macro-F1(b) : écart moyen, IC 95 % et P(a > b)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        # Mêmes exemples rééchantillonnés pour les deux modèles : on mesure l'écart, pas le bruit du test
        idx = rng.integers(0, len(y_true), len(y_true))
        diffs[b] = f1_score(
            y_true[idx], pred_a[idx], average="macro", zero_division=0
        ) - f1_score(y_true[idx], pred_b[idx], average="macro", zero_division=0)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff_mean": diffs.mean(),
        "ci_low": low,
        "ci_high": high,
        "p_a_better": (diffs > 0).mean(),
    }
