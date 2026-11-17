"""Régression logistique one-vs-rest (un classifieur binaire par tag), avec PCA optionnelle."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.config import SEED, LogRegConfig

DEFAULT_LOGREG_CFG = LogRegConfig()


def build_logreg(cfg: LogRegConfig = DEFAULT_LOGREG_CFG) -> Pipeline:
    """[PCA ->] 8 régressions logistiques L2 indépendantes, classes pondérées (`balanced`)."""
    steps = []
    if cfg.n_components is not None:
        # Dans le pipeline, la PCA n'est ajustée que sur les données d'entraînement : pas de fuite
        steps.append(("pca", PCA(n_components=cfg.n_components, random_state=SEED)))
    classifier = LogisticRegression(
        C=cfg.C, max_iter=cfg.max_iter, class_weight="balanced", random_state=SEED
    )
    steps.append(("clf", OneVsRestClassifier(classifier)))
    return Pipeline(steps)


def fit_predict_logreg(
    cfg: LogRegConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray | None = None,
) -> np.ndarray:
    """Entraîne puis renvoie les probabilités sur `x_eval` (`y_eval` inutilisé : interface commune)."""
    return build_logreg(cfg).fit(x_train, y_train).predict_proba(x_eval)
