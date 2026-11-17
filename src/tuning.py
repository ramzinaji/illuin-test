"""Recherche d'hyperparamètres par validation croisée multi-label (critère sans seuil : Macro-AP)."""

import itertools
import logging
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import average_precision_score

from src.config import SEED

logger = logging.getLogger(__name__)

# (x_train, y_train, x_eval, y_eval) -> probabilités (n_eval x n_tags)
FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def macro_average_precision(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Moyenne des aires sous la courbe précision-rappel par tag : indépendante des seuils."""
    return average_precision_score(y_true, proba, average="macro")


def cross_val_scores(
    fit_predict: FitPredict,
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int = SEED,
) -> list[float]:
    """Macro-AP de chaque fold d'une validation croisée stratifiée multi-label (`x` : matrice ou dict de blocs)."""
    folds = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    scores = []
    for train_idx, eval_idx in folds.split(np.zeros(len(y)), y):
        proba = fit_predict(
            _take(x, train_idx), y[train_idx], _take(x, eval_idx), y[eval_idx]
        )
        scores.append(macro_average_precision(y[eval_idx], proba))
    return scores


def _take(x: np.ndarray | dict[str, np.ndarray], idx: np.ndarray):
    return (
        {name: arr[idx] for name, arr in x.items()} if isinstance(x, dict) else x[idx]
    )


def grid_search(
    grid: dict[str, tuple],
    make_fit_predict: Callable[[dict], FitPredict],
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
) -> pd.DataFrame:
    """Évalue toutes les combinaisons de `grid` ; résultats triés par Macro-AP moyenne décroissante."""
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values, strict=True))
        start = time.perf_counter()
        scores = cross_val_scores(make_fit_predict(params), x, y, n_splits)
        rows.append(
            {
                **params,
                "cv_macro_ap": np.mean(scores),
                "cv_std": np.std(scores),
                "seconds": time.perf_counter() - start,
            }
        )
        logger.info("%s -> Macro-AP %.4f", params, rows[-1]["cv_macro_ap"])
    return pd.DataFrame(rows).sort_values(
        "cv_macro_ap", ascending=False, ignore_index=True
    )
