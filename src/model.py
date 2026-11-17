"""Modèle retenu : stacking d'une régression logistique par bloc, avec un seuil de décision par tag."""

from pathlib import Path

import joblib
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

from src.config import N_FOLDS, SEED, TFIDF_MIN_DF, THRESHOLD_GRID
from src.features import Blocks, DifficultyEncoder


def logistic() -> OneVsRestClassifier:
    # Une régression par tag ; `balanced` compense la rareté des tags (1,8 % à 28 %)
    return OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=2000)
    )


def block_model(name: str):
    """Modèle de niveau 1 d'un bloc ; les transformations (tranches, vocabulaire) sont apprises au fit."""
    if name == "difficulty":
        return make_pipeline(DifficultyEncoder(), logistic())
    if name == "tfidf_desc":
        return make_pipeline(
            TfidfVectorizer(
                stop_words="english", min_df=TFIDF_MIN_DF, sublinear_tf=True
            ),
            logistic(),
        )
    return logistic()  # embeddings


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class StackingModel:
    """Niveau 1 : un modèle par bloc. Niveau 2 : par tag, une régression logistique sur les logits out-of-fold."""

    def __init__(self, block_names: list[str]):
        self.block_names = block_names

    def fit(self, blocks: Blocks, Y: np.ndarray) -> "StackingModel":
        # Out-of-fold : le niveau 2 apprend sur des prédictions faites sans avoir vu l'exemple
        folds = MultilabelStratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=SEED
        )
        oof = np.stack(
            [
                cross_val_predict(
                    block_model(n), blocks[n], Y, cv=folds, method="predict_proba"
                )
                for n in self.block_names
            ],
            axis=1,
        )
        self.block_models_ = {
            n: block_model(n).fit(blocks[n], Y) for n in self.block_names
        }
        self.meta_models_ = [
            LogisticRegression(class_weight="balanced").fit(
                _logit(oof[:, :, k]), Y[:, k]
            )
            for k in range(Y.shape[1])
        ]
        return self

    def predict_proba(self, blocks: Blocks) -> np.ndarray:
        probas = np.stack(
            [self.block_models_[n].predict_proba(blocks[n]) for n in self.block_names],
            axis=1,
        )
        return np.column_stack(
            [
                meta.predict_proba(_logit(probas[:, :, k]))[:, 1]
                for k, meta in enumerate(self.meta_models_)
            ]
        )


def tune_thresholds(Y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Pour chaque tag, le seuil qui maximise le F1 (à régler sur la validation, jamais sur le test)."""
    thresholds = []
    for k in range(Y.shape[1]):
        f1s = [
            f1_score(Y[:, k], proba[:, k] >= t, zero_division=0) for t in THRESHOLD_GRID
        ]
        thresholds.append(THRESHOLD_GRID[int(np.argmax(f1s))])
    return np.array(thresholds)


class TagClassifier:
    """Stacking entraîné + seuils par tag : l'objet sauvegardé sur disque."""

    def __init__(
        self, stacking: StackingModel, thresholds: np.ndarray, tags: list[str]
    ):
        self.stacking, self.thresholds, self.tags = stacking, thresholds, tags

    def predict(self, blocks: Blocks) -> np.ndarray:
        return (self.stacking.predict_proba(blocks) >= self.thresholds).astype(int)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "TagClassifier":
        if not path.exists():
            raise FileNotFoundError(
                f"Modèle absent ({path}) : lancer `python -m src.cli train`"
            )
        return joblib.load(path)
