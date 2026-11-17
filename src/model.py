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

from src.config import Config
from src.features import Blocks, DifficultyEncoder


def block_model(name: str, cfg: Config):
    """Modèle de niveau 1 d'un bloc ; les transformations (tranches, vocabulaire) sont apprises au fit."""
    level1 = cfg.model.level1
    # Une régression par tag ; `balanced` compense la rareté des tags (1,8 % à 28 %)
    logistic = OneVsRestClassifier(
        LogisticRegression(
            C=level1.C[name],
            class_weight=level1.class_weight,
            max_iter=level1.max_iter,
        )
    )
    if name == "difficulty":
        return make_pipeline(DifficultyEncoder(cfg.features.difficulty_bins), logistic)
    if name == "tfidf_desc":
        tfidf = cfg.features.tfidf_desc
        vectorizer = TfidfVectorizer(
            stop_words=tfidf.stop_words,
            min_df=tfidf.min_df,
            sublinear_tf=tfidf.sublinear_tf,
        )
        return make_pipeline(vectorizer, logistic)
    return logistic  # embeddings


class StackingModel:
    """Niveau 1 : un modèle par bloc. Niveau 2 : par tag, une régression logistique sur les logits out-of-fold."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.block_names = cfg.model.blocks

    def _logit(self, p: np.ndarray) -> np.ndarray:
        eps = self.cfg.model.level2.logit_eps
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def fit(self, blocks: Blocks, Y: np.ndarray) -> "StackingModel":
        level2 = self.cfg.model.level2
        # Out-of-fold : le niveau 2 apprend sur des prédictions faites sans avoir vu l'exemple
        folds = MultilabelStratifiedKFold(
            n_splits=level2.n_folds, shuffle=True, random_state=self.cfg.seed
        )
        oof = np.stack(
            [
                cross_val_predict(
                    block_model(n, self.cfg),
                    blocks[n],
                    Y,
                    cv=folds,
                    method="predict_proba",
                )
                for n in self.block_names
            ],
            axis=1,
        )
        self.block_models_ = {
            n: block_model(n, self.cfg).fit(blocks[n], Y) for n in self.block_names
        }
        self.meta_models_ = [
            LogisticRegression(
                C=level2.C, class_weight=level2.class_weight, max_iter=level2.max_iter
            ).fit(self._logit(oof[:, :, k]), Y[:, k])
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
                meta.predict_proba(self._logit(probas[:, :, k]))[:, 1]
                for k, meta in enumerate(self.meta_models_)
            ]
        )


# Modèles disponibles, choisis par la clé `model.type` du YAML
MODELS = {"stacking": StackingModel}


def build_model(cfg: Config):
    if cfg.model.type not in MODELS:
        raise ValueError(
            f"Modèle inconnu : {cfg.model.type} (disponibles : {list(MODELS)})"
        )
    return MODELS[cfg.model.type](cfg)


def tune_thresholds(Y: np.ndarray, proba: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Pour chaque tag, le seuil de `grid` qui maximise le F1 (à régler sur la validation, jamais sur le test)."""
    thresholds = []
    for k in range(Y.shape[1]):
        f1s = [f1_score(Y[:, k], proba[:, k] >= t, zero_division=0) for t in grid]
        thresholds.append(grid[int(np.argmax(f1s))])
    return np.array(thresholds)


class TagClassifier:
    """Modèle entraîné, seuils par tag et config d'entraînement : l'objet sauvegardé sur disque."""

    def __init__(self, model, thresholds: np.ndarray, tags: list[str], config: Config):
        self.model = model
        self.thresholds = thresholds
        self.tags = tags
        self.config = config  # relue à l'inférence : mêmes encodeurs et mêmes blocs

    def predict(self, blocks: Blocks) -> np.ndarray:
        return (self.model.predict_proba(blocks) >= self.thresholds).astype(int)

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
