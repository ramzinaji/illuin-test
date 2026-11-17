"""Fusion des blocs de features : concaténation (1 régression logistique) ou stacking (1 par bloc)."""

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.class_tfidf import CODE_STOP_WORDS, CODE_TOKEN_PATTERN
from src.config import SEED, FeatureConfig, LogRegConfig
from src.features import (
    DIFFICULTY_BLOCK,
    TFIDF_CODE_BLOCK,
    TFIDF_DESC_BLOCK,
    Blocks,
    DifficultyEncoder,
    concat_blocks,
    prepare_descriptions,
    subset_blocks,
)
from src.linear import build_logreg

DEFAULT_LOGREG_CFG = LogRegConfig()
DEFAULT_FEATURE_CFG = FeatureConfig()
EPS = 1e-6


def _ovr_logreg(C: float) -> OneVsRestClassifier:
    return OneVsRestClassifier(
        LogisticRegression(
            C=C,
            max_iter=DEFAULT_LOGREG_CFG.max_iter,
            class_weight="balanced",
            random_state=SEED,
        )
    )


def block_model(name: str, cfg: FeatureConfig = DEFAULT_FEATURE_CFG) -> Pipeline:
    """Modèle de niveau 1 d'un bloc ; chaque transformation (tranches, vocabulaire, IDF) est apprise au fit."""
    if name == DIFFICULTY_BLOCK:
        return make_pipeline(DifficultyEncoder(), _ovr_logreg(DEFAULT_LOGREG_CFG.C))
    if name == TFIDF_CODE_BLOCK:
        vectorizer = TfidfVectorizer(
            token_pattern=CODE_TOKEN_PATTERN,
            lowercase=False,
            stop_words=CODE_STOP_WORDS,
            min_df=cfg.tfidf_code_min_df,
            sublinear_tf=True,  # un template répété 50 fois ne pèse pas 50 fois plus
        )
        return make_pipeline(vectorizer, _ovr_logreg(cfg.tfidf_code_C))
    if name == TFIDF_DESC_BLOCK:
        vectorizer = TfidfVectorizer(
            stop_words="english", min_df=cfg.tfidf_desc_min_df, sublinear_tf=True
        )
        # Lemmatisation dans le pipeline : appliquée à l'identique à l'entraînement et à l'inférence
        return make_pipeline(
            FunctionTransformer(prepare_descriptions),
            vectorizer,
            _ovr_logreg(cfg.tfidf_desc_C),
        )
    return build_logreg(
        DEFAULT_LOGREG_CFG
    )  # embeddings : identique au benchmark du notebook 03


class ConcatTagger(BaseEstimator, ClassifierMixin):
    """Blocs denses concaténés -> standardisation de chaque colonne -> 8 régressions logistiques."""

    def __init__(self, block_names: list[str], C: float = DEFAULT_LOGREG_CFG.C) -> None:
        self.block_names = block_names
        self.C = C

    def fit(self, blocks: Blocks, Y: np.ndarray) -> "ConcatTagger":
        X, layout = concat_blocks(blocks, self.block_names)
        transformers = [
            (
                name,
                DifficultyEncoder() if name == DIFFICULTY_BLOCK else "passthrough",
                sl,
            )
            for name, sl in layout.items()
        ]
        # La standardisation met toutes les colonnes à la même échelle : la pénalité L2 les traite alors équitablement
        self.model_ = make_pipeline(
            ColumnTransformer(transformers), StandardScaler(), _ovr_logreg(self.C)
        )
        self.model_.fit(X, Y)
        return self

    def predict_proba(self, blocks: Blocks) -> np.ndarray:
        return self.model_.predict_proba(concat_blocks(blocks, self.block_names)[0])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class StackingTagger(BaseEstimator, ClassifierMixin):
    """Niveau 1 : un modèle par bloc. Niveau 2 : par tag, une régression logistique sur les probabilités
    out-of-fold des blocs, qui apprend explicitement le poids de chaque bloc."""

    def __init__(
        self,
        block_names: list[str],
        feature_cfg: FeatureConfig = DEFAULT_FEATURE_CFG,
        n_splits: int = 5,
        seed: int = SEED,
    ) -> None:
        self.block_names = block_names
        self.feature_cfg = feature_cfg
        self.n_splits = n_splits
        self.seed = seed

    def _out_of_fold_probas(self, blocks: Blocks, Y: np.ndarray) -> np.ndarray:
        # Prédictions sur des exemples non vus à l'entraînement : sinon le niveau 2 surestime les blocs qui sur-apprennent
        oof = np.zeros((len(Y), len(self.block_names), Y.shape[1]))
        folds = MultilabelStratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )
        for train_idx, eval_idx in folds.split(np.zeros(len(Y)), Y):
            train, held_out = (
                subset_blocks(blocks, train_idx),
                subset_blocks(blocks, eval_idx),
            )
            for b, name in enumerate(self.block_names):
                model = block_model(name, self.feature_cfg).fit(
                    train[name], Y[train_idx]
                )
                oof[eval_idx, b] = model.predict_proba(held_out[name])
        return oof

    def fit(self, blocks: Blocks, Y: np.ndarray) -> "StackingTagger":
        oof = self._out_of_fold_probas(blocks, Y)
        self.block_models_ = {
            name: block_model(name, self.feature_cfg).fit(blocks[name], Y)
            for name in self.block_names
        }
        self.meta_models_ = [
            LogisticRegression(class_weight="balanced", random_state=self.seed).fit(
                _logit(oof[:, :, k]), Y[:, k]
            )
            for k in range(Y.shape[1])
        ]
        return self

    def predict_proba(self, blocks: Blocks) -> np.ndarray:
        # (n, n_blocs, n_tags) : probabilités de chaque bloc de niveau 1
        logits = _logit(
            np.stack(
                [
                    self.block_models_[name].predict_proba(blocks[name])
                    for name in self.block_names
                ],
                axis=1,
            )
        )
        return np.column_stack(
            [
                m.predict_proba(logits[:, :, k])[:, 1]
                for k, m in enumerate(self.meta_models_)
            ]
        )

    def meta_coefficients(self, tags: list[str]) -> pd.DataFrame:
        """Poids appris par le niveau 2 : importance de chaque bloc pour chaque tag."""
        return pd.DataFrame(
            [m.coef_[0] for m in self.meta_models_],
            index=tags,
            columns=self.block_names,
        )
