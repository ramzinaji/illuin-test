"""Modèles multi-label : régression logistique (PCA optionnelle), MLP PyTorch et fusion multi-blocs."""

import copy
import logging

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import SEED, FeatureConfig, LogRegConfig, MLPConfig
from src.features import (
    CODE_STOP_WORDS,
    CODE_TOKEN_PATTERN,
    DIFFICULTY_BLOCK,
    TFIDF_CODE_BLOCK,
    TFIDF_DESC_BLOCK,
    Blocks,
    DifficultyEncoder,
    concat_blocks,
    prepare_descriptions,
    subset_blocks,
)

logger = logging.getLogger(__name__)

DEFAULT_LOGREG_CFG = LogRegConfig()
DEFAULT_MLP_CFG = MLPConfig()
DEFAULT_FEATURE_CFG = FeatureConfig()
EPS = 1e-6


# --- Régression logistique One-vs-Rest -----------------------------------------
def _ovr_logreg(
    C: float = DEFAULT_LOGREG_CFG.C, max_iter: int = DEFAULT_LOGREG_CFG.max_iter
) -> OneVsRestClassifier:
    return OneVsRestClassifier(
        LogisticRegression(
            C=C, max_iter=max_iter, class_weight="balanced", random_state=SEED
        )
    )


def build_logreg(cfg: LogRegConfig = DEFAULT_LOGREG_CFG) -> Pipeline:
    """[PCA ->] 8 régressions logistiques L2 indépendantes, classes pondérées (`balanced`)."""
    steps = []
    if cfg.n_components is not None:
        # Dans le pipeline, la PCA n'est ajustée que sur les données d'entraînement : pas de fuite
        steps.append(("pca", PCA(n_components=cfg.n_components, random_state=SEED)))
    steps.append(("clf", _ovr_logreg(cfg.C, cfg.max_iter)))
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


# --- MLP PyTorch ---------------------------------------------------------------
class TagMLP(nn.Module):
    """Une couche cachée ; renvoie des logits (sigmoïde appliquée dans la loss / à l'inférence)."""

    def __init__(
        self, input_dim: int, n_tags: int, cfg: MLPConfig = DEFAULT_MLP_CFG
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, n_tags),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32)


def _pos_weight(y: np.ndarray) -> torch.Tensor:
    # Compense le déséquilibre : un positif rare pèse neg/pos fois plus qu'un négatif
    positives = y.sum(axis=0).clip(min=1)
    return _to_tensor((len(y) - positives) / positives)


@torch.no_grad()
def predict_proba(model: TagMLP, x: np.ndarray) -> np.ndarray:
    """Probabilités sigmoïdes prédites par le MLP."""
    model.eval()
    return torch.sigmoid(model(_to_tensor(x))).numpy()


def _run_epoch(
    model: TagMLP, loader: DataLoader, loss_fn: nn.Module, optimizer
) -> float:
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


@torch.no_grad()
def _eval_loss(
    model: TagMLP, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module
) -> float:
    model.eval()
    return loss_fn(model(x), y).item()


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: MLPConfig = DEFAULT_MLP_CFG,
    seed: int = SEED,
) -> tuple[TagMLP, pd.DataFrame]:
    """Entraîne avec early stopping sur la loss de val ; renvoie le meilleur modèle et l'historique."""
    torch.manual_seed(seed)
    model = TagMLP(x_train.shape[1], y_train.shape[1], cfg)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y_train))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loader = DataLoader(
        TensorDataset(_to_tensor(x_train), _to_tensor(y_train)),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    xv, yv = _to_tensor(x_val), _to_tensor(y_val)

    history, best_loss, best_state, since_best = [], float("inf"), None, 0
    for epoch in range(cfg.max_epochs):
        train_loss = _run_epoch(model, loader, loss_fn, optimizer)
        val_loss = _eval_loss(model, xv, yv, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            since_best = 0
        else:
            since_best += 1
        if since_best >= cfg.patience:
            break

    model.load_state_dict(best_state)
    logger.info("Arrêt à l'epoch %d, meilleure val_loss=%.4f", epoch, best_loss)
    return model, pd.DataFrame(history)


def fit_predict_mlp(
    cfg: MLPConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> np.ndarray:
    """Interface de `evaluate.grid_search` ; `(x_eval, y_eval)` sert aussi à l'early stopping."""
    model, _ = train_mlp(x_train, y_train, x_eval, y_eval, cfg)
    return predict_proba(model, x_eval)


# --- Fusion multi-blocs : concaténation & stacking -----------------------------
def block_model(name: str, cfg: FeatureConfig = DEFAULT_FEATURE_CFG) -> Pipeline:
    """Modèle de niveau 1 d'un bloc ; chaque transformation (tranches, vocabulaire, IDF) est apprise au fit."""
    if name == DIFFICULTY_BLOCK:
        return make_pipeline(DifficultyEncoder(), _ovr_logreg())
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
        return make_pipeline(
            FunctionTransformer(prepare_descriptions),
            vectorizer,
            _ovr_logreg(cfg.tfidf_desc_C),
        )
    return build_logreg(DEFAULT_LOGREG_CFG)


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
        # La standardisation met toutes les colonnes à la même échelle pour la pénalité L2
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
    """Niveau 1 : un modèle par bloc. Niveau 2 : régression logistique sur les logits out-of-fold."""

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
        # Prédictions sur des exemples non vus à l'entraînement pour éviter que le niveau 2 ne surestime un bloc
        oof = np.zeros((len(Y), len(self.block_names), Y.shape[1]))
        folds = MultilabelStratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )
        for train_idx, eval_idx in folds.split(np.zeros(len(Y)), Y):
            train = subset_blocks(blocks, train_idx)
            held_out = subset_blocks(blocks, eval_idx)
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
