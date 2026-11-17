"""Blocs de features : embeddings SBERT des textes, UniXcoder du code, difficulté, textes bruts pour TF-IDF."""

import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import EMBEDDINGS_DIR, FeatureConfig
from src.embeddings import (
    compute_embeddings,
    embeddings_cache_path,
    load_cached_embeddings,
    load_encoder,
)
from src.preprocessing import normalize_text, prepare_description

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_CFG = FeatureConfig()
DIFFICULTY_BLOCK = "difficulty"

TFIDF_CODE_BLOCK = "tfidf_code"
TFIDF_DESC_BLOCK = "tfidf_desc"
TFIDF_BLOCKS = (TFIDF_CODE_BLOCK, TFIDF_DESC_BLOCK)

# Un tableau par bloc, aligné sur les exemples : embeddings (n x 768), difficulté (n x 1), textes bruts (n,)
Blocks = dict[str, np.ndarray]
# Position des colonnes de chaque bloc dans une matrice concaténée
FeatureLayout = dict[str, slice]


class DifficultyEncoder(BaseEstimator, TransformerMixin):
    """One-hot de la difficulté en tranches de quantiles (apprises au fit) + une colonne « inconnue »."""

    def __init__(self, n_bins: int = DEFAULT_FEATURE_CFG.difficulty_bins) -> None:
        self.n_bins = n_bins

    @staticmethod
    def _unknown(values: np.ndarray) -> np.ndarray:
        # -1 est une sentinelle du dataset (vu dans 01_eda), au même titre que NaN
        return np.isnan(values) | (values <= 0)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "DifficultyEncoder":
        values = np.asarray(X, dtype=float)[:, 0]
        known = values[~self._unknown(values)]
        self.edges_ = np.quantile(known, np.linspace(0, 1, self.n_bins + 1)[1:-1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)[:, 0]
        unknown = self._unknown(values)
        onehot = np.zeros((len(values), self.n_bins + 1))
        bins = np.digitize(np.where(unknown, 0, values), self.edges_)
        onehot[np.arange(len(values)), np.where(unknown, self.n_bins, bins)] = 1
        return onehot


def _block_inputs(df: pd.DataFrame, column: str, is_code: bool) -> list[str]:
    raw = df[column].fillna("")
    # Le code est gardé brut : la casse et la ponctuation y portent du sens pour UniXcoder
    return raw.tolist() if is_code else raw.map(normalize_text).tolist()


def _embedding_block(
    df: pd.DataFrame,
    column: str,
    model_name: str,
    batch_size: int,
    is_code: bool = False,
    max_seq_length: int | None = None,
) -> np.ndarray:
    """Embeddings d'une colonne ; l'encodeur n'est chargé que si le cache est absent ou obsolète."""
    sample_ids = df.index.to_numpy()
    cache_path = embeddings_cache_path(EMBEDDINGS_DIR, model_name, column)
    cached = load_cached_embeddings(cache_path, sample_ids, model_name)
    if cached is not None:
        return cached
    texts = _block_inputs(df, column, is_code)
    encoder = load_encoder(model_name, max_seq_length)
    return compute_embeddings(
        texts, sample_ids, encoder, cache_path, model_name, batch_size
    )


def build_blocks(df: pd.DataFrame, cfg: FeatureConfig = DEFAULT_FEATURE_CFG) -> Blocks:
    """Tous les blocs, alignés sur `df` ; embeddings lus depuis le cache (calculés s'il est absent)."""
    blocks = {
        name: _embedding_block(df, column, cfg.text_model, cfg.text_batch_size)
        for name, column in cfg.text_blocks
    }
    code_name, code_column = cfg.code_block
    blocks[code_name] = _embedding_block(
        df,
        code_column,
        cfg.code_model,
        cfg.code_batch_size,
        is_code=True,
        max_seq_length=cfg.code_max_seq_length,
    )
    return _add_raw_blocks(blocks, df, cfg)


def encode_blocks(
    df: pd.DataFrame,
    text_encoder: SentenceTransformer,
    code_encoder: SentenceTransformer,
    cfg: FeatureConfig = DEFAULT_FEATURE_CFG,
) -> Blocks:
    """Mêmes blocs que `build_blocks`, calculés sans cache (inférence sur de nouveaux exemples)."""
    blocks = {
        name: text_encoder.encode(
            _block_inputs(df, column, is_code=False), normalize_embeddings=True
        )
        for name, column in cfg.text_blocks
    }
    code_name, code_column = cfg.code_block
    blocks[code_name] = code_encoder.encode(
        _block_inputs(df, code_column, is_code=True), normalize_embeddings=True
    )
    return _add_raw_blocks(blocks, df, cfg)


def _add_raw_blocks(blocks: Blocks, df: pd.DataFrame, cfg: FeatureConfig) -> Blocks:
    """Blocs non encodés ici : leur transformation est apprise dans le pipeline, sur le train uniquement."""
    blocks[DIFFICULTY_BLOCK] = df[[cfg.difficulty_column]].to_numpy(dtype=float)
    # Textes bruts pour les blocs TF-IDF (vocabulaire et IDF appris dans chaque fold)
    blocks[TFIDF_CODE_BLOCK] = df[cfg.code_block[1]].fillna("").to_numpy(dtype=object)
    blocks[TFIDF_DESC_BLOCK] = (
        df[cfg.text_blocks[0][1]].fillna("").to_numpy(dtype=object)
    )
    return blocks


def subset_blocks(blocks: Blocks, idx: np.ndarray) -> Blocks:
    return {name: block[idx] for name, block in blocks.items()}


def concat_rows(*parts: Blocks) -> Blocks:
    """Empile les exemples de plusieurs jeux de blocs (ex. train + val)."""
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def concat_blocks(blocks: Blocks, names: list[str]) -> tuple[np.ndarray, FeatureLayout]:
    """Matrice dense des blocs `names` concaténés, et la position de chaque bloc."""
    layout, start = {}, 0
    for name in names:
        layout[name] = slice(start, start + blocks[name].shape[1])
        start += blocks[name].shape[1]
    return np.hstack([blocks[name] for name in names]), layout


def prepare_descriptions(texts: np.ndarray) -> list[str]:
    """Nettoyage + lemmatisation d'un lot (fonction nommée : le pipeline reste sérialisable)."""
    return [prepare_description(text) for text in texts]
