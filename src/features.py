"""Blocs de features : embeddings (Sentence-BERT, UniXcoder), difficulté encodée, texte pour la TF-IDF."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import EncoderConfig

logger = logging.getLogger(__name__)

# Un tableau par bloc, aligné sur les lignes du DataFrame
Blocks = dict[str, np.ndarray]

# Blocs embeddings de texte : colonne nettoyée en entrée, colonne d'origine (nom du cache)
TEXT_BLOCKS = {
    "description": ("text_clean", "prob_desc_description"),
    "input_spec": ("input_spec_clean", "prob_desc_input_spec"),
    "notes": ("notes_clean", "prob_desc_notes"),
}


@dataclass
class Encoders:
    text: SentenceTransformer
    code: SentenceTransformer
    config: EncoderConfig


def load_encoders(cfg: EncoderConfig) -> Encoders:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    code = SentenceTransformer(cfg.code_model, device=device)
    code.max_seq_length = cfg.code_max_tokens
    text = SentenceTransformer(cfg.text_model, device=device)
    return Encoders(text=text, code=code, config=cfg)


def embed(
    texts: pd.Series,
    encoder: SentenceTransformer,
    model_name: str,
    column: str,
    batch_size: int,
    cache_dir: Path | None,
) -> np.ndarray:
    """Embeddings normalisés L2 ; avec `cache_dir`, relus depuis le disque s'ils portent sur les mêmes exemples."""
    # Mêmes noms et contenu de cache que dans notebooks_prod : les caches sont partagés
    cache_path = (
        cache_dir / f"{model_name.split('/')[-1]}__{column}.npz" if cache_dir else None
    )
    if cache_path is not None and cache_path.exists():
        cache = np.load(cache_path)
        if str(cache["model"]) == model_name and np.array_equal(
            cache["sample_id"], texts.index
        ):
            return cache["embeddings"]
        logger.warning("Cache %s obsolète : recalcul", cache_path.name)

    embeddings = encoder.encode(
        texts.tolist(), batch_size=batch_size, normalize_embeddings=True
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            embeddings=embeddings,
            sample_id=texts.index.to_numpy(),
            model=model_name,
        )
    return embeddings


def _build_block(
    name: str, df: pd.DataFrame, encoders: Encoders, cache_dir: Path | None
) -> np.ndarray:
    cfg = encoders.config
    if name in TEXT_BLOCKS:
        column, cache_name = TEXT_BLOCKS[name]
        return embed(
            df[column],
            encoders.text,
            cfg.text_model,
            cache_name,
            cfg.text_batch_size,
            cache_dir,
        )
    if name == "code":
        return embed(
            df["source_code"],
            encoders.code,
            cfg.code_model,
            "source_code",
            cfg.code_batch_size,
            cache_dir,
        )
    # Blocs bruts : leur transformation est apprise dans le modèle, sur les données d'entraînement
    if name == "difficulty":
        return df[["difficulty"]].to_numpy(dtype=float)
    if name == "tfidf_desc":
        return df["text_lemma"].to_numpy()
    raise ValueError(f"Bloc inconnu : {name}")


def build_blocks(
    df: pd.DataFrame,
    encoders: Encoders,
    names: list[str],
    cache_dir: Path | None = None,
) -> Blocks:
    """Les blocs `names` ; `df` doit contenir les colonnes de `preprocessing.add_text_columns`."""
    return {name: _build_block(name, df, encoders, cache_dir) for name in names}


def subset(blocks: Blocks, rows: np.ndarray) -> Blocks:
    return {name: block[rows] for name, block in blocks.items()}


class DifficultyEncoder(BaseEstimator, TransformerMixin):
    """One-hot de la difficulté : `n_bins` tranches de quantiles (apprises au fit) + une colonne « inconnue »."""

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    def fit(self, X: np.ndarray, y=None) -> "DifficultyEncoder":
        values = X[:, 0]
        known = values[values > 0]  # NaN > 0 est faux : NaN et -1 sont exclus
        self.edges_ = np.quantile(known, np.linspace(0, 1, self.n_bins + 1)[1:-1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = X[:, 0]
        column = np.where(
            values > 0, np.digitize(np.nan_to_num(values), self.edges_), self.n_bins
        )
        return np.eye(self.n_bins + 1)[column]
