"""Blocs de features : embeddings (Sentence-BERT, UniXcoder), difficulté encodée, texte pour la TF-IDF."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    CODE_BATCH_SIZE,
    CODE_MAX_TOKENS,
    CODE_MODEL,
    DIFFICULTY_BINS,
    TEXT_BATCH_SIZE,
    TEXT_MODEL,
)

logger = logging.getLogger(__name__)

# Un tableau par bloc, aligné sur les lignes du DataFrame
Blocks = dict[str, np.ndarray]


@dataclass
class Encoders:
    text: SentenceTransformer
    code: SentenceTransformer


def load_encoders() -> Encoders:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    code = SentenceTransformer(CODE_MODEL, device=device)
    code.max_seq_length = CODE_MAX_TOKENS
    return Encoders(text=SentenceTransformer(TEXT_MODEL, device=device), code=code)


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


def build_blocks(
    df: pd.DataFrame, encoders: Encoders, cache_dir: Path | None = None
) -> Blocks:
    """Tous les blocs du modèle ; `df` doit contenir les colonnes de `preprocessing.add_text_columns`."""
    text, code = encoders.text, encoders.code
    return {
        "description": embed(
            df["text_clean"],
            text,
            TEXT_MODEL,
            "prob_desc_description",
            TEXT_BATCH_SIZE,
            cache_dir,
        ),
        "input_spec": embed(
            df["input_spec_clean"],
            text,
            TEXT_MODEL,
            "prob_desc_input_spec",
            TEXT_BATCH_SIZE,
            cache_dir,
        ),
        "notes": embed(
            df["notes_clean"],
            text,
            TEXT_MODEL,
            "prob_desc_notes",
            TEXT_BATCH_SIZE,
            cache_dir,
        ),
        "code": embed(
            df["source_code"],
            code,
            CODE_MODEL,
            "source_code",
            CODE_BATCH_SIZE,
            cache_dir,
        ),
        # Blocs bruts : leur transformation est apprise dans le modèle, sur les données d'entraînement
        "difficulty": df[["difficulty"]].to_numpy(dtype=float),
        "tfidf_desc": df["text_lemma"].to_numpy(),
    }


def subset(blocks: Blocks, rows: np.ndarray) -> Blocks:
    return {name: block[rows] for name, block in blocks.items()}


class DifficultyEncoder(BaseEstimator, TransformerMixin):
    """One-hot de la difficulté : tranches de quantiles (apprises au fit) + une colonne « inconnue » (NaN, -1)."""

    def fit(self, X: np.ndarray, y=None) -> "DifficultyEncoder":
        values = X[:, 0]
        known = values[values > 0]  # NaN > 0 est faux : NaN et -1 sont exclus
        self.edges_ = np.quantile(known, np.linspace(0, 1, DIFFICULTY_BINS + 1)[1:-1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = X[:, 0]
        column = np.where(
            values > 0, np.digitize(np.nan_to_num(values), self.edges_), DIFFICULTY_BINS
        )
        return np.eye(DIFFICULTY_BINS + 1)[column]
