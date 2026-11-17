"""Encodeurs Sentence-Transformers (texte ou code) et cache disque des embeddings."""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def get_device() -> str:
    """GPU Apple (mps) ou CUDA si disponible, sinon CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_encoder(
    model_name: str, max_seq_length: int | None = None
) -> SentenceTransformer:
    encoder = SentenceTransformer(model_name, device=get_device())
    if max_seq_length is not None:
        encoder.max_seq_length = max_seq_length
    return encoder


def benchmark_encoder(
    encoder: SentenceTransformer, texts: list[str], batch_size: int
) -> float:
    """Temps moyen d'encodage par texte (secondes), après un batch de warm-up."""
    encoder.encode(texts[:batch_size], batch_size=batch_size, show_progress_bar=False)
    start = time.perf_counter()
    encoder.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return (time.perf_counter() - start) / len(texts)


def embeddings_cache_path(cache_dir: Path, model_name: str, column: str) -> Path:
    """Un fichier par couple (modèle, colonne encodée)."""
    return cache_dir / f"{model_name.split('/')[-1]}__{column}.npz"


def load_cached_embeddings(
    cache_path: Path, sample_ids: np.ndarray, model_name: str
) -> np.ndarray | None:
    """Embeddings du cache s'il correspond exactement aux mêmes exemples et au même modèle."""
    if not cache_path.exists():
        return None
    cache = np.load(cache_path, allow_pickle=False)
    # Comparer les ids (et pas seulement la taille) évite de réutiliser un cache issu d'un autre nettoyage
    if str(cache["model"]) != model_name or not np.array_equal(
        cache["sample_id"], sample_ids
    ):
        logger.warning("Cache %s obsolète (modèle ou exemples différents)", cache_path)
        return None
    logger.info("Embeddings chargés depuis %s", cache_path)
    return cache["embeddings"]


def compute_embeddings(
    texts: list[str],
    sample_ids: np.ndarray,
    encoder: SentenceTransformer,
    cache_path: Path,
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """Encode les textes (normalisés L2) ; sauvegarde `.npz` (embeddings, sample_id, modèle) réutilisée ensuite."""
    cached = load_cached_embeddings(cache_path, sample_ids, model_name)
    if cached is not None:
        return cached

    start = time.perf_counter()
    embeddings = encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )
    logger.info(
        "%d embeddings calculés en %.1fs", len(texts), time.perf_counter() - start
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, sample_id=sample_ids, model=model_name)
    logger.info("Embeddings sauvegardés dans %s", cache_path)
    return embeddings
