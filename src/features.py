"""Extraction de features : embeddings (SBERT/UniXcoder), c-TF-IDF, difficulté et blocs multi-modaux."""

import keyword
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import EMBEDDINGS_DIR, FeatureConfig
from src.preprocessing import normalize_text, prepare_description

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_CFG = FeatureConfig()
DIFFICULTY_BLOCK = "difficulty"
TFIDF_CODE_BLOCK = "tfidf_code"
TFIDF_DESC_BLOCK = "tfidf_desc"
TFIDF_BLOCKS = (TFIDF_CODE_BLOCK, TFIDF_DESC_BLOCK)

NO_TAG = "(aucun tag cible)"
# Identifiants Python d'au moins 2 caractères et nombres d'au moins 2 chiffres (ex. 998244353)
CODE_TOKEN_PATTERN = r"[A-Za-z_][A-Za-z0-9_]+|\d{2,}"
# Mots-clés Python et tokens d'entrées/sorties : présents dans toutes les solutions, sans lien avec le tag
CODE_STOP_WORDS = [
    *keyword.kwlist,
    "sys",
    "stdin",
    "stdout",
    "readline",
    "input",
    "raw_input",
    "print",
    "write",
    "flush",
    "os",
    "io",
    "self",
]

# Un tableau par bloc, aligné sur les exemples : embeddings (n x 768), difficulté (n x 1), textes bruts (n,)
Blocks = dict[str, np.ndarray]
FeatureLayout = dict[str, slice]


# --- Encodeurs Sentence-Transformers & cache disque ----------------------------
def get_device() -> str:
    """GPU CUDA ou Apple Silicon (mps) si disponible, sinon CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_encoder(
    model_name: str, max_seq_length: int | None = None
) -> SentenceTransformer:
    """Charge un encodeur Sentence-Transformer sur le meilleur device disponible."""
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
    """Chemin du fichier cache pour un couple `(modèle, colonne)`."""
    return cache_dir / f"{model_name.split('/')[-1]}__{column}.npz"


def load_cached_embeddings(
    cache_path: Path, sample_ids: np.ndarray, model_name: str
) -> np.ndarray | None:
    """Charge le cache `.npz` s'il correspond exactement au modèle et aux `sample_ids`."""
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
    """Encode les textes (normalisés L2) et met le résultat en cache `.npz`."""
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


# --- TF-IDF par classe (c-TF-IDF) ---------------------------------------------
def class_masks(labels: pd.DataFrame) -> dict[str, np.ndarray]:
    """Masque booléen des exercices de chaque tag, plus un pour les exercices sans tag cible."""
    masks = {tag: labels[tag].to_numpy() == 1 for tag in labels.columns}
    masks[NO_TAG] = labels.sum(axis=1).to_numpy() == 0
    return masks


def naive_class_tfidf(
    texts: pd.Series, labels: pd.DataFrame, **vectorizer_kwargs
) -> pd.DataFrame:
    """TF-IDF classique sur 9 documents de classe (textes concaténés par tag)."""
    docs = pd.Series(
        {name: " ".join(texts[mask]) for name, mask in class_masks(labels).items()}
    )
    vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    scores = vectorizer.fit_transform(docs)
    return pd.DataFrame(
        scores.toarray(), index=docs.index, columns=vectorizer.get_feature_names_out()
    )


def class_tfidf(
    texts: pd.Series, labels: pd.DataFrame, min_df: int = 5, **vectorizer_kwargs
) -> pd.DataFrame:
    """Score(terme, tag) = part des exercices du tag contenant le terme × IDF sur tous les exercices."""
    # Présence binaire : un template copié-collé compte une fois par exercice
    vectorizer = CountVectorizer(min_df=min_df, binary=True, **vectorizer_kwargs)
    presence = vectorizer.fit_transform(texts)
    # IDF sur tous les exercices et non sur 9 documents de classe, où elle serait quasi plate
    idf = np.log(presence.shape[0] / presence.sum(axis=0).A1)
    scores = {
        name: presence[mask].mean(axis=0).A1 * idf
        for name, mask in class_masks(labels).items()
    }
    return pd.DataFrame(scores, index=vectorizer.get_feature_names_out()).T


def top_terms(scores: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Les `n` termes les plus caractéristiques de chaque tag (une colonne par tag)."""
    return pd.DataFrame(
        {tag: row.nlargest(n).index.tolist() for tag, row in scores.iterrows()},
        index=pd.RangeIndex(1, n + 1, name="rang"),
    )


# --- Encodage de la difficulté & assemblage multi-blocs -----------------------
class DifficultyEncoder(BaseEstimator, TransformerMixin):
    """One-hot de la difficulté en tranches de quantiles (apprises au fit) + une colonne « inconnue »."""

    def __init__(self, n_bins: int = DEFAULT_FEATURE_CFG.difficulty_bins) -> None:
        self.n_bins = n_bins

    @staticmethod
    def _unknown(values: np.ndarray) -> np.ndarray:
        # -1 est une sentinelle du dataset (vu dans l'EDA), au même titre que NaN
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


def _add_raw_blocks(blocks: Blocks, df: pd.DataFrame, cfg: FeatureConfig) -> Blocks:
    """Blocs non encodés ici : leur transformation est apprise dans le pipeline, sur le train uniquement."""
    blocks[DIFFICULTY_BLOCK] = df[[cfg.difficulty_column]].to_numpy(dtype=float)
    blocks[TFIDF_CODE_BLOCK] = df[cfg.code_block[1]].fillna("").to_numpy(dtype=object)
    blocks[TFIDF_DESC_BLOCK] = (
        df[cfg.text_blocks[0][1]].fillna("").to_numpy(dtype=object)
    )
    return blocks


def build_blocks(df: pd.DataFrame, cfg: FeatureConfig = DEFAULT_FEATURE_CFG) -> Blocks:
    """Tous les blocs, alignés sur `df` ; embeddings lus depuis le cache ou calculés."""
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


def subset_blocks(blocks: Blocks, idx: np.ndarray) -> Blocks:
    """Sous-ensemble de lignes pour chaque bloc."""
    return {name: block[idx] for name, block in blocks.items()}


def concat_rows(*parts: Blocks) -> Blocks:
    """Empile les exemples de plusieurs jeux de blocs (ex. train + val)."""
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def concat_blocks(blocks: Blocks, names: list[str]) -> tuple[np.ndarray, FeatureLayout]:
    """Matrice dense des blocs `names` concaténés, et position de chaque bloc."""
    layout, start = {}, 0
    for name in names:
        layout[name] = slice(start, start + blocks[name].shape[1])
        start += blocks[name].shape[1]
    return np.hstack([blocks[name] for name in names]), layout


def prepare_descriptions(texts: np.ndarray) -> list[str]:
    """Nettoyage + lemmatisation d'un lot (fonction nommée pour la sérialisation sklearn)."""
    return [prepare_description(text) for text in texts]
