"""Nettoyage des descriptions : normalisation, filtre de langue, dédoublonnage."""

import logging
import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from nltk.stem import WordNetLemmatizer
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CleaningConfig

logger = logging.getLogger(__name__)

DEFAULT_CLEANING_CFG = CleaningConfig()
DetectorFactory.seed = 0  # langdetect est non déterministe sans seed

# Lettres cyrilliques glissées dans des énoncés anglais (vu dans 01_eda) : on les translittère
_HOMOGLYPHS = str.maketrans("АВЕКМНОРСТХаеорсухі", "ABEKMHOPCTXaeopcyxi")
_LATEX_DELIM = re.compile(r"\${3}")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")
# Mots alphabétiques (toutes langues) de 3+ lettres
_WORDS = re.compile(r"[^\W\d_]{3,}")
_LEMMATIZER = WordNetLemmatizer()


def normalize_text(text: str) -> str:
    """Minuscules, homoglyphes translittérés, caractères spéciaux et LaTeX `$$$` retirés."""
    text = unicodedata.normalize("NFKC", text).translate(_HOMOGLYPHS)
    text = _LATEX_DELIM.sub(" ", text.lower())
    text = _NON_ALNUM.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def detect_language(text: str) -> str:
    """Code langue ISO ('en', 'ru'...) ; 'unknown' si la détection échoue."""
    # Sans ce filtre, les formules (variables d'une lettre, nombres) font classer des énoncés anglais en 'ca'
    words = " ".join(_WORDS.findall(text))
    try:
        return detect(words)
    except LangDetectException:
        return "unknown"


def lemmatize(text: str, lemmatizer: WordNetLemmatizer | None = None) -> str:
    """Lemmatisation mot à mot (WordNet) d'un texte déjà normalisé."""
    lemmatizer = lemmatizer or _LEMMATIZER
    return " ".join(lemmatizer.lemmatize(token) for token in text.split())


@lru_cache(maxsize=16_384)
def prepare_description(text: str) -> str:
    """Texte brut -> normalisé -> lemmatisé : même traitement à l'entraînement et à l'inférence."""
    # Cache : la recherche sur grille du notebook 03 retraite plusieurs fois les mêmes énoncés
    return lemmatize(normalize_text(text))


def near_duplicate_groups(texts: pd.Series, threshold: float) -> np.ndarray:
    """Identifiant de groupe par texte ; deux textes au cosinus TF-IDF > `threshold` partagent un groupe."""
    # TF-IDF est L2-normalisé : le produit X·Xᵀ (scipy.sparse) donne directement la similarité cosinus
    tfidf = TfidfVectorizer(min_df=2).fit_transform(texts)
    similarity = (tfidf @ tfidf.T).tocsr()
    similarity.data = similarity.data > threshold
    similarity.eliminate_zeros()
    _, labels = connected_components(similarity, directed=False)
    return labels


def filter_language(
    df: pd.DataFrame, cfg: CleaningConfig = DEFAULT_CLEANING_CFG
) -> pd.DataFrame:
    """Ajoute la colonne `language` et ne garde que les énoncés dans `cfg.language`."""
    out = df.assign(language=df[cfg.text_column].map(detect_language))
    kept = out[out["language"] == cfg.language]
    logger.info("Filtre langue : %d -> %d exemples", len(df), len(kept))
    return kept


def add_clean_text(
    df: pd.DataFrame, cfg: CleaningConfig = DEFAULT_CLEANING_CFG
) -> pd.DataFrame:
    """Ajoute `text_clean` (normalisé) et `text_lemma` (normalisé + lemmatisé)."""
    clean = df[cfg.text_column].map(normalize_text)
    return df.assign(text_clean=clean, text_lemma=clean.map(lemmatize))


def assign_dup_groups(
    df: pd.DataFrame, cfg: CleaningConfig = DEFAULT_CLEANING_CFG
) -> pd.DataFrame:
    """Ajoute `dup_group`, calculé sur le texte lemmatisé."""
    return df.assign(
        dup_group=near_duplicate_groups(df["text_lemma"], cfg.near_dup_threshold)
    )


def drop_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Garde un exemple par `dup_group` : le plus petit `sample_id` (choix déterministe)."""
    kept = df.sort_index().drop_duplicates("dup_group", keep="first")
    logger.info("Dédoublonnage : %d -> %d exemples", len(df), len(kept))
    return kept


def clean_dataset(
    df: pd.DataFrame, cfg: CleaningConfig = DEFAULT_CLEANING_CFG
) -> pd.DataFrame:
    """Pipeline complet : langue -> normalisation/lemmatisation -> suppression des quasi-doublons."""
    df = add_clean_text(filter_language(df, cfg), cfg)
    return drop_near_duplicates(assign_dup_groups(df, cfg))
