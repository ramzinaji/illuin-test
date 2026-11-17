"""Nettoyage des textes : normalisation, lemmatisation, filtre de langue et quasi-doublons."""

import logging
import re
import unicodedata

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from nltk.stem import WordNetLemmatizer
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import LANGUAGE, NEAR_DUP_THRESHOLD

logger = logging.getLogger(__name__)

DetectorFactory.seed = 0  # langdetect est aléatoire sans graine
# Lettres cyrilliques glissées dans des énoncés anglais (vues pendant l'exploration)
HOMOGLYPHS = str.maketrans("АВЕКМНОРСТХаеорсухі", "ABEKMHOPCTXaeopcyxi")
_lemmatizer = WordNetLemmatizer()


def normalize_text(text: str | None) -> str:
    """Minuscules, homoglyphes translittérés, LaTeX `$$$` et ponctuation retirés ; champ absent → ""."""
    if not isinstance(text, str):  # None ou NaN (champ facultatif absent)
        return ""
    text = unicodedata.normalize("NFKC", text).translate(HOMOGLYPHS).lower()
    text = text.replace("$$$", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lemmatize(text: str) -> str:
    return " ".join(_lemmatizer.lemmatize(word) for word in text.split())


def detect_language(text: str) -> str:
    # Mots de 3+ lettres seulement : sinon les formules (a, b, n...) font classer l'anglais en catalan
    words = " ".join(re.findall(r"[^\W\d_]{3,}", text))
    try:
        return detect(words)
    except LangDetectException:
        return "unknown"


def add_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes de texte nettoyé utilisées par les features (à l'entraînement comme à l'inférence)."""
    out = df.copy()
    out["text_clean"] = out["prob_desc_description"].map(normalize_text)
    out["text_lemma"] = out["text_clean"].map(lemmatize)
    out["input_spec_clean"] = out["prob_desc_input_spec"].map(normalize_text)
    out["notes_clean"] = out["prob_desc_notes"].map(normalize_text)
    out["source_code"] = out["source_code"].fillna("")
    return out


def drop_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Garde un exemple (le plus petit sample_id) par groupe d'énoncés au cosinus TF-IDF > seuil."""
    tfidf = TfidfVectorizer(min_df=2).fit_transform(df["text_lemma"])
    # TF-IDF normalisé L2 : le produit scalaire est la similarité cosinus
    similar = (tfidf @ tfidf.T) > NEAR_DUP_THRESHOLD
    _, groups = connected_components(similar, directed=False)
    return (
        df.assign(dup_group=groups)
        .sort_index()
        .drop_duplicates("dup_group")
        .drop(columns="dup_group")
    )


def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Jeu d'entraînement : énoncés anglais, textes nettoyés, sans quasi-doublons."""
    english = raw[raw["prob_desc_description"].map(detect_language) == LANGUAGE]
    df = drop_near_duplicates(add_text_columns(english))
    logger.info(
        "Nettoyage : %d -> %d (langue) -> %d (quasi-doublons)",
        len(raw),
        len(english),
        len(df),
    )
    return df
