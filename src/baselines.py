"""Baselines sans modèle statistique : règles regex sur la description et le code."""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.class_tfidf import (
    CODE_STOP_WORDS,
    CODE_TOKEN_PATTERN,
    NO_TAG,
    class_tfidf,
)
from src.config import RegexConfig
from src.preprocessing import prepare_description

DEFAULT_REGEX_CFG = RegexConfig()
NEVER_MATCH = r"(?!)"

# Mots-clés écrits a priori (intuition métier), pas ajustés sur les données : c'est le plancher à battre
TAG_PATTERNS: dict[str, str] = {
    "math": r"\b(?:sum|product|divisible|modulo|equation|formula|integer)\b",
    "graphs": r"\b(?:graph|vertex|vertices|edges?|roads?|cities)\b",
    "strings": r"\b(?:strings?|substrings?|palindromes?|letters?|characters?)\b",
    "number theory": r"\b(?:primes?|gcd|lcm|divisors?|coprime|modulo)\b",
    "trees": r"\b(?:trees?|rooted|subtrees?|leaf|leaves)\b",
    "geometry": r"\b(?:polygons?|circles?|angles?|coordinates|triangles?|segments?)\b",
    "games": r"\b(?:game|players?|winner|wins|optimally)\b",
    "probabilities": r"\b(?:probability|expected|expectation|random)\b",
}


class RegexTagger:
    """Prédit un tag dès qu'un de ses mots-clés apparaît dans le texte."""

    def __init__(self, patterns: dict[str, str] = TAG_PATTERNS) -> None:
        self.tags = list(patterns)
        self._compiled = {
            tag: re.compile(p, re.IGNORECASE) for tag, p in patterns.items()
        }

    def predict(self, texts: pd.Series) -> np.ndarray:
        return np.column_stack(
            [texts.str.contains(self._compiled[tag]).to_numpy() for tag in self.tags]
        ).astype(int)


def select_tag_terms(
    scores: pd.DataFrame, k: int, min_lift: float
) -> dict[str, list[str]]:
    """Top-k termes TF-IDF de chaque tag, parmi ceux `min_lift` fois plus marqués que chez « aucun tag »."""
    reference = scores.loc[NO_TAG]
    tags = [tag for tag in scores.index if tag != NO_TAG]
    return {
        tag: scores.loc[tag][scores.loc[tag] > min_lift * reference]
        .nlargest(k)
        .index.tolist()
        for tag in tags
    }


def build_patterns(terms: dict[str, list[str]]) -> dict[str, str]:
    """Une regex « mot entier » par tag ; un tag sans terme ne se déclenche jamais."""
    return {
        tag: r"\b(?:" + "|".join(map(re.escape, words)) + r")\b"
        if words
        else NEVER_MATCH
        for tag, words in terms.items()
    }


class KeywordTagger:
    """Regex dont le vocabulaire vient de la TF-IDF par tag ; le code n'est utilisé que pour les tags où il aide.

    Les descriptions sont attendues **brutes** : elles sont nettoyées et lemmatisées ici, comme à l'entraînement.
    """

    def __init__(
        self, desc_terms: dict[str, list[str]], code_terms: dict[str, list[str]]
    ) -> None:
        self.desc_terms, self.code_terms = desc_terms, code_terms
        self._desc = RegexTagger(build_patterns(desc_terms))
        self._code = RegexTagger(build_patterns(code_terms))
        self.use_code = np.zeros(len(desc_terms), dtype=bool)

    def predict_by_source(
        self, descriptions: pd.Series, codes: pd.Series
    ) -> dict[str, np.ndarray]:
        """Prédictions de chaque source séparément et combinées (pour comparer leurs apports)."""
        from_desc = self._desc.predict(descriptions.map(prepare_description))
        from_code = self._code.predict(codes)
        return {
            "description": from_desc,
            "code": from_code,
            "description OU code": np.maximum(from_desc, from_code),
        }

    def predict(self, descriptions: pd.Series, codes: pd.Series) -> np.ndarray:
        sources = self.predict_by_source(descriptions, codes)
        return np.where(
            self.use_code, sources["description OU code"], sources["description"]
        )

    def choose_code_usage(
        self, descriptions: pd.Series, codes: pd.Series, y_val: np.ndarray
    ) -> None:
        """Par tag, active le code si « description OU code » bat « description seule » en F1 de validation."""
        sources = self.predict_by_source(descriptions, codes)
        self.use_code = np.array(
            [
                f1_score(
                    y_val[:, k], sources["description OU code"][:, k], zero_division=0
                )
                > f1_score(y_val[:, k], sources["description"][:, k], zero_division=0)
                for k in range(y_val.shape[1])
            ]
        )


def fit_keyword_tagger(
    descriptions: pd.Series,
    codes: pd.Series,
    labels: pd.DataFrame,
    cfg: RegexConfig = DEFAULT_REGEX_CFG,
) -> KeywordTagger:
    """Vocabulaire tiré de la TF-IDF par tag des données fournies (le train uniquement), descriptions lemmatisées."""
    desc_scores = class_tfidf(
        descriptions.map(prepare_description),
        labels,
        stop_words="english",
        ngram_range=(1, 2),
    )
    code_scores = class_tfidf(
        codes,
        labels,
        token_pattern=CODE_TOKEN_PATTERN,
        lowercase=False,
        stop_words=CODE_STOP_WORDS,
    )
    return KeywordTagger(
        select_tag_terms(desc_scores, cfg.k_desc, cfg.desc_min_lift),
        select_tag_terms(code_scores, cfg.k_code, cfg.code_min_lift),
    )
