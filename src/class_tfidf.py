"""TF-IDF par classe (dans l'esprit du c-TF-IDF de BERTopic) : termes caractéristiques de chaque tag."""

import keyword

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

NO_TAG = "(aucun tag cible)"
# Identifiants Python d'au moins 2 caractères et nombres d'au moins 2 chiffres (ex. 998244353)
CODE_TOKEN_PATTERN = r"[A-Za-z_][A-Za-z0-9_]+|\d{2,}"
# Mots-clés Python et tokens d'entrées/sorties : présents dans toutes les solutions, sans lien avec le tag
CODE_STOP_WORDS = [
    *keyword.kwlist,
    *["sys", "stdin", "stdout", "readline", "input", "raw_input", "print", "write"],
    *["flush", "os", "io", "self"],
]


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
    """Score(terme, tag) = part des exercices du tag contenant le terme × IDF sur l'ensemble des exercices."""
    # Présence binaire : un template copié-collé compte une fois par exercice, pas autant de fois qu'il est répété
    vectorizer = CountVectorizer(min_df=min_df, binary=True, **vectorizer_kwargs)
    presence = vectorizer.fit_transform(texts)
    # IDF sur les ~4 800 exercices et non sur 9 documents de classe, où elle serait quasi plate
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
