"""Chargement des exercices JSON, matrice de labels et split train / val / test."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.config import SEED, TAGS, TEST_SIZE, VAL_SIZE


def _sample_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_samples(path: Path) -> pd.DataFrame:
    """Un fichier `sample_<id>.json`, ou un dossier de ces fichiers ; index = sample_id."""
    files = (
        [path] if path.is_file() else sorted(path.glob("sample_*.json"), key=_sample_id)
    )
    if not files:
        raise FileNotFoundError(f"Aucun fichier sample_*.json dans {path}")
    records = [json.loads(f.read_text()) for f in files]
    return pd.DataFrame(
        records, index=pd.Index([_sample_id(f) for f in files], name="sample_id")
    )


def label_matrix(tags: pd.Series) -> np.ndarray:
    """Matrice binaire (n x 8) des tags cibles."""
    return np.array([[tag in ts for tag in TAGS] for ts in tags], dtype=int)


def _stratified_split(Y: np.ndarray, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=SEED
    )
    return next(splitter.split(np.zeros(len(Y)), Y))


def split_labels(Y: np.ndarray) -> np.ndarray:
    """Nom du split de chaque exemple, stratifié sur les 8 tags (itératif, adapté au multi-label)."""
    rest, test = _stratified_split(Y, TEST_SIZE)
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)  # pour que val fasse 15 % du total
    _, val = _stratified_split(Y[rest], val_ratio)
    split = np.full(len(Y), "train", dtype=object)
    split[rest[val]] = "val"
    split[test] = "test"
    return split
