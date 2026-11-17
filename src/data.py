"""Chargement du dataset brut et assemblage du jeu de modélisation (données, embeddings, split)."""

import logging
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    DATASET_GDRIVE_ID,
    EMBEDDINGS_DIR,
    EXPECTED_N_SAMPLES,
    RAW_DATASET_DIR,
    TopicConfig,
)
from src.features import (
    compute_embeddings,
    embeddings_cache_path,
    load_cached_embeddings,
    load_encoder,
)
from src.preprocessing import clean_dataset
from src.split import build_label_matrix, train_val_test_split

logger = logging.getLogger(__name__)
DEFAULT_TOPIC_CFG = TopicConfig()


def download_dataset(dest_dir: Path = DATA_DIR, force: bool = False) -> Path:
    """Télécharge l'archive Google Drive du sujet et l'extrait dans `dest_dir`."""
    import gdown  # import local : dépendance inutile hors téléchargement

    target = dest_dir / RAW_DATASET_DIR.name
    if target.exists() and not force:
        logger.info("Dataset déjà présent dans %s", target)
        return target

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "dataset.zip"
        gdown.download(id=DATASET_GDRIVE_ID, output=str(archive), quiet=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    logger.info("Dataset extrait dans %s", target)
    return target


def _sample_id(path: Path) -> int:
    return int(path.stem.split("_")[1])


def load_raw_dataset(data_dir: Path = RAW_DATASET_DIR) -> pd.DataFrame:
    """Charge tous les `sample_*.json` en DataFrame, triés par identifiant de fichier."""
    files = sorted(data_dir.glob("sample_*.json"), key=_sample_id)
    # dtype/convert_dates désactivés : on garde les champs bruts (ex. timestamps en str)
    samples = [
        pd.read_json(path, typ="series", dtype=False, convert_dates=False)
        for path in files
    ]
    df = pd.DataFrame(
        samples, index=pd.Index([_sample_id(p) for p in files], name="sample_id")
    )
    if len(df) != EXPECTED_N_SAMPLES:
        logger.warning("%d exemples chargés, %d attendus", len(df), EXPECTED_N_SAMPLES)
    return df


@dataclass
class ModelingData:
    df: pd.DataFrame
    # Embeddings de la description par split ("train", "val", "test")
    x: dict[str, np.ndarray]
    y: dict[str, np.ndarray]  # labels binaires par split
    positions: dict[str, np.ndarray]  # positions des exemples de chaque split dans `df`

    def dev(self) -> tuple[np.ndarray, np.ndarray]:
        """Train + val concaténés, pour la validation croisée (le test reste à part)."""
        return np.vstack([self.x["train"], self.x["val"]]), np.vstack(
            [self.y["train"], self.y["val"]]
        )

    def split(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Découpe une matrice alignée sur `df` selon le split train/val/test."""
        return {name: features[p] for name, p in self.positions.items()}


def load_modeling_data(cfg: TopicConfig = DEFAULT_TOPIC_CFG) -> ModelingData:
    """Nettoie, charge (ou calcule) les embeddings de la description et découpe en train/val/test."""
    df = clean_dataset(load_raw_dataset())
    sample_ids = df.index.to_numpy()
    cache_path = embeddings_cache_path(
        EMBEDDINGS_DIR, cfg.embedding_model, "prob_desc_description"
    )
    embeddings = load_cached_embeddings(cache_path, sample_ids, cfg.embedding_model)
    if embeddings is None:
        encoder = load_encoder(cfg.embedding_model)
        embeddings = compute_embeddings(
            df["text_clean"].tolist(),
            sample_ids,
            encoder,
            cache_path,
            cfg.embedding_model,
            cfg.batch_size,
        )

    labels = build_label_matrix(df["tags"])
    positions = {
        name: df.index.get_indexer(idx)
        for name, idx in train_val_test_split(labels).items()
    }
    return ModelingData(
        df=df,
        x={name: embeddings[p] for name, p in positions.items()},
        y={name: labels.iloc[p].to_numpy() for name, p in positions.items()},
        positions=positions,
    )
