"""Téléchargement et chargement du dataset brut (un fichier JSON par exercice)."""

import logging
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, DATASET_GDRIVE_ID, EXPECTED_N_SAMPLES, RAW_DATASET_DIR

logger = logging.getLogger(__name__)


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
