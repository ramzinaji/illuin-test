"""Ligne de commande : entraînement, prédiction et évaluation.

python -m src.cli train [--config fichier.yaml]  # entraîne et sauvegarde models/stacking.joblib
python -m src.cli predict <fichier|dossier>  # tags prédits, une ligne JSON par exercice
python -m src.cli evaluate [dossier]         # métriques ; sans dossier : split test du dataset
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG_PATH, MODEL_PATH, RAW_DATA_DIR, Config
from src.data import label_matrix, load_samples, split_labels
from src.evaluate import global_metrics, per_tag_metrics
from src.features import load_encoders
from src.model import TagClassifier
from src.predict import predict, to_tag_lists
from src.preprocessing import clean_dataset
from src.train import train

logger = logging.getLogger("src")


def _test_split(cfg: Config) -> pd.DataFrame:
    """Le split test du dataset, avec le nettoyage et le découpage de la config d'entraînement."""
    df = clean_dataset(load_samples(RAW_DATA_DIR), cfg.cleaning)
    return df[split_labels(label_matrix(df["tags"]), cfg.split, cfg.seed) == "test"]


def run_predict(path: Path, output: Path | None) -> None:
    df = load_samples(path)
    classifier = TagClassifier.load(MODEL_PATH)
    encoders = load_encoders(classifier.config.encoders)
    predictions, _ = predict(df, classifier, encoders)
    lines = [
        json.dumps({"sample_id": int(sid), "tags": tags}, ensure_ascii=False)
        for sid, tags in zip(df.index, to_tag_lists(predictions, classifier.tags))
    ]
    if output is None:
        print("\n".join(lines))
    else:
        output.write_text("\n".join(lines) + "\n")
        logger.info("%d prédictions écrites dans %s", len(lines), output)


def run_evaluate(path: Path | None) -> None:
    classifier = TagClassifier.load(MODEL_PATH)
    df = load_samples(path) if path else _test_split(classifier.config)
    encoders = load_encoders(classifier.config.encoders)
    predictions, latencies = predict(df, classifier, encoders)
    Y = label_matrix(df["tags"])
    print(
        f"{len(df)} exercices | latence médiane {np.median(latencies) * 1000:.0f} ms, max {latencies.max() * 1000:.0f} ms\n"
    )
    print(pd.Series(global_metrics(Y, predictions)).round(3).to_string(), "\n")
    print(per_tag_metrics(Y, predictions).round(3).to_string())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train_cmd = commands.add_parser(
        "train", help="entraîne le modèle et l'évalue sur le split test"
    )
    train_cmd.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="fichier YAML d'hyperparamètres",
    )
    predict_cmd = commands.add_parser(
        "predict", help="prédit les tags de fichiers sample_*.json"
    )
    predict_cmd.add_argument("path", type=Path)
    predict_cmd.add_argument(
        "--output", type=Path, help="fichier JSONL de sortie (défaut : sortie standard)"
    )
    evaluate_cmd = commands.add_parser(
        "evaluate", help="évalue sur un dossier annoté (défaut : split test)"
    )
    evaluate_cmd.add_argument("path", type=Path, nargs="?")
    args = parser.parse_args()

    if args.command == "train":
        train(args.config)
    elif args.command == "predict":
        run_predict(args.path, args.output)
    else:
        run_evaluate(args.path)


if __name__ == "__main__":
    main()
