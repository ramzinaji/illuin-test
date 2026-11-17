"""Classification des tags par un LLM (Gemini sur Vertex AI), en sortie JSON structurée.

Authentification : identifiants par défaut de `gcloud` (ADC), aucune clé à stocker.
Mise en place, une seule fois, dans un projet GCP dédié :

    python -m src.llm_tagger setup --billing-account <ID>   # crée le projet, lie la facturation, active Vertex AI
    python -m src.llm_tagger models                         # modèles Gemini disponibles
    python -m src.llm_tagger predict --split test --limit 5 # essai
    python -m src.llm_tagger predict --split test           # prédictions mises en cache (reprise automatique)
"""

import argparse
import json
import logging
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import LLM_PREDICTIONS_DIR, TARGET_TAGS, LLMConfig

logger = logging.getLogger(__name__)

DEFAULT_LLM_CFG = LLMConfig()

PROMPT = """You are an expert competitive programmer. Classify the Codeforces problem below.

Choose, among the 8 tags listed, every tag that the official Codeforces tagging would assign to this problem.
A problem can have several tags, or none of them (it often has other tags such as greedy or dp that are not listed).

Tags:
- "math": mathematical reasoning is central (formulas, arithmetic properties, combinatorial counting, modular arithmetic).
- "graphs": the problem is modeled as a graph (vertices/edges, connectivity, shortest paths, BFS/DFS on a graph).
- "strings": string processing is central (substrings, palindromes, string matching, character manipulation).
- "number theory": divisibility, primes, gcd/lcm, modular inverses, factorization.
- "trees": the structure is a tree (rooted tree, subtrees, LCA, tree DP).
- "geometry": points, lines, polygons, distances, areas, angles in the plane.
- "games": two players play optimally, find the winner (game theory).
- "probabilities": probabilities or expected values must be computed.

Be precise: only assign a tag when it is clearly part of the intended solution.

## Problem statement
{description}

## Input specification
{input_spec}

## Notes
{notes}

## Difficulty rating
{difficulty}

## Accepted Python solution
```python
{code}
```"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {"type": "array", "items": {"type": "string", "enum": TARGET_TAGS}},
        "justification": {
            "type": "string",
            "description": "one sentence explaining the choice",
        },
    },
    "required": ["tags", "justification"],
}


# --- Mise en place GCP ---------------------------------------------------------
def _gcloud(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    logger.info("gcloud %s", " ".join(args))
    return subprocess.run(
        ["gcloud", *args], capture_output=True, text=True, check=check
    )


def setup_gcp(billing_account: str, cfg: LLMConfig = DEFAULT_LLM_CFG) -> None:
    """Projet dédié (créé s'il n'existe pas), facturation liée, API Vertex AI activée. Idempotent."""
    exists = (
        _gcloud("projects", "describe", cfg.project_id, check=False).returncode == 0
    )
    if not exists:
        _gcloud("projects", "create", cfg.project_id, "--name=illuin codeforces tags")
    _gcloud(
        "billing",
        "projects",
        "link",
        cfg.project_id,
        f"--billing-account={billing_account}",
    )
    _gcloud(
        "services", "enable", "aiplatform.googleapis.com", f"--project={cfg.project_id}"
    )
    logger.info("Projet %s prêt (Vertex AI activé)", cfg.project_id)


# --- Appels au modèle ------------------------------------------------------------
def build_client(cfg: LLMConfig = DEFAULT_LLM_CFG):
    import google.auth  # imports locaux : dépendances inutiles hors de ce module
    from google import genai

    # Projet de quota forcé : sinon les ADC utilisent celui configuré par défaut (un autre projet),
    # et l'usage serait imputé à ce projet-là au lieu du projet dédié
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
        quota_project_id=cfg.project_id,
    )
    return genai.Client(
        vertexai=True,
        project=cfg.project_id,
        location=cfg.location,
        credentials=credentials,
    )


def list_models(client) -> list[str]:
    # Vertex nomme les modèles `publishers/google/models/<nom>` : on ne garde que le dernier segment
    return sorted(
        m.name.rsplit("/", 1)[-1] for m in client.models.list() if "gemini" in m.name
    )


def build_prompt(row: pd.Series, cfg: LLMConfig = DEFAULT_LLM_CFG) -> str:
    """Le LLM reçoit les mêmes colonnes que le meilleur modèle entraîné (énoncé, entrée, notes, difficulté, code)."""
    difficulty = row["difficulty"]
    return PROMPT.format(
        description=row["prob_desc_description"],
        input_spec=row["prob_desc_input_spec"] or "(none)",
        notes=row["prob_desc_notes"] or "(none)",
        difficulty="unknown"
        if pd.isna(difficulty) or difficulty <= 0
        else int(difficulty),
        code=row["source_code"][: cfg.max_code_chars],
    )


def classify(client, prompt: str, cfg: LLMConfig = DEFAULT_LLM_CFG) -> dict:
    """Un appel, sortie JSON contrainte par le schéma ; nouvelles tentatives espacées en cas d'erreur (quota…)."""
    from google.genai import types

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=cfg.temperature,
    )
    for attempt in range(cfg.max_retries):
        try:
            response = client.models.generate_content(
                model=cfg.model, contents=prompt, config=config
            )
            return json.loads(response.text)
        except (
            Exception
        ) as exc:  # erreurs réseau / quota de l'API : on réessaie avant d'abandonner
            if attempt == cfg.max_retries - 1:
                raise
            wait = 2**attempt
            logger.warning(
                "Échec (%s), nouvelle tentative dans %ds", type(exc).__name__, wait
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")


def predictions_path(split: str, cfg: LLMConfig = DEFAULT_LLM_CFG) -> Path:
    return LLM_PREDICTIONS_DIR / f"{cfg.model}__{split}.jsonl"


def load_predictions(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {record["sample_id"]: record for record in map(json.loads, f)}


def predict_dataframe(
    df: pd.DataFrame, path: Path, cfg: LLMConfig = DEFAULT_LLM_CFG
) -> dict[int, dict]:
    """Prédit chaque exemple absent du cache JSONL et l'y ajoute aussitôt : une interruption ne perd rien."""
    done = load_predictions(path)
    todo = df[~df.index.isin(done)]
    logger.info(
        "%d prédictions en cache, %d à faire (modèle %s)",
        len(done),
        len(todo),
        cfg.model,
    )
    if todo.empty:
        return done

    client, lock = build_client(cfg), threading.Lock()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f, ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {
            pool.submit(classify, client, build_prompt(row, cfg), cfg): sample_id
            for sample_id, row in todo.iterrows()
        }
        for i, future in enumerate(as_completed(futures), 1):
            sample_id = futures[future]
            try:
                record = {"sample_id": int(sample_id), **future.result()}
            # Un exemple en échec (après toutes les tentatives) ne doit pas faire perdre le reste du lot
            except Exception as exc:  # noqa: BLE001
                logger.error("Exemple %s abandonné : %s", sample_id, exc)
                continue
            with lock:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                done[int(sample_id)] = record
            if i % 50 == 0:
                logger.info("%d / %d", i, len(todo))
    return done


def to_label_matrix(predictions: dict[int, dict], sample_ids: pd.Index) -> np.ndarray:
    """Matrice binaire (n x 8) alignée sur `sample_ids` ; un exemple sans prédiction compte comme « aucun tag »."""
    rows = [set(predictions.get(int(i), {}).get("tags", [])) for i in sample_ids]
    return np.array([[tag in tags for tag in TARGET_TAGS] for tags in rows], dtype=int)


# --- Ligne de commande -----------------------------------------------------------
def _split_frame(split: str) -> pd.DataFrame:
    """Exemples d'un split, avec le même nettoyage et le même découpage que les autres modèles."""
    from src.data import load_raw_dataset
    from src.preprocessing import clean_dataset
    from src.split import build_label_matrix, train_val_test_split

    df = clean_dataset(load_raw_dataset())
    return df.loc[train_val_test_split(build_label_matrix(df["tags"]))[split]]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="crée le projet GCP dédié et active Vertex AI")
    setup.add_argument("--billing-account", required=True)
    sub.add_parser("models", help="liste les modèles Gemini disponibles")
    predict = sub.add_parser(
        "predict", help="prédit un split (cache JSONL, reprise automatique)"
    )
    predict.add_argument("--split", default="test", choices=["train", "val", "test"])
    predict.add_argument(
        "--limit",
        type=int,
        default=None,
        help="n'interroge que les N premiers exemples",
    )
    predict.add_argument("--model", default=DEFAULT_LLM_CFG.model)
    args = parser.parse_args()

    if args.command == "setup":
        setup_gcp(args.billing_account)
    elif args.command == "models":
        print("\n".join(list_models(build_client())))
    else:
        cfg = LLMConfig(model=args.model)
        df = _split_frame(args.split)
        predict_dataframe(
            df.head(args.limit) if args.limit else df,
            predictions_path(args.split, cfg),
            cfg,
        )


if __name__ == "__main__":
    main()
