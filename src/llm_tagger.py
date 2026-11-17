"""Tags prédits par un LLM zero-shot (Gemini sur Vertex AI), pour comparaison au modèle entraîné.

Authentification par les identifiants par défaut de gcloud (`gcloud auth application-default login`) ;
le projet GCP est lu dans la variable d'environnement GOOGLE_CLOUD_PROJECT.

    python -m src.llm_tagger --limit 5      # essai sur 5 exercices du split test
    python -m src.llm_tagger                # tout le split test (reprend là où il s'était arrêté)
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.config import LLM_PREDICTIONS_DIR, RAW_DATA_DIR, TAGS
from src.data import label_matrix, load_samples, split_labels
from src.preprocessing import clean_dataset

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-flash-lite"
LOCATION = "global"  # les modèles Gemini ne sont pas servis dans toutes les régions
MAX_CODE_CHARS = (
    6000  # les solutions les plus longues sont surtout du template copié-collé
)
MAX_RETRIES = 5
MAX_WORKERS = 4

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

# Sortie JSON contrainte : le modèle ne peut répondre qu'avec des tags de la liste
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {"type": "array", "items": {"type": "string", "enum": TAGS}},
        "justification": {
            "type": "string",
            "description": "one sentence explaining the choice",
        },
    },
    "required": ["tags", "justification"],
}


def build_prompt(row: pd.Series) -> str:
    """Le LLM reçoit les mêmes colonnes que le modèle entraîné."""
    difficulty = row["difficulty"]
    return PROMPT.format(
        description=row["prob_desc_description"],
        input_spec=row["prob_desc_input_spec"] or "(none)",
        notes=row["prob_desc_notes"] or "(none)",
        difficulty="unknown"
        if pd.isna(difficulty) or difficulty <= 0
        else int(difficulty),
        code=row["source_code"][:MAX_CODE_CHARS],
    )


def make_client():
    import google.auth  # imports locaux : dépendances inutiles hors de ce module
    from google import genai

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    # Sans quota_project_id, l'usage serait imputé au projet gcloud par défaut et non à `project`
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
        quota_project_id=project,
    )
    return genai.Client(
        vertexai=True, project=project, location=LOCATION, credentials=credentials
    )


def classify(client, prompt: str, model: str) -> dict:
    """Un appel au LLM ; on réessaie (attente exponentielle) seulement sur quota dépassé ou erreur serveur."""
    from google.genai import errors, types

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=0.0,
    )
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model, contents=prompt, config=config
            )
            return json.loads(response.text)
        except errors.APIError as exc:
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt == MAX_RETRIES - 1:
                raise
            logger.warning("Erreur %s, nouvel essai dans %d s", exc.code, 2**attempt)
            time.sleep(2**attempt)


def classify_or_skip(client, row: pd.Series, model: str) -> dict | None:
    """Un exemple en échec (après les nouveaux essais) est journalisé et sauté, sans arrêter le lot."""
    from google.genai import errors

    try:
        return classify(client, build_prompt(row), model)
    except (errors.APIError, json.JSONDecodeError) as exc:
        logger.error("Exemple %s abandonné : %s", row.name, exc)
        return None


def predict_split(df: pd.DataFrame, model: str, output: Path) -> None:
    """Prédit les exemples absents de `output` (JSONL) et les y ajoute au fil de l'eau : une interruption ne perd rien."""
    done = set()
    if output.exists():
        done = {
            json.loads(line)["sample_id"] for line in output.read_text().splitlines()
        }
    todo = df[~df.index.isin(done)]
    logger.info(
        "%d exemples déjà prédits, %d à faire (modèle %s)", len(done), len(todo), model
    )
    if todo.empty:
        return

    client = make_client()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = pool.map(
            lambda row: classify_or_skip(client, row, model),
            (row for _, row in todo.iterrows()),
        )
        # pool.map rend les résultats dans l'ordre ; l'écriture se fait dans ce seul thread
        for i, (sample_id, result) in enumerate(zip(todo.index, results), 1):
            if result is None:
                continue
            f.write(
                json.dumps({"sample_id": int(sample_id), **result}, ensure_ascii=False)
                + "\n"
            )
            f.flush()
            if i % 50 == 0:
                logger.info("%d / %d", i, len(todo))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--limit", type=int, help="ne traite que les N premiers exercices"
    )
    args = parser.parse_args()

    df = clean_dataset(load_samples(RAW_DATA_DIR))
    test = df[split_labels(label_matrix(df["tags"])) == "test"]
    predict_split(
        test.head(args.limit) if args.limit else test,
        args.model,
        LLM_PREDICTIONS_DIR / f"{args.model}__test.jsonl",
    )


if __name__ == "__main__":
    main()
