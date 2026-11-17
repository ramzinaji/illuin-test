"""Chemins, tags cibles et hyperparamètres du modèle de production (choisis dans notebooks_prod)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "code_classification_dataset"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
LLM_PREDICTIONS_DIR = ROOT / "data" / "llm"
MODEL_PATH = ROOT / "models" / "stacking.joblib"

TAGS = [
    "math",
    "graphs",
    "strings",
    "number theory",
    "trees",
    "geometry",
    "games",
    "probabilities",
]
SEED = 42

# Nettoyage et split (notebook 01)
LANGUAGE = "en"
NEAR_DUP_THRESHOLD = 0.8
VAL_SIZE, TEST_SIZE = 0.15, 0.15

# Features (notebooks 02 et 04)
TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"
CODE_MODEL = "microsoft/unixcoder-base"
CODE_MAX_TOKENS = 1024  # 9 % des codes tronqués à 1 024 tokens, 25 % à 512
TEXT_BATCH_SIZE, CODE_BATCH_SIZE = 32, 16
DIFFICULTY_BINS = 5
TFIDF_MIN_DF = 10

# Modèle retenu (notebook 04) : stacking des embeddings, de la difficulté et de la TF-IDF de description
BLOCKS = ["description", "input_spec", "notes", "code", "difficulty", "tfidf_desc"]
N_FOLDS = 5  # folds out-of-fold du stacking
THRESHOLD_GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]  # 0.05 → 0.95
