"""Chemins et constantes partagés par tout le projet."""

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATASET_DIR = DATA_DIR / "code_classification_dataset"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

DATASET_GDRIVE_ID = "1FyNPiEKkZBfhz4ua0eM3PQjbRWVwO95J"
EXPECTED_N_SAMPLES = 4982

TARGET_TAGS: list[str] = [
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


@dataclass(frozen=True)
class CleaningConfig:
    text_column: str = "prob_desc_description"
    language: str = "en"
    # Seuil validé dans 01_eda : au-delà, les paires inspectées sont des variantes d'un même énoncé
    near_dup_threshold: float = 0.8


@dataclass(frozen=True)
class SplitConfig:
    val_size: float = 0.15
    test_size: float = 0.15


@dataclass(frozen=True)
class RegexConfig:
    # Vocabulaire tiré de la TF-IDF par tag (train) ; valeurs réglées sur la validation (notebook 03)
    k_desc: int = 2
    k_code: int = 5
    # Un terme n'est retenu que s'il est min_lift fois plus marqué pour le tag que pour « aucun tag cible »
    desc_min_lift: float = 5.0
    # Plus strict pour le code : beaucoup de ses tokens sont génériques (pop, add, set…)
    code_min_lift: float = 5.0


@dataclass(frozen=True)
class MLPConfig:
    hidden_dim: int = 256
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 15  # early stopping sur la loss de validation


@dataclass(frozen=True)
class LogRegConfig:
    n_components: int | None = None  # None : pas de PCA, les 768 dimensions brutes
    C: float = 1.0  # inverse de la force de régularisation L2
    max_iter: int = 2000


@dataclass(frozen=True)
class TuningConfig:
    n_splits: int = 5  # validation croisée sur train+val ; le test n'est jamais utilisé
    n_bootstrap: int = 1000
    logreg_grid: tuple[tuple[str, tuple], ...] = (
        ("n_components", (8, 16, 32, 64, 128, 256, None)),
        ("C", (0.01, 0.1, 1.0, 10.0)),
    )
    mlp_grid: tuple[tuple[str, tuple], ...] = (
        ("lr", (1e-4, 3e-4, 1e-3, 3e-3)),
        ("weight_decay", (1e-4, 1e-2, 1e-1)),
        ("dropout", (0.2, 0.5)),
    )
    # 2e passe : l'optimum de la 1re grille est sur son bord (régularisation maximale), on prolonge
    mlp_refine_grid: tuple[tuple[str, tuple], ...] = (
        ("lr", (1e-3, 3e-3)),
        ("weight_decay", (0.1, 0.3, 1.0)),
        ("dropout", (0.5, 0.7)),
    )


@dataclass(frozen=True)
class FeatureConfig:
    # (nom du bloc, colonne source) : un embedding SBERT par colonne de texte, pour éviter la troncature
    text_blocks: tuple[tuple[str, str], ...] = (
        ("description", "prob_desc_description"),
        ("input_spec", "prob_desc_input_spec"),
        ("notes", "prob_desc_notes"),
    )
    code_block: tuple[str, str] = ("code", "source_code")
    difficulty_column: str = "difficulty"
    text_model: str = "sentence-transformers/all-mpnet-base-v2"
    code_model: str = "microsoft/unixcoder-base"
    text_batch_size: int = 32
    code_batch_size: int = 16
    # RoBERTa : 1026 positions dont 2 réservées ; à 1024 tokens, 9 % des codes sont tronqués (25 % à 512)
    code_max_seq_length: int = 1024
    difficulty_bins: int = 5
    # Blocs TF-IDF (vocabulaire et IDF appris sur le train de chaque fold) : ~1 700 et ~1 800 colonnes creuses
    tfidf_code_min_df: int = 5
    tfidf_desc_min_df: int = 10
    # Régularisation propre à chaque bloc TF-IDF (réglée en CV, notebook 05)
    tfidf_code_C: float = 0.1
    tfidf_desc_C: float = 1.0


# Meilleur modèle entraîné (notebook 05) : stacking des embeddings, de la difficulté et de la TF-IDF de description
BEST_STACKING_BLOCKS: tuple[str, ...] = (
    "description",
    "input_spec",
    "notes",
    "code",
    "difficulty",
    "tfidf_desc",
)


@dataclass(frozen=True)
class LLMConfig:
    # Projet GCP dédié, séparé des projets existants (créé par `python -m src.llm_tagger setup`)
    project_id: str = "illuin-codeforces-tags"
    # "global" route vers la capacité disponible : les modèles Gemini ne sont pas servis dans toutes les régions
    location: str = "global"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_code_chars: int = (
        6000  # les plus longues solutions sont surtout du template copié-collé
    )
    max_workers: int = 4
    max_retries: int = 5


LLM_PREDICTIONS_DIR = DATA_DIR / "llm"


@dataclass(frozen=True)
class TopicConfig:
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 32
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: int = 5
