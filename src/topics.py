"""Clustering thématique BERTopic (UMAP + HDBSCAN) sur des embeddings pré-calculés."""

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src.config import SEED, TopicConfig

DEFAULT_TOPIC_CFG = TopicConfig()


def build_topic_model(cfg: TopicConfig = DEFAULT_TOPIC_CFG) -> BERTopic:
    """BERTopic avec UMAP (seed fixe) + HDBSCAN ; mots-outils anglais exclus des représentations."""
    umap_model = UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        n_components=cfg.umap_n_components,
        metric="cosine",
        random_state=SEED,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=CountVectorizer(
            stop_words="english", min_df=2, ngram_range=(1, 2)
        ),
        calculate_probabilities=False,
    )
