"""MLP multi-label (sorties sigmoïdes indépendantes) entraîné sur des embeddings figés."""

import copy
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import SEED, MLPConfig

logger = logging.getLogger(__name__)

DEFAULT_MLP_CFG = MLPConfig()


class TagMLP(nn.Module):
    """Une couche cachée ; renvoie des logits (sigmoïde appliquée dans la loss / à l'inférence)."""

    def __init__(
        self, input_dim: int, n_tags: int, cfg: MLPConfig = DEFAULT_MLP_CFG
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, n_tags),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32)


def _pos_weight(y: np.ndarray) -> torch.Tensor:
    # Compense le déséquilibre : un positif rare pèse neg/pos fois plus qu'un négatif
    positives = y.sum(axis=0).clip(min=1)
    return _to_tensor((len(y) - positives) / positives)


@torch.no_grad()
def predict_proba(model: TagMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(_to_tensor(x))).numpy()


def _run_epoch(
    model: TagMLP, loader: DataLoader, loss_fn: nn.Module, optimizer
) -> float:
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


@torch.no_grad()
def _eval_loss(
    model: TagMLP, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module
) -> float:
    model.eval()
    return loss_fn(model(x), y).item()


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: MLPConfig = DEFAULT_MLP_CFG,
    seed: int = SEED,
) -> tuple[TagMLP, pd.DataFrame]:
    """Entraîne avec early stopping sur la loss de val ; renvoie le meilleur modèle et l'historique."""
    # Entraînement sur CPU : petit modèle, et résultats reproductibles
    torch.manual_seed(seed)
    model = TagMLP(x_train.shape[1], y_train.shape[1], cfg)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y_train))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loader = DataLoader(
        TensorDataset(_to_tensor(x_train), _to_tensor(y_train)),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    xv, yv = _to_tensor(x_val), _to_tensor(y_val)

    history, best_loss, best_state, since_best = [], float("inf"), None, 0
    for epoch in range(cfg.max_epochs):
        train_loss = _run_epoch(model, loader, loss_fn, optimizer)
        val_loss = _eval_loss(model, xv, yv, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_state, since_best = (
                val_loss,
                copy.deepcopy(model.state_dict()),
                0,
            )
        else:
            since_best += 1
        if since_best >= cfg.patience:
            break

    model.load_state_dict(best_state)
    logger.info("Arrêt à l'epoch %d, meilleure val_loss=%.4f", epoch, best_loss)
    return model, pd.DataFrame(history)


def fit_predict_mlp(
    cfg: MLPConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> np.ndarray:
    """Interface de `tuning.grid_search` ; `(x_eval, y_eval)` sert aussi à l'early stopping."""
    model, _ = train_mlp(x_train, y_train, x_eval, y_eval, cfg)
    return predict_proba(model, x_eval)
