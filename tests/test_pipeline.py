"""Tests rapides : nettoyage, split, encodage de la difficulté, modèle, métriques (sans encodeur pré-entraîné)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import DEFAULT_CONFIG_PATH, TAGS, load_config
from src.data import label_matrix, split_labels
from src.evaluate import global_metrics
from src.features import DifficultyEncoder
from src.model import StackingModel, build_model, tune_thresholds
from src.predict import to_tag_lists
from src.preprocessing import add_text_columns, normalize_text


def test_normalize_text_removes_latex_homoglyphs_and_punctuation():
    # "Т" est un T cyrillique
    assert (
        normalize_text("Given a Тree with $$$n$$$ vertices!")
        == "given a tree with n vertices"
    )


def test_normalize_text_handles_missing_fields():
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""


def test_add_text_columns_lemmatizes_description():
    df = pd.DataFrame(
        {
            "prob_desc_description": ["Two vertices"],
            "prob_desc_input_spec": [None],
            "prob_desc_notes": [None],
            "source_code": ["print(1)"],
        }
    )
    row = add_text_columns(df).iloc[0]
    assert row["text_lemma"] == "two vertex"
    assert row["notes_clean"] == ""


def test_split_is_disjoint_and_stratified():
    rng = np.random.default_rng(0)
    Y = (rng.random((1000, len(TAGS))) < 0.2).astype(int)
    cfg = load_config()
    split = split_labels(Y, cfg.split, cfg.seed)
    assert set(split) == {"train", "val", "test"}
    assert abs((split == "test").mean() - 0.15) < 0.01
    # Chaque tag garde à peu près sa prévalence dans le test
    assert np.allclose(Y[split == "test"].mean(axis=0), Y.mean(axis=0), atol=0.02)


def test_label_matrix_keeps_only_target_tags():
    Y = label_matrix(pd.Series([["math", "dp"], [], ["games"]]))
    assert Y.shape == (3, len(TAGS))
    assert Y.sum() == 2


def test_difficulty_encoder_has_unknown_column():
    X = np.array([[800.0], [1500.0], [2500.0], [np.nan], [-1.0]])
    onehot = DifficultyEncoder(n_bins=3).fit(X).transform(X)
    assert onehot.shape == (5, 3 + 1)
    assert (onehot.sum(axis=1) == 1).all()
    assert onehot[3:, -1].tolist() == [1, 1]  # NaN et -1 : « inconnue »


def test_stacking_and_thresholds_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 300
    Y = (rng.random((n, len(TAGS))) < 0.3).astype(int)
    blocks = {
        "description": Y + rng.normal(scale=0.5, size=Y.shape),  # bloc informatif
        "code": rng.normal(size=(n, 4)),  # bloc de bruit
    }
    cfg = load_config()
    cfg = replace(cfg, model=replace(cfg.model, blocks=["description", "code"]))
    proba = StackingModel(cfg).fit(blocks, Y).predict_proba(blocks)
    assert proba.shape == Y.shape
    pred = (proba >= tune_thresholds(Y, proba, cfg.thresholds.grid)).astype(int)
    # Prédire tous les tags donne un F1 ≈ 0,46 (prévalence 30 %) : le modèle doit faire nettement mieux
    assert global_metrics(Y, pred)["macro_f1"] > 0.7


def test_to_tag_lists():
    pred = np.zeros((2, len(TAGS)), dtype=int)
    pred[0, TAGS.index("games")] = 1
    assert to_tag_lists(pred, TAGS) == [["games"], []]


def test_default_config_matches_production_choices():
    cfg = load_config()
    assert cfg.model.type == "stacking"
    assert set(cfg.model.blocks) <= set(cfg.model.level1.C)
    grid = cfg.thresholds.grid
    assert (grid[0], grid[-1], len(grid)) == (0.05, 0.95, 91)
    assert isinstance(build_model(cfg), StackingModel)


def _write_config(tmp_path, edit) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    edit(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_config_rejects_unknown_key(tmp_path):
    path = _write_config(tmp_path, lambda raw: raw["cleaning"].update(typo=1))
    with pytest.raises(TypeError):
        load_config(path)


def test_config_requires_a_C_for_each_block(tmp_path):
    path = _write_config(tmp_path, lambda raw: raw["model"]["level1"]["C"].pop("code"))
    with pytest.raises(ValueError, match="code"):
        load_config(path)
