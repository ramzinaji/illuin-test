"""Smoke test de la CLI : nécessite le modèle entraîné et les encodeurs (ignoré sinon)."""

import json
import subprocess
import sys

import pytest

from src.config import MODEL_PATH, RAW_DATA_DIR, TAGS

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists() or not RAW_DATA_DIR.exists(),
    reason="modèle ou dataset absent : lancer `python -m src.cli train`",
)


def test_predict_outputs_one_json_line_per_example(tmp_path):
    sample = RAW_DATA_DIR / "sample_0.json"
    output = tmp_path / "predictions.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "src.cli",
        "predict",
        str(sample),
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    lines = output.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["sample_id"] == 0
    assert set(record["tags"]) <= set(TAGS)
