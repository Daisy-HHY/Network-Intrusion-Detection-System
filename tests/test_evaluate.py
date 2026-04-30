from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.ids_ml.config import FIGURES_DIR
from src.ids_ml.evaluate import build_confusion_matrix, compute_binary_metrics
from src.ids_ml.plotting import plot_metric_comparison


def test_compute_binary_metrics_returns_expected_keys():
    metrics = compute_binary_metrics([0, 1, 1], [0, 1, 0], [0.1, 0.8, 0.4])
    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }


def test_plot_metric_comparison_writes_file():
    frame = pd.DataFrame(
        [
            {"model": "rf", "accuracy": 0.9, "precision": 0.8, "recall": 0.85, "f1": 0.82}
        ]
    )
    output = FIGURES_DIR / f"test_metrics_{uuid4().hex}.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        plot_metric_comparison(frame, output)
        assert output.exists()
    finally:
        if output.exists():
            output.unlink()


def test_build_confusion_matrix_shape():
    cm = build_confusion_matrix([0, 1, 1], [0, 1, 0])
    assert cm.shape == (2, 2)
