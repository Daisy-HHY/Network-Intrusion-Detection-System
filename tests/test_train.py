import pandas as pd

from src.ids_ml.preprocess import build_preprocessor
from src.ids_ml.train import fit_model, get_binary_models


def test_get_binary_models_exposes_required_baselines():
    models = get_binary_models()
    assert set(models) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    }


def test_fit_model_trains_logistic_regression_pipeline():
    x_train = pd.DataFrame(
        {
            "dur": [0.1, 0.2, 0.3, 0.4],
            "sbytes": [10, 20, 30, 40],
            "proto": ["tcp", "udp", "tcp", "udp"],
            "service": ["http", "dns", "http", "dns"],
            "state": ["FIN", "INT", "FIN", "INT"],
        }
    )
    y_train = [0, 1, 0, 1]

    pipeline = fit_model(
        build_preprocessor(x_train),
        get_binary_models()["logistic_regression"],
        x_train,
        y_train,
    )

    assert pipeline.predict(x_train).shape[0] == 4
