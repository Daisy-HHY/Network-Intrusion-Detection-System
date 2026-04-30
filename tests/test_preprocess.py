import pandas as pd

from src.ids_ml.preprocess import build_preprocessor, split_binary_features_target


def test_split_binary_features_target_separates_label_columns():
    frame = pd.DataFrame(
        {
            "dur": [0.1, 0.2],
            "proto": ["tcp", "udp"],
            "label": [0, 1],
            "attack_cat": ["Normal", "Generic"],
        }
    )

    features, target = split_binary_features_target(frame)

    assert "label" not in features.columns
    assert "attack_cat" not in features.columns
    assert target.tolist() == [0, 1]


def test_build_preprocessor_transforms_dataframe():
    features = pd.DataFrame(
        {
            "dur": [0.1, 0.2],
            "sbytes": [10, 20],
            "proto": ["tcp", "udp"],
            "service": ["http", "dns"],
            "state": ["FIN", "INT"],
        }
    )

    transformer = build_preprocessor(features)
    matrix = transformer.fit_transform(features)

    assert matrix.shape[0] == 2
