from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY_TARGET, CATEGORICAL_COLUMNS, MULTICLASS_TARGET


def split_binary_features_target(frame):
    x = frame.drop(columns=[BINARY_TARGET, MULTICLASS_TARGET], errors="ignore")
    y = frame[BINARY_TARGET]
    return x, y


def split_multiclass_features_target(frame):
    x = frame.drop(columns=[BINARY_TARGET, MULTICLASS_TARGET], errors="ignore")
    y = frame[MULTICLASS_TARGET].fillna("Normal")
    return x, y


def build_preprocessor(feature_frame):
    inferred_categorical_columns = feature_frame.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    categorical_columns = list(
        dict.fromkeys(
            [col for col in CATEGORICAL_COLUMNS if col in feature_frame.columns]
            + inferred_categorical_columns
        )
    )
    numeric_columns = [col for col in feature_frame.columns if col not in categorical_columns]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )
