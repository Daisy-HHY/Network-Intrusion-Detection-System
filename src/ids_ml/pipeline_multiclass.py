import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from .config import FIGURES_DIR, MODELS_DIR
from .data import load_unsw_nb15
from .preprocess import build_preprocessor, split_multiclass_features_target
from .train import fit_model, get_multiclass_models, save_model


def main():
    train_df, test_df = load_unsw_nb15()
    x_train, y_train = split_multiclass_features_target(train_df)
    x_test, y_test = split_multiclass_features_target(test_df)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    rows = []
    for model_name, estimator in get_multiclass_models().items():
        target_train = y_train_encoded if model_name == "xgboost" else y_train
        pipeline = fit_model(build_preprocessor(x_train), estimator, x_train, target_train)
        y_pred = pipeline.predict(x_test)
        if model_name == "xgboost":
            y_pred = label_encoder.inverse_transform(y_pred.astype(int))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        rows.append(
            {
                "model": model_name,
                "accuracy": report["accuracy"],
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
            }
        )
        save_model(pipeline, MODELS_DIR / f"{model_name}_multiclass.joblib")

    metrics_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    metrics_df.to_csv(FIGURES_DIR.parent / "multiclass_metrics.csv", index=False)


if __name__ == "__main__":
    main()
