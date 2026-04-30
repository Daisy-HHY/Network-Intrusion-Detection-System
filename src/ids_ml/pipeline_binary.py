import pandas as pd

from .config import FIGURES_DIR, MODELS_DIR
from .data import load_unsw_nb15
from .evaluate import build_confusion_matrix, compute_binary_metrics
from .plotting import plot_confusion, plot_feature_importance, plot_metric_comparison
from .preprocess import build_preprocessor, split_binary_features_target
from .train import fit_model, get_binary_models, save_model


def extract_feature_importance(pipeline, feature_names):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None

    values = model.feature_importances_
    frame = pd.DataFrame({"feature": feature_names, "importance": values})
    return frame.sort_values("importance", ascending=False).head(15)


def main():
    train_df, test_df = load_unsw_nb15()
    x_train, y_train = split_binary_features_target(train_df)
    x_test, y_test = split_binary_features_target(test_df)

    rows = []
    for model_name, estimator in get_binary_models().items():
        pipeline = fit_model(build_preprocessor(x_train), estimator, x_train, y_train)
        y_pred = pipeline.predict(x_test)
        y_score = pipeline.predict_proba(x_test)[:, 1]
        metrics = compute_binary_metrics(y_test, y_pred, y_score)
        rows.append({"model": model_name, **metrics})

        save_model(pipeline, MODELS_DIR / f"{model_name}_binary.joblib")
        cm = build_confusion_matrix(y_test, y_pred)
        plot_confusion(cm, ["Normal", "Attack"], FIGURES_DIR / f"{model_name}_binary_cm.png")
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        importance_df = extract_feature_importance(pipeline, feature_names)
        if importance_df is not None:
            importance_df.to_csv(FIGURES_DIR.parent / f"{model_name}_feature_importance.csv", index=False)
            plot_feature_importance(
                importance_df,
                FIGURES_DIR / f"{model_name}_feature_importance.png",
            )

    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    metrics_df.to_csv(FIGURES_DIR.parent / "binary_metrics.csv", index=False)
    plot_metric_comparison(metrics_df, FIGURES_DIR / "binary_metrics.png")


if __name__ == "__main__":
    main()
