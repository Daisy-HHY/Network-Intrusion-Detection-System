from pathlib import Path

import pandas as pd

from .config import FIGURES_DIR, RANDOM_STATE
from .data import load_unsw_nb15
from .plotting import plot_feature_importance
from .preprocess import build_preprocessor, split_binary_features_target
from .train import fit_model


def get_lightgbm_model():
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is optional. Install it with `pip install -r requirements-optional.txt`."
        ) from exc

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )


def extract_lightgbm_feature_importance(pipeline, top_n: int = 15):
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    return frame.sort_values("importance", ascending=False).head(top_n)


def main():
    train_df, _ = load_unsw_nb15()
    x_train, y_train = split_binary_features_target(train_df)
    pipeline = fit_model(
        build_preprocessor(x_train),
        get_lightgbm_model(),
        x_train,
        y_train,
    )

    importance_df = extract_lightgbm_feature_importance(pipeline)
    output_path = Path("results") / "lightgbm_feature_importance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False)
    plot_feature_importance(
        importance_df,
        FIGURES_DIR / "lightgbm_feature_importance.png",
    )
    print(output_path)


if __name__ == "__main__":
    main()
