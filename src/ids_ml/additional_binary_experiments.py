import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .data import load_unsw_nb15
from .evaluate import compute_binary_metrics
from .preprocess import build_preprocessor, split_binary_features_target
from .train import fit_model, get_binary_models


@dataclass(frozen=True)
class ModelSpec:
    estimator: object
    dense_output: bool = False


def get_additional_binary_model_specs() -> dict[str, ModelSpec]:
    return {
        "xgboost_reference": ModelSpec(get_binary_models()["xgboost"]),
        "extra_trees": ModelSpec(
            ExtraTreesClassifier(
                n_estimators=300,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
        "extra_trees_balanced": ModelSpec(
            ExtraTreesClassifier(
                n_estimators=300,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
        "hist_gradient_boosting": ModelSpec(
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=RANDOM_STATE,
            ),
            dense_output=True,
        ),
        "gradient_boosting": ModelSpec(
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
            dense_output=True,
        ),
        "ada_boost_tree": ModelSpec(
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
                n_estimators=200,
                learning_rate=0.5,
                random_state=RANDOM_STATE,
            ),
            dense_output=True,
        ),
        "linear_svm_sgd": ModelSpec(
            SGDClassifier(
                loss="modified_huber",
                alpha=0.0001,
                max_iter=1000,
                tol=0.001,
                random_state=RANDOM_STATE,
            ),
            dense_output=True,
        ),
        "gaussian_nb": ModelSpec(GaussianNB(), dense_output=True),
    }


def get_third_party_binary_model_specs() -> dict[str, ModelSpec]:
    specs = {}

    try:
        from lightgbm import LGBMClassifier

        specs["lightgbm"] = ModelSpec(
            LGBMClassifier(
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
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        specs["catboost"] = ModelSpec(
            CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=RANDOM_STATE,
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
            ),
            dense_output=True,
        )
    except ImportError:
        pass

    return specs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional subset of model names to run. Defaults to all additional models.",
    )
    parser.add_argument(
        "--include-third-party",
        action="store_true",
        help="Include optional LightGBM and CatBoost baselines when installed.",
    )
    parser.add_argument(
        "--output",
        default="results/additional_ml_binary_experiments.csv",
        help="CSV path for experiment metrics.",
    )
    return parser.parse_args()


def select_model_specs(
    specs: dict[str, ModelSpec],
    selected_names: Optional[list[str]],
) -> dict[str, ModelSpec]:
    if not selected_names:
        return specs

    unknown = sorted(set(selected_names) - set(specs))
    if unknown:
        available = ", ".join(sorted(specs))
        raise ValueError(f"Unknown model(s): {unknown}. Available models: {available}")

    return {name: specs[name] for name in selected_names}


def main():
    args = parse_args()
    train_df, test_df = load_unsw_nb15()
    x_train, y_train = split_binary_features_target(train_df)
    x_test, y_test = split_binary_features_target(test_df)

    available_model_specs = get_additional_binary_model_specs()
    if args.include_third_party:
        third_party_specs = get_third_party_binary_model_specs()
        if not third_party_specs:
            print("No optional third-party models are installed.")
        available_model_specs.update(third_party_specs)

    model_specs = select_model_specs(available_model_specs, args.models)

    rows = []
    for model_name, spec in model_specs.items():
        print(f"Running {model_name}...")
        start = time.time()
        pipeline = fit_model(
            build_preprocessor(x_train, dense_output=spec.dense_output),
            spec.estimator,
            x_train,
            y_train,
        )
        train_seconds = round(time.time() - start, 2)

        y_pred = pipeline.predict(x_test)
        y_score = pipeline.predict_proba(x_test)[:, 1]
        rows.append(
            {
                "model": model_name,
                "train_seconds": train_seconds,
                **compute_binary_metrics(y_test, y_pred, y_score),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("f1", ascending=False).to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
