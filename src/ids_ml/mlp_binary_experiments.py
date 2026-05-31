import time
from pathlib import Path

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .data import load_unsw_nb15
from .evaluate import compute_binary_metrics
from .preprocess import build_preprocessor, split_binary_features_target


def get_experiment_configs():
    return [
        {
            "name": "mlp_baseline",
            "params": {
                "hidden_layer_sizes": (128, 64),
                "activation": "relu",
                "learning_rate_init": 0.001,
                "max_iter": 50,
                "early_stopping": True,
                "n_iter_no_change": 5,
                "random_state": 42,
            },
        },
        {
            "name": "mlp_tuned_longer",
            "params": {
                "hidden_layer_sizes": (128, 64),
                "activation": "relu",
                "learning_rate_init": 0.0005,
                "alpha": 0.0005,
                "batch_size": 512,
                "max_iter": 120,
                "early_stopping": True,
                "n_iter_no_change": 10,
                "random_state": 42,
            },
        },
        {
            "name": "mlp_tuned_wider",
            "params": {
                "hidden_layer_sizes": (256, 128),
                "activation": "relu",
                "learning_rate_init": 0.0005,
                "alpha": 0.001,
                "batch_size": 512,
                "max_iter": 120,
                "early_stopping": True,
                "n_iter_no_change": 10,
                "random_state": 42,
            },
        },
    ]


def main():
    train_df, test_df = load_unsw_nb15()
    x_train, y_train = split_binary_features_target(train_df)
    x_test, y_test = split_binary_features_target(test_df)

    rows = []
    for config in get_experiment_configs():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(x_train, dense_output=True)),
                ("model", MLPClassifier(**config["params"])),
            ]
        )
        start = time.time()
        pipeline.fit(x_train, y_train)
        elapsed = time.time() - start

        model = pipeline.named_steps["model"]
        y_pred = pipeline.predict(x_test)
        y_score = pipeline.predict_proba(x_test)[:, 1]
        metrics = compute_binary_metrics(y_test, y_pred, y_score)

        rows.append(
            {
                "experiment": config["name"],
                "train_seconds": round(elapsed, 2),
                "n_iter": model.n_iter_,
                "best_validation_score": getattr(model, "best_validation_score_", None),
                "loss": float(model.loss_),
                **config["params"],
                **metrics,
            }
        )

    output_path = Path("results") / "mlp_improvement_experiments.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
