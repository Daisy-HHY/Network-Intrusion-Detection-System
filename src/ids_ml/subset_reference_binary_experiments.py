import argparse
import time

import numpy as np
import pandas as pd

from .config import DROP_COLUMNS, RANDOM_STATE, TEST_FILE, TRAIN_FILE
from .evaluate import compute_binary_metrics
from .preprocess import build_preprocessor, split_binary_features_target
from .train import fit_model, get_binary_models, get_mlp_binary_model

TRAIN_SUBSET_ROWS = 160000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["xgboost_subset_144k", "mlp_subset_144k"],
        help="Optional subset of reference models to run.",
    )
    parser.add_argument(
        "--output",
        default="results/subset_reference_binary_experiments.csv",
        help="CSV path for experiment metrics.",
    )
    return parser.parse_args()


def split_mask(labels: np.ndarray, rng: np.random.Generator, validation_ratio: float = 0.1):
    zeros = np.where(labels == 0)[0]
    ones = np.where(labels == 1)[0]
    val_mask = np.zeros(len(labels), dtype=bool)
    val_mask[zeros] = rng.random(len(zeros)) < validation_ratio
    val_mask[ones] = rng.random(len(ones)) < validation_ratio
    return val_mask


def load_subset_frames():
    usecols = [
        "id",
        "dur",
        "proto",
        "service",
        "state",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "rate",
        "sttl",
        "dttl",
        "sload",
        "dload",
        "sloss",
        "dloss",
        "sinpkt",
        "dinpkt",
        "sjit",
        "djit",
        "swin",
        "stcpb",
        "dtcpb",
        "dwin",
        "tcprtt",
        "synack",
        "ackdat",
        "smean",
        "dmean",
        "trans_depth",
        "response_body_len",
        "ct_srv_src",
        "ct_state_ttl",
        "ct_dst_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_flw_http_mthd",
        "ct_src_ltm",
        "ct_srv_dst",
        "is_sm_ips_ports",
        "attack_cat",
        "label",
    ]
    dtype_map = {
        "proto": "category",
        "service": "category",
        "state": "category",
        "label": "int8",
        "id": "int32",
        "attack_cat": "category",
    }
    for column in usecols:
        if column not in dtype_map:
            dtype_map[column] = "float32"

    train_df = pd.read_csv(
        TRAIN_FILE,
        nrows=TRAIN_SUBSET_ROWS,
        encoding="utf-8-sig",
        usecols=usecols,
        dtype=dtype_map,
    ).drop(columns=DROP_COLUMNS, errors="ignore")
    test_df = pd.read_csv(
        TEST_FILE,
        encoding="utf-8-sig",
        usecols=usecols,
        dtype=dtype_map,
    ).drop(columns=DROP_COLUMNS, errors="ignore")
    return train_df, test_df


def main():
    args = parse_args()
    train_df, test_df = load_subset_frames()
    x_train_full, y_train_full = split_binary_features_target(train_df)
    x_test, y_test = split_binary_features_target(test_df)

    rng = np.random.default_rng(RANDOM_STATE)
    mask = split_mask(y_train_full.to_numpy(dtype=np.int8), rng)
    train_mask = ~mask
    x_train = x_train_full.loc[train_mask]
    y_train = y_train_full.loc[train_mask]

    models = {
        "xgboost_subset_144k": get_binary_models()["xgboost"],
        "mlp_subset_144k": get_mlp_binary_model(),
    }
    if args.models:
        models = {name: models[name] for name in args.models}

    rows = []
    for name, estimator in models.items():
        print(f"Running {name}...")
        start = time.time()
        pipeline = fit_model(
            build_preprocessor(x_train, dense_output=name.startswith("mlp")),
            estimator,
            x_train,
            y_train,
        )
        elapsed = round(time.time() - start, 2)
        y_pred = pipeline.predict(x_test)
        y_score = pipeline.predict_proba(x_test)[:, 1]
        rows.append(
            {
                "experiment": name,
                "train_seconds": elapsed,
                **compute_binary_metrics(y_test, y_pred, y_score),
            }
        )

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
