import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import BINARY_TARGET, CATEGORICAL_COLUMNS, RANDOM_STATE
from .config import DROP_COLUMNS, TEST_FILE, TRAIN_FILE
from .preprocess import split_binary_features_target

TRAIN_SUBSET_ROWS = 160000
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def embedding_dim(cardinality: int) -> int:
    return min(32, max(4, int(round(1.6 * (cardinality**0.56)))))


class TabularDataset(Dataset):
    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x_num[index], self.x_cat[index], self.y[index]


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class TabularResNetBinary(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        categorical_cardinalities: list[int],
        width: int = 256,
        n_blocks: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_bn = nn.BatchNorm1d(numeric_dim)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, embedding_dim(cardinality)) for cardinality in categorical_cardinalities]
        )
        embedded_dim = sum(embedding_dim(cardinality) for cardinality in categorical_cardinalities)
        self.input = nn.Sequential(
            nn.Linear(numeric_dim + embedded_dim, width),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(n_blocks)])
        self.output = nn.Linear(width, 1)

    def forward(self, x_num, x_cat):
        x_num = self.num_bn(x_num)
        embedded = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embedded, dim=1)
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x).squeeze(1)


@dataclass
class ExperimentConfig:
    name: str
    width: int
    n_blocks: int
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    tune_threshold: bool


def split_mask(labels: np.ndarray, rng: np.random.Generator, validation_ratio: float = 0.1):
    zeros = np.where(labels == 0)[0]
    ones = np.where(labels == 1)[0]
    val_mask = np.zeros(len(labels), dtype=bool)
    val_mask[zeros] = rng.random(len(zeros)) < validation_ratio
    val_mask[ones] = rng.random(len(ones)) < validation_ratio
    return val_mask


def prepare_arrays():
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
    )
    test_df = pd.read_csv(
        TEST_FILE,
        encoding="utf-8-sig",
        usecols=usecols,
        dtype=dtype_map,
    )
    train_df = train_df.drop(columns=DROP_COLUMNS, errors="ignore")
    test_df = test_df.drop(columns=DROP_COLUMNS, errors="ignore")
    x_train_full, y_train_full = split_binary_features_target(train_df)
    x_test_full, y_test_full = split_binary_features_target(test_df)
    numeric_columns = [column for column in x_train_full.columns if column not in CATEGORICAL_COLUMNS]
    scaler = StandardScaler()
    rng = np.random.default_rng(RANDOM_STATE)
    labels = y_train_full.to_numpy(dtype=np.int8)
    val_mask = split_mask(labels, rng)
    train_mask = ~val_mask

    numeric_full = x_train_full[numeric_columns].to_numpy(dtype=np.float32)
    scaler.fit(numeric_full[train_mask])

    category_maps = {}
    categorical_cardinalities = []
    for column in CATEGORICAL_COLUMNS:
        values = sorted(x_train_full.loc[train_mask, column].astype(str).unique().tolist())
        category_maps[column] = {value: idx for idx, value in enumerate(values)}
        categorical_cardinalities.append(len(values))

    def encode_categories(frame: pd.DataFrame):
        arrays = []
        for column in CATEGORICAL_COLUMNS:
            mapping = category_maps[column]
            unknown_index = len(mapping)
            arrays.append(
                frame[column]
                .astype(str)
                .map(lambda value: mapping.get(value, unknown_index))
                .to_numpy(dtype=np.int64)
            )
        return np.stack(arrays, axis=1)

    return (
        categorical_cardinalities,
        scaler.transform(numeric_full[train_mask]).astype(np.float32),
        scaler.transform(numeric_full[val_mask]).astype(np.float32),
        scaler.transform(x_test_full[numeric_columns].to_numpy(dtype=np.float32)).astype(np.float32),
        encode_categories(x_train_full.loc[train_mask]),
        encode_categories(x_train_full.loc[val_mask]),
        encode_categories(x_test_full),
        labels[train_mask].astype(np.float32),
        labels[val_mask].astype(np.float32),
        y_test_full.to_numpy(dtype=np.float32),
    )


def predict_probabilities(model: nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()
    probabilities = []
    targets = []
    with torch.no_grad():
        for x_num, x_cat, y in data_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            logits = model(x_num, x_cat)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def run_experiment(config: ExperimentConfig, prepared_data):
    (
        categorical_cardinalities,
        x_train_num,
        x_val_num,
        x_test_num,
        x_train_cat,
        x_val_cat,
        x_test_cat,
        y_train,
        y_val,
        y_test,
    ) = prepared_data

    device = torch.device("cpu")
    train_loader = DataLoader(
        TabularDataset(x_train_num, x_train_cat, y_train),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TabularDataset(x_val_num, x_val_cat, y_val),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TabularDataset(x_test_num, x_test_cat, y_test),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = TabularResNetBinary(
        numeric_dim=x_train_num.shape[1],
        categorical_cardinalities=categorical_cardinalities,
        width=config.width,
        n_blocks=config.n_blocks,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state = None
    best_val_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for x_num, x_cat, y in train_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x_num, x_cat)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        val_scores, val_targets = predict_probabilities(model, val_loader, device)
        val_auc = roc_auc_score(val_targets, val_scores)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    train_seconds = round(time.time() - start, 2)
    model.load_state_dict(best_state)

    val_scores, val_targets = predict_probabilities(model, val_loader, device)
    threshold = 0.5
    val_best_f1 = f1_score(val_targets, (val_scores >= threshold).astype(int), zero_division=0)
    if config.tune_threshold:
        threshold, val_best_f1 = best_f1_threshold(val_targets, val_scores)

    test_scores, test_targets = predict_probabilities(model, test_loader, device)
    test_pred = (test_scores >= threshold).astype(int)
    return {
        "experiment": config.name,
        "train_seconds": train_seconds,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "selected_threshold": threshold,
        "val_best_f1": val_best_f1,
        "accuracy": accuracy_score(test_targets, test_pred),
        "precision": precision_score(test_targets, test_pred, zero_division=0),
        "recall": recall_score(test_targets, test_pred, zero_division=0),
        "f1": f1_score(test_targets, test_pred, zero_division=0),
        "roc_auc": roc_auc_score(test_targets, test_scores),
        "width": config.width,
        "n_blocks": config.n_blocks,
        "dropout": config.dropout,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "tune_threshold": config.tune_threshold,
    }


def main():
    set_seed(RANDOM_STATE)
    prepared_data = prepare_arrays()
    configs = [
        ExperimentConfig(
            name="tabular_resnet_default",
            width=256,
            n_blocks=2,
            dropout=0.15,
            lr=1e-3,
            weight_decay=1e-5,
            batch_size=2048,
            max_epochs=20,
            patience=4,
            tune_threshold=False,
        ),
        ExperimentConfig(
            name="tabular_resnet_threshold_tuned",
            width=256,
            n_blocks=3,
            dropout=0.20,
            lr=8e-4,
            weight_decay=5e-5,
            batch_size=2048,
            max_epochs=25,
            patience=5,
            tune_threshold=True,
        ),
    ]

    rows = [run_experiment(config, prepared_data) for config in configs]
    output_path = Path("results") / "tabular_dl_binary_experiments.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
