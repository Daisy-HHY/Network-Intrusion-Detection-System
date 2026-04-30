from pathlib import Path

import pandas as pd

from .config import DROP_COLUMNS, TEST_FILE, TRAIN_FILE


def load_unsw_nb15() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    train_df = train_df.drop(columns=DROP_COLUMNS, errors="ignore")
    test_df = test_df.drop(columns=DROP_COLUMNS, errors="ignore")
    return train_df, test_df


def save_frame(frame: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
