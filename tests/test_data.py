from pathlib import Path
from src.ids_ml.config import RAW_DIR


def test_raw_dir_is_under_project_root():
    assert RAW_DIR.name == "raw"
    assert isinstance(RAW_DIR, Path)


def test_load_unsw_nb15_drops_id_column(monkeypatch):
    import pandas as pd

    from src.ids_ml.data import load_unsw_nb15

    fake = pd.DataFrame({"id": [1], "dur": [0.1], "label": [0]})

    monkeypatch.setattr("src.ids_ml.data.pd.read_csv", lambda _: fake.copy())
    train_df, test_df = load_unsw_nb15()

    assert "id" not in train_df.columns
    assert "id" not in test_df.columns
