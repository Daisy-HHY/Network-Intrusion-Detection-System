# UNSW-NB15 Intrusion Detection

Place the dataset files under `data/raw/`.

Expected files:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Main commands:
- `python -m src.ids_ml.pipeline_binary`
- `python -m src.ids_ml.pipeline_multiclass`
- `pytest -q`
