# UNSW-NB15 Intrusion Detection

Place the dataset files under `data/raw/`.

Expected files:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Main commands:
- `python -m src.ids_ml.pipeline_binary`
- `python -m src.ids_ml.pipeline_multiclass`
- `python -m src.ids_ml.additional_binary_experiments`
- `pytest -q`

## Run Order

1. Install dependencies in the target environment: `pip install -r requirements.txt`
2. Optional, install third-party boosting baselines: `pip install -r requirements-optional.txt`
3. Place dataset files in `data/raw/`
4. Run the default binary experiment: `python -m src.ids_ml.pipeline_binary`
5. Run the default multiclass experiment: `python -m src.ids_ml.pipeline_multiclass`
6. Run the optional MLP binary baseline: `python -m src.ids_ml.pipeline_binary --include-mlp`
7. Run the optional MLP multiclass baseline: `python -m src.ids_ml.pipeline_multiclass --include-mlp`
8. Run the additional binary machine-learning comparison: `python -m src.ids_ml.additional_binary_experiments`
9. Run the additional comparison with optional LightGBM/CatBoost baselines: `python -m src.ids_ml.additional_binary_experiments --include-third-party`
10. Run optional LightGBM feature importance: `python -m src.ids_ml.lightgbm_feature_importance`
11. Run tests: `pytest -q`

## Output Files

- Binary metrics: `results/binary_metrics.csv`
- Multiclass metrics: `results/multiclass_metrics.csv`
- Additional binary ML comparison: `results/additional_ml_binary_experiments.csv`
- Binary models: `models/*_binary.joblib`
- Multiclass models: `models/*_multiclass.joblib`
- Figures: `results/figures/`

## Analysis Notes

- MLP versus XGBoost analysis: `results/mlp_vs_xgboost_analysis.md`
- Tabular deep learning versus tree models: `results/deep_tabular_vs_tree_analysis.md`
