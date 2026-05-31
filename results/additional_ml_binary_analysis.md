# Additional Binary Machine-Learning Comparison

## Experiment Scope

This experiment keeps the same binary UNSW-NB15 setup as the existing project:

- Training data: `data/raw/UNSW_NB15_training-set.csv`
- Test data: `data/raw/UNSW_NB15_testing-set.csv`
- Target: `label`
- Preprocessing: the existing `StandardScaler` + `OneHotEncoder` pipeline
- Metrics: accuracy, precision, recall, F1, and ROC-AUC
- Output table: `results/additional_ml_binary_experiments.csv`

The `xgboost_reference` row is included as the direct reference point for the current best model.

## Results

| Model | Train(s) | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 4.60 | 0.8760 | 0.8234 | 0.9863 | 0.8975 | 0.9855 |
| XGBoost reference | 7.77 | 0.8753 | 0.8231 | 0.9854 | 0.8969 | 0.9842 |
| HistGradientBoosting | 14.48 | 0.8735 | 0.8210 | 0.9851 | 0.8956 | 0.9851 |
| CatBoost | 17.18 | 0.8723 | 0.8190 | 0.9859 | 0.8947 | 0.9829 |
| GradientBoosting | 224.38 | 0.8647 | 0.8096 | 0.9861 | 0.8892 | 0.9795 |
| ExtraTrees balanced | 1626.40 | 0.8644 | 0.8104 | 0.9839 | 0.8888 | 0.9724 |
| ExtraTrees | 1416.10 | 0.8635 | 0.8096 | 0.9834 | 0.8880 | 0.9730 |
| AdaBoost tree | 238.74 | 0.8517 | 0.7991 | 0.9761 | 0.8787 | 0.9737 |
| Linear SVM SGD | 2.97 | 0.8136 | 0.7527 | 0.9852 | 0.8534 | 0.9438 |
| GaussianNB | 0.78 | 0.5562 | 0.9998 | 0.1941 | 0.3251 | 0.7012 |

## Conclusion

Among the additional methods tested, `LightGBM` achieved the best binary F1 score:

- LightGBM F1: `0.8975187698237443`
- XGBoost reference F1: `0.8969408551951247`
- Difference: about `+0.00058`

LightGBM also produced a higher ROC-AUC than the XGBoost reference:

- LightGBM ROC-AUC: `0.9854891369022777`
- XGBoost reference ROC-AUC: `0.9842207208200878`

Therefore, LightGBM can be reported as the best model found in this expanded binary comparison. The improvement over XGBoost is small, so the paper should describe it as a marginal improvement rather than a large performance gain.

For the final paper, the main model remains XGBoost because it is the strongest model in the original project dependency set. LightGBM should be presented as the best optional third-party baseline in the expanded comparison, not as a replacement for the main model.

## LightGBM Feature Importance

The LightGBM feature-importance outputs are:

- `results/lightgbm_feature_importance.csv`
- `results/figures/lightgbm_feature_importance.png`

Top features:

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `smean` | 898 |
| 2 | `sbytes` | 749 |
| 3 | `ct_srv_src` | 705 |
| 4 | `ct_srv_dst` | 456 |
| 5 | `ct_dst_src_ltm` | 424 |
