# Experiment Summary

- Best binary model: `xgboost`
- Best binary F1: `0.896843416584621`
- Best expanded binary model with optional third-party baselines: `lightgbm`
- Best expanded binary F1: `0.8975187698237443`
- Best multiclass model: `xgboost`
- Best multiclass macro F1: `0.5064682212447587`
- Most important features: `sttl`, `proto_tcp`, `ct_srv_dst`, `proto_arp`, `ct_dst_sport_ltm`
- Main error pattern from confusion matrix: the largest error is false positives, with `9,571` normal samples misclassified as attacks by the best binary `xgboost` model.
