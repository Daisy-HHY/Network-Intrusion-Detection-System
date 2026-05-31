# 主流机器学习方法扩展对比实验报告

## 1. 实验目的

本次扩展实验的目的是在当前项目已有模型基础上，继续尝试更多主流机器学习方法，判断是否存在相对于当前最优 `XGBoost` 更好的二分类模型。

原项目中，`XGBoost` 在 UNSW-NB15 二分类任务上取得了最高 F1 分数。因此，本实验将 `XGBoost` 作为参考基线，并在相同数据划分、相同预处理流程和相同评价指标下，引入更多传统机器学习和梯度提升类模型进行比较。

本文最终写作口径保持为：`XGBoost` 仍作为项目主模型，`LightGBM` 作为扩展实验中的补充对照结果。这样可以避免将新增可选依赖模型与原项目默认模型集合混为一谈。

## 2. 实验设置

本实验保持与项目已有二分类实验一致的基本设置：

- 数据集：`UNSW-NB15`
- 训练集：`data/raw/UNSW_NB15_training-set.csv`
- 测试集：`data/raw/UNSW_NB15_testing-set.csv`
- 任务类型：二分类入侵检测
- 目标标签：`label`
- 特征处理：
  - 数值特征使用 `StandardScaler`
  - 类别特征使用 `OneHotEncoder(handle_unknown="ignore")`
- 评价指标：
  - Accuracy
  - Precision
  - Recall
  - F1
  - ROC-AUC
- 结果文件：`results/additional_ml_binary_experiments.csv`

其中，`xgboost_reference` 表示在同一扩展实验脚本中重新运行的 XGBoost 参考结果，用于和新增模型进行直接比较。

## 3. 对比模型

本次扩展实验尝试了以下主流方法：

| 模型名称 | 说明 |
| --- | --- |
| `xgboost_reference` | 当前项目原有最优模型的同口径参考结果 |
| `lightgbm` | LightGBM 梯度提升树模型，属于可选第三方依赖 |
| `catboost` | CatBoost 梯度提升树模型，属于可选第三方依赖 |
| `hist_gradient_boosting` | scikit-learn 的直方图梯度提升树 |
| `gradient_boosting` | scikit-learn 传统 GBDT |
| `ada_boost_tree` | 基于浅层决策树的 AdaBoost |
| `extra_trees` | 极端随机树 |
| `extra_trees_balanced` | 带类别权重平衡的极端随机树 |
| `linear_svm_sgd` | 基于 SGD 的线性分类器 |
| `gaussian_nb` | 高斯朴素贝叶斯 |

## 4. 实验结果

按照 F1 分数从高到低排序，实验结果如下：

| 模型 | 训练耗时(s) | Accuracy | Precision | Recall | F1 | ROC-AUC |
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

## 5. 结果分析

从 F1 指标看，`LightGBM` 是本次扩展实验中表现最好的模型：

- `LightGBM` F1：`0.8975187698237443`
- `XGBoost reference` F1：`0.8969408551951247`
- F1 提升幅度：约 `+0.00058`

从 ROC-AUC 指标看，`LightGBM` 同样高于 XGBoost：

- `LightGBM` ROC-AUC：`0.9854891369022777`
- `XGBoost reference` ROC-AUC：`0.9842207208200878`

这说明，在加入可选第三方梯度提升模型后，`LightGBM` 在当前 UNSW-NB15 二分类设置下取得了略优于 `XGBoost` 的结果。

不过，这种优势幅度很小。`LightGBM` 的 F1 只比 `XGBoost` 高约 `0.00058`，因此不应在论文中表述为“显著提升”或“大幅超过”。更稳妥的说法是：`LightGBM` 在本次扩展实验中以微弱优势取得最优结果，说明它可以作为 XGBoost 的有力补充对照模型，但本文主模型仍保持为原项目默认模型集合中的 `XGBoost`。

## 6. 与其他模型的比较

`HistGradientBoosting` 的表现也较强，F1 为 `0.8956`，但仍低于 `XGBoost` 和 `LightGBM`。它的 ROC-AUC 为 `0.9851`，接近 LightGBM，但 F1 不占优。

`CatBoost` 的 F1 为 `0.8947`，低于 XGBoost。虽然 CatBoost 是主流梯度提升模型，但在本项目当前预处理和参数设置下没有超过 XGBoost。

`ExtraTrees` 和 `ExtraTrees balanced` 的 F1 分别为 `0.8880` 和 `0.8888`，低于梯度提升类模型，而且训练耗时明显更长。

`GradientBoosting` 和 `AdaBoost` 属于经典 boosting 方法，但效果低于 XGBoost、LightGBM、CatBoost 和 HistGradientBoosting。

`linear_svm_sgd` 和 `gaussian_nb` 与树模型差距较大，不适合作为本项目主模型。

## 7. LightGBM 特征重要性

为了补充扩展实验的可解释性分析，项目进一步导出了 LightGBM 的特征重要性结果：

- 结果文件：`results/lightgbm_feature_importance.csv`
- 图像文件：`results/figures/lightgbm_feature_importance.png`

LightGBM 的前 15 个重要特征如下：

| 排名 | 特征 | Importance |
| ---: | --- | ---: |
| 1 | `smean` | 898 |
| 2 | `sbytes` | 749 |
| 3 | `ct_srv_src` | 705 |
| 4 | `ct_srv_dst` | 456 |
| 5 | `ct_dst_src_ltm` | 424 |
| 6 | `dbytes` | 339 |
| 7 | `sload` | 265 |
| 8 | `dmean` | 261 |
| 9 | `sinpkt` | 247 |
| 10 | `tcprtt` | 244 |
| 11 | `ct_src_ltm` | 234 |
| 12 | `synack` | 231 |
| 13 | `dload` | 221 |
| 14 | `stcpb` | 214 |
| 15 | `ct_dst_ltm` | 206 |

与 XGBoost 的重要特征相比，LightGBM 更强调 `smean`、`sbytes`、`ct_srv_src`、`ct_srv_dst` 等流量统计与连接计数特征。这说明不同梯度提升实现虽然整体性能接近，但在特征利用偏好上存在差异。论文中可以将其作为扩展实验的补充分析，而不应替代 XGBoost 的主模型解释。

## 8. 论文可用结论

论文建议采用以下表述：

> 为进一步检验 XGBoost 作为主模型的合理性，本文在原有模型基础上扩展比较了 Extra Trees、AdaBoost、Gradient Boosting、HistGradientBoosting、LightGBM、CatBoost、线性 SGD 分类器和高斯朴素贝叶斯等方法。实验结果表明，LightGBM 在当前 UNSW-NB15 二分类任务中取得最高 F1 分数 0.8975，略高于 XGBoost reference 的 0.8969，同时 ROC-AUC 也由 0.9842 提升至 0.9855。然而，该提升幅度仅约为 0.00058，差异较小，且 LightGBM 属于额外引入的可选依赖。因此，本文仍将 XGBoost 作为项目主模型，将 LightGBM 作为扩展实验中表现最好的补充对照模型。

也可以在结论部分补充：

> 在原项目默认模型集合中，XGBoost 仍是表现最好的主模型；在引入额外第三方梯度提升库后，LightGBM 取得了略优于 XGBoost 的结果，但提升幅度有限，因此更适合作为扩展实验结果而非替代主模型。

## 9. 注意事项

论文中需要避免以下不准确表述：

- 不能写“LightGBM 大幅超过 XGBoost”，因为 F1 只提升约 `0.00058`。
- 不能写“所有新增方法都优于 XGBoost”，因为除 LightGBM 外，其他新增方法均未超过 XGBoost 的 F1。
- 不能把 `LightGBM` 写成论文主模型，因为当前确定的写作口径是主模型仍为 `XGBoost`。
- 不能把 `LightGBM` 写成原项目默认依赖模型，因为它需要 `requirements-optional.txt` 中的可选依赖。
- 不能声称本实验做了大规模超参数搜索。本次实验采用的是固定参数下的扩展模型比较。

## 10. 最终结论

本次扩展实验找到了一个相对于当前 XGBoost 略优的主流方法：`LightGBM`。

但考虑到提升幅度很小，并且 `LightGBM` 属于额外可选依赖，论文中仍应将 `XGBoost` 作为项目主模型，将 `LightGBM` 作为扩展实验中的最佳补充模型进行讨论。
