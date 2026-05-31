# MLP 与 XGBoost 效果差异分析

## 1. 当前实验结论

基于当前 `UNSW-NB15` 二分类实验结果，`XGBoost` 仍然是整体表现最好的模型，`MLP` 已经能够跑通，但效果略低于 `XGBoost`。

当前结果来自 [`results/binary_metrics.csv`](E:\Github_project\Network-Intrusion-Detection-System\results\binary_metrics.csv)：

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.8753 | 0.8234 | 0.9846 | 0.8968 | 0.9843 |
| MLP | 0.8737 | 0.8300 | 0.9691 | 0.8942 | 0.9749 |
| Random Forest | 0.8708 | 0.8172 | 0.9859 | 0.8936 | 0.9789 |

需要注意的是，`MLP` 与 `XGBoost` 的差距并不大，`F1` 只低了约 `0.0027`，但在当前实验设置下，`XGBoost` 仍然稳定领先。

## 2. 为什么神经网络没有超过 XGBoost

### 2.1 任务本身是典型的表格数据分类问题

`UNSW-NB15` 当前输入主要由以下两类特征组成：

- 数值统计特征：如 `sttl`、`dload`、`rate`、`ct_state_ttl`
- 低基数类别特征：如 `proto`、`service`、`state`

这类数据更接近经典的结构化表格数据，而不是图像、文本或语音。对这种任务，树模型，尤其是 `XGBoost`，通常比浅层全连接神经网络更有优势，因为：

- 树模型天然擅长学习特征阈值切分
- 树模型对不同尺度、非线性关系和特征交互更鲁棒
- 树模型对表格数据通常不需要大量特征工程就能取得较强效果

### 2.2 MLP 需要把类别特征展开成稠密输入，学习难度更高

当前预处理会把类别特征做 `OneHotEncoder`，再把输入转换为适配 `MLP` 的稠密矩阵。当前训练集经过预处理后，输入维度为：

- `175341 x 194`

这意味着：

- `MLP` 不是直接处理原始类别语义，而是在学习 one-hot 展开后的稠密特征组合
- 对树模型而言，类别拆分后的稀疏模式比较容易被局部决策规则捕获
- 对 `MLP` 而言，需要通过参数优化去逼近这些“阈值 + 组合”关系，优化难度更高

### 2.3 当前 MLP 优化目标和评估目标并不完全一致

当前二分类标签分布并不均衡，训练集比例约为：

- `label=0`: `31.94%`
- `label=1`: `68.06%`

而 `MLPClassifier` 默认优化的是基于概率的损失，不是直接优化 `F1`。这会带来两个问题：

- 模型更容易取得较高召回率
- 但精确率不一定同步提升

这点在当前结果中已经能看到：

- `MLP recall = 0.9691`
- `MLP precision = 0.8300`

也就是说，`MLP` 更倾向于把更多样本判成攻击，从而提高召回，但精确率受到一定影响。

### 2.4 当前 MLP 配置本身比较保守

当前实际使用的 `MLP` 基线配置见 [`src/ids_ml/train.py`](E:\Github_project\Network-Intrusion-Detection-System\src\ids_ml\train.py)：

```python
MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    learning_rate_init=0.001,
    max_iter=50,
    early_stopping=True,
    n_iter_no_change=5,
    random_state=42,
)
```

从已训练模型读取到的关键信息是：

- 实际训练轮数：`37`
- 最佳验证分数：`0.9501`
- 最终 loss：`0.0993`

这说明当前 `MLP` 在比较早的时候就触发了早停。它未必完全欠拟合，但至少可以说明：

- 当前训练策略偏保守
- 训练轮数和超参数搜索范围都比较有限

### 2.5 实验结果表明：问题不只在“训练不够久”

我后面又做了两组改进实验，尝试通过更长训练和更宽网络来提升效果，结果都没有超过基线 `MLP`，更没有超过 `XGBoost`。这说明：

- `MLP` 表现稍差，并不只是因为 `max_iter=50`
- 更核心的原因仍然是：**当前任务更适合梯度提升树，而不是简单的 sklearn MLP**

## 3. 我已经尝试过的改进过程

为了验证“是不是神经网络训练不够充分导致效果偏低”，我新增了可复现实验脚本：

- [`src/ids_ml/mlp_binary_experiments.py`](E:\Github_project\Network-Intrusion-Detection-System\src\ids_ml\mlp_binary_experiments.py)

实验结果保存在：

- [`results/mlp_improvement_experiments.csv`](E:\Github_project\Network-Intrusion-Detection-System\results\mlp_improvement_experiments.csv)

### 3.1 基线 MLP

参数：

- `hidden_layer_sizes=(128, 64)`
- `learning_rate_init=0.001`
- `max_iter=50`
- `early_stopping=True`
- `n_iter_no_change=5`

结果：

- `Accuracy = 0.8737`
- `Precision = 0.8300`
- `Recall = 0.9691`
- `F1 = 0.8942`
- `ROC-AUC = 0.9749`
- `训练耗时 = 326.44s`
- `n_iter = 37`

### 3.2 改进尝试一：更长训练、更小学习率、更大 batch

参数：

- `hidden_layer_sizes=(128, 64)`
- `learning_rate_init=0.0005`
- `alpha=0.0005`
- `batch_size=512`
- `max_iter=120`
- `early_stopping=True`
- `n_iter_no_change=10`

结果：

- `Accuracy = 0.8616`
- `Precision = 0.8129`
- `Recall = 0.9724`
- `F1 = 0.8855`
- `ROC-AUC = 0.9758`
- `训练耗时 = 179.43s`
- `n_iter = 55`

结论：

- 更长训练并没有提高 `F1`
- 虽然 `Recall` 略升，但 `Precision` 明显下降
- 说明单纯拉长训练并不能解决核心问题

### 3.3 改进尝试二：更宽网络 + 更强正则化

参数：

- `hidden_layer_sizes=(256, 128)`
- `learning_rate_init=0.0005`
- `alpha=0.001`
- `batch_size=512`
- `max_iter=120`
- `early_stopping=True`
- `n_iter_no_change=10`

结果：

- `Accuracy = 0.8641`
- `Precision = 0.8126`
- `Recall = 0.9789`
- `F1 = 0.8881`
- `ROC-AUC = 0.9773`
- `训练耗时 = 222.73s`
- `n_iter = 46`

结论：

- 更宽网络提高了 `Recall`
- 但 `Precision` 进一步下降
- `F1` 仍然低于基线 `MLP`

## 4. 改进实验总结

三组 `MLP` 实验的核心对比如下：

| Experiment | Train(s) | n_iter | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp_baseline` | 326.44 | 37 | 0.8737 | 0.8300 | 0.9691 | 0.8942 | 0.9749 |
| `mlp_tuned_longer` | 179.43 | 55 | 0.8616 | 0.8129 | 0.9724 | 0.8855 | 0.9758 |
| `mlp_tuned_wider` | 222.73 | 46 | 0.8641 | 0.8126 | 0.9789 | 0.8881 | 0.9773 |

实验结论可以直接写进论文：

1. `MLP` 已经具备较强分类能力，但在当前 `UNSW-NB15` 二分类任务上仍略低于 `XGBoost`
2. 通过“更长训练”和“更宽网络”并没有进一步提升 `F1`
3. 神经网络的主要收益体现在高召回率，但精确率不如 `XGBoost` 稳定
4. 对当前结构化表格数据任务，`XGBoost` 仍然是更适合的主模型

## 5. 论文中可以使用的解释表述

可以直接采用下面这个逻辑：

> 虽然多层感知机（MLP）能够学习非线性关系，但在 UNSW-NB15 这类结构化表格数据上，XGBoost 对特征阈值切分与局部交互关系的建模更具优势。实验中，MLP 的召回率较高，但精确率略低，导致其 F1 分数仍略低于 XGBoost。进一步的超参数调优表明，增加训练轮数和扩大网络宽度并未带来更优的 F1，说明性能差距并非仅由训练不足引起，而与模型对表格数据的适配性密切相关。

## 6. 如果继续提升神经网络，下一步应该怎么改

如果后续还要继续优化神经网络，而不是只停留在 `sklearn` 的 `MLPClassifier`，更有价值的方向是：

1. **做阈值调优，而不是固定 0.5**
   - 当前 `MLP` 的 `Recall` 很高，说明概率输出有潜力
   - 可以在验证集上搜索最优阈值，直接以 `F1` 为目标调节预测阈值

2. **使用支持类别嵌入的神经网络**
   - 当前 `proto/service/state` 仍然通过 one-hot 编码输入
   - 如果改用 `PyTorch`，可对类别特征使用 embedding，通常会比 one-hot + MLP 更自然

3. **使用更适合表格数据的深度模型**
   - 如 `TabNet`、`FT-Transformer`、`TabTransformer`
   - 它们通常比 `sklearn` MLP 更适合结构化数据

4. **按 F1 目标重新设计训练**
   - 当前优化目标和最终评价指标并不完全一致
   - 可以考虑带权损失、焦点损失或正负样本重加权

5. **系统化超参数搜索**
   - 当前只尝试了少量人工调参
   - 后续可对隐藏层规模、学习率、`alpha`、`batch_size`、早停耐心值做网格搜索或贝叶斯优化

## 7. 最终建议

就当前课程项目的结果和时间成本而言：

- **主模型建议使用 `XGBoost`**
- **MLP 适合作为“神经网络基线”保留在论文中**

这样写最合理：

- `XGBoost` 作为最终推荐模型
- `MLP` 作为对照实验，说明“神经网络并不一定在结构化入侵检测数据上优于树模型”
