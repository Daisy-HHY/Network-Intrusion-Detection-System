# 面向表格数据的深度学习架构与树模型对比分析

## 1. 分析目标

本节关注的问题不是“普通神经网络能否工作”，而是：

1. 对于 `UNSW-NB15` 这种结构化表格数据，是否可以通过**更适合表格数据的深度学习架构**取得比树模型更好的效果；
2. 如果不能，原因是什么；
3. 如果能接近甚至略有提升，这种提升是否值得额外的训练成本与实现复杂度。

## 2. 为什么不能只看普通 MLP

此前已经验证，普通 `MLP` 在当前二分类任务上能够取得较强结果，但略低于 `XGBoost`。  
相关结果见：

- [`results/binary_metrics.csv`](E:\Github_project\Network-Intrusion-Detection-System\results\binary_metrics.csv)
- [`results/mlp_vs_xgboost_analysis.md`](E:\Github_project\Network-Intrusion-Detection-System\results\mlp_vs_xgboost_analysis.md)

但普通 `MLP` 并不是专门为表格数据设计的。它的局限主要在于：

- 需要依赖 `one-hot` 展开类别特征；
- 不擅长直接学习树模型那种阈值切分模式；
- 缺乏针对表格数据常见的“数值特征 + 少量类别特征 + 稀疏交互”的结构偏置。

因此，如果要认真比较深度学习与树模型，应该至少尝试一类**面向表格数据的深度学习架构**。

## 3. 文献与架构选择依据

在表格深度学习方向里，常见代表方法包括：

1. **TabNet**：通过 sequential attention 做特征选择  
   论文链接：[TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)

2. **TabTransformer**：对类别特征做 contextual embedding，再与数值特征融合  
   论文链接：[TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678)

3. **FT-Transformer / ResNet-like tabular models**：将数值特征和类别嵌入统一映射到深层网络，文献中指出它们是强有力的深度学习表格基线  
   论文链接：[Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)

结合当前数据集的实际结构：

- 类别列只有 `3` 个：
  - `proto`: `133` 类
  - `service`: `13` 类
  - `state`: `9` 类
- 其余大部分都是连续统计特征

这意味着：

- `TabTransformer` 这类**主要强化类别特征上下文建模**的方法，在当前数据上发挥空间有限，因为类别列数量并不多；
- `TabNet` 很有解释性，但工程实现和调参成本更高，而且在文献里也并不是稳定压过 GBDT 的统一最优方法；
- 对当前数据最现实的深度学习方向，是采用 **类别嵌入 + 数值特征标准化 + 残差块** 的 **ResNet-like tabular network**。

因此，本项目选择实现一个**面向表格数据的 ResNet-like 二分类网络**，而不是继续只调普通 MLP。

## 4. 当前数据结构对深度学习的启示

当前训练数据的标签分布：

- `label=0`: `31.94%`
- `label=1`: `68.06%`

类别特征基数：

- `proto`: `133`
- `service`: `13`
- `state`: `9`

这带来两个重要结论：

1. 当前数据属于**以数值统计特征为主、少量中低基数类别特征为辅**的典型表格数据；
2. 如果用深度学习，最值得尝试的不是 plain MLP，而是：
   - 对类别特征做 embedding
   - 对数值特征做标准化
   - 通过 residual block 学习更深层的交互关系

## 5. 已实现的深度学习改进方案

我新增了以下实验脚本：

- 表格深度学习实验：
  [`src/ids_ml/tabular_dl_binary_experiments.py`](E:\Github_project\Network-Intrusion-Detection-System\src\ids_ml\tabular_dl_binary_experiments.py)

- 同口径参考基线实验：
  [`src/ids_ml/subset_reference_binary_experiments.py`](E:\Github_project\Network-Intrusion-Detection-System\src\ids_ml\subset_reference_binary_experiments.py)

对应结果文件：

- 深度学习实验结果：
  [`results/tabular_dl_binary_experiments.csv`](E:\Github_project\Network-Intrusion-Detection-System\results\tabular_dl_binary_experiments.csv)

- 同口径参考基线：
  [`results/subset_reference_binary_experiments.csv`](E:\Github_project\Network-Intrusion-Detection-System\results\subset_reference_binary_experiments.csv)

### 5.1 为什么要做“同口径参考基线”

当前正式主结果中的 `XGBoost` 和 `MLP` 来自完整官方训练集。  
而深度学习实验在本地 CPU/内存环境中，整份 `175341` 条训练集读入和训练稳定性较差。

因此我采用了一个**近全量子集**方案：

- 训练集前 `160000` 条样本作为实验母集
- 其中再按固定随机种子划分为：
  - 训练子集：约 `144131`
  - 验证子集：约 `15869`
- 测试集仍使用**完整官方测试集** `82332`

为了保证对比尽可能公平，我不仅跑了深度学习模型，也在**同一子集、同一拆分口径**下重新跑了：

- `XGBoost`
- `MLP`

这样可以避免出现“全量树模型 vs 子集深度学习”的不公平对比。

## 6. 实验结果

### 6.1 同口径参考基线（160k 子集 -> 144k train / 16k val -> full test）

来自 [`results/subset_reference_binary_experiments.csv`](E:\Github_project\Network-Intrusion-Detection-System\results\subset_reference_binary_experiments.csv)：

| Experiment | Train(s) | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `xgboost_subset_144k` | 7.12 | 0.8795 | 0.8294 | 0.9833 | 0.8998 | 0.9845 |
| `mlp_subset_144k` | 69.33 | 0.8688 | 0.8229 | 0.9707 | 0.8907 | 0.9758 |

### 6.2 面向表格数据的深度学习实验

来自 [`results/tabular_dl_binary_experiments.csv`](E:\Github_project\Network-Intrusion-Detection-System\results\tabular_dl_binary_experiments.csv)：

| Experiment | Train(s) | Best Epoch | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tabular_resnet_default` | 361.61 | 20 | 0.50 | 0.8781 | 0.8355 | 0.9694 | 0.8975 | 0.9783 |
| `tabular_resnet_threshold_tuned` | 403.63 | 22 | 0.40 | 0.8481 | 0.7894 | 0.9876 | 0.8775 | 0.9769 |

## 7. 结果解读

### 7.1 相比普通 MLP，表格深度学习架构确实有改进

在相同 144k 训练子集上：

- `mlp_subset_144k`: `F1 = 0.8907`
- `tabular_resnet_default`: `F1 = 0.8975`

提升约为：

- `+0.0068` 的 F1

这说明：

- **不是所有深度学习都不适合这个任务**
- 与普通 `MLP` 相比，**带类别嵌入和残差结构的表格深度学习模型确实更适合 `UNSW-NB15`**

也就是说，“普通 MLP 不够强”并不等于“深度学习在这个任务上完全没希望”。

### 7.2 但它仍然没有超过同口径 XGBoost

在公平的子集对比中：

- `xgboost_subset_144k`: `F1 = 0.8998`
- `tabular_resnet_default`: `F1 = 0.8975`

差距约为：

- `-0.0024` 的 F1

同时在 `ROC-AUC` 上：

- `xgboost_subset_144k`: `0.9845`
- `tabular_resnet_default`: `0.9783`

差距更明显。

这说明：

- 即使换成更适合表格数据的深度学习架构，当前任务上**树模型仍然略优**
- 深度学习模型已经很接近，但还没有真正反超

### 7.3 阈值调优并没有带来更好的综合指标

`tabular_resnet_threshold_tuned` 通过验证集选择了更低阈值 `0.40`，结果表现为：

- `Recall` 从 `0.9694` 升到 `0.9876`
- 但 `Precision` 从 `0.8355` 降到 `0.7894`
- 最终 `F1` 从 `0.8975` 降到 `0.8775`

这说明：

- 当前深度学习模型的输出概率分布已经偏向高召回
- 再进一步压低阈值只会把更多正常流量错判为攻击
- 在当前任务上，**阈值调优并不是主要增益来源**

### 7.4 训练成本明显更高

训练时间对比：

- `xgboost_subset_144k`: `7.12s`
- `mlp_subset_144k`: `69.33s`
- `tabular_resnet_default`: `361.61s`
- `tabular_resnet_threshold_tuned`: `403.63s`

这意味着：

- `tabular_resnet_default` 的训练成本约为 `XGBoost` 的 `50.8x`
- 但性能并没有超过 `XGBoost`

这点对论文很重要，因为它直接支撑以下结论：

> 即使表格深度学习架构能在一定程度上改善普通 MLP 的表现，但在当前 UNSW-NB15 二分类任务上，其性能增益不足以抵消显著增加的训练成本与实现复杂度。

## 8. 为什么表格深度学习仍然没有反超树模型

结合实验结果，可以把原因分成四类：

### 8.1 数据模式仍然更适合树模型

当前特征里有大量典型的阈值型统计量，例如：

- `sttl`
- `dload`
- `rate`
- `ct_state_ttl`

这些特征非常适合通过树模型做逐层切分。  
对 `XGBoost` 而言，这些切分几乎是天然表达方式；而深度学习模型需要通过连续参数优化去“逼近”这些阈值边界。

### 8.2 类别特征虽然存在，但并不构成 Transformer 类模型的大优势场景

当前类别列只有 `3` 个：

- `proto`
- `service`
- `state`

其中只有 `proto` 的基数相对较高（`133`）。  
这意味着：

- embedding 确实有价值
- 但像 `TabTransformer` 那种依赖多类别字段上下文交互的模型，在这里未必能发挥明显优势

也就是说，本数据更像是：

- “数值统计特征主导”
- 而不是“高维类别上下文主导”

### 8.3 深度学习模型更依赖训练细节

当前 `tabular_resnet_default` 已经比普通 `MLP` 明显更好，但它仍然对以下因素更敏感：

- 学习率
- dropout
- 残差块深度
- batch size
- 早停策略
- 阈值选择

而 `XGBoost` 在当前任务上表现更稳定，调参弹性更大。

### 8.4 本地资源限制也约束了深度学习进一步搜索

当前环境是：

- `PyTorch 2.8.0+cpu`
- 无 GPU
- `pandas` 在整份训练集读入时存在较明显内存不稳定问题

因此本次深度学习实验虽然已经能形成有效结论，但仍然没有做到：

- 更大规模系统化超参数搜索
- 更深层模型堆叠
- 更复杂的 attention 型 tabular 模型训练

这意味着：

- 当前结论是“在现有资源与工程成本下，树模型仍更划算”
- 而不是“任何深度学习架构都不可能超过树模型”

## 9. 论文中可以使用的结论表述

可以直接使用下面这段：

> 为了进一步验证深度学习在结构化入侵检测任务中的潜力，本文在普通 MLP 之外，又实现了一个面向表格数据的 ResNet-like 深度学习模型。该模型采用类别特征嵌入、数值特征标准化以及残差块结构，以增强对表格特征交互关系的建模能力。实验表明，该模型在同一训练子集上的 F1 分数较普通 MLP 提升了约 0.0068，说明针对表格数据设计的深度学习结构确实优于普通全连接网络。然而，与同口径的 XGBoost 基线相比，该模型的 F1 仍低约 0.0024，且训练时间约为 XGBoost 的 8 倍以上。因此，在当前 UNSW-NB15 二分类任务中，表格深度学习模型虽然能够缩小与树模型之间的性能差距，但尚不足以在效果和成本的综合权衡上取代 XGBoost。

## 10. 后续如果继续追求“深度学习反超树模型”，最值得尝试的方向

如果后续还要继续冲击树模型，优先级最高的方向应当是：

1. **完整实现 FT-Transformer 或 TabTransformer**
   - 不是只做 ResNet-like baseline
   - 尤其是如果后续加入更多类别语义特征，这类模型潜力会更大

2. **使用 GPU 环境进行系统化超参数搜索**
   - 当前 CPU 环境下，搜索空间明显受限
   - 深度学习模型的性能很大程度依赖训练预算

3. **引入数值特征分桶 / learnable numerical embeddings**
   - 当前数值特征仍主要通过标准化后直接输入
   - 如果把连续特征映射成更适合神经网络处理的表示，可能进一步提升效果

4. **联合优化阈值与损失函数**
   - 当前模型主要还是 BCE 损失
   - 可继续尝试 focal loss、class-balanced loss、AUC-oriented loss

5. **严格统一训练数据规模做更完整对比**
   - 例如全量训练 `XGBoost`、`Tabular ResNet`、`FT-Transformer`
   - 在相同官方划分下给出最终结论

## 11. 当前最稳妥的结论

截至目前，可以得出一个比较稳健、也适合写进论文的结论：

- 普通 `MLP` 不是当前任务上最合适的深度学习基线
- 采用**面向表格数据的深度学习结构**后，效果确实优于普通 `MLP`
- 但在当前 `UNSW-NB15` 二分类实验中，**树模型 `XGBoost` 仍然略优**
- 更重要的是，树模型在训练成本和工程复杂度上显著更划算

因此：

- **最终推荐模型仍然是 `XGBoost`**
- **面向表格数据的深度学习模型应作为重要补充实验和分析对象写入论文**
