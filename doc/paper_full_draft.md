# 基于 UNSW-NB15 数据集的网络入侵检测系统研究

## 一、绪论

### 1.1 研究背景

随着网络应用规模扩大，网络边界、业务系统和终端设备产生的流量类型日益复杂。传统安全防护手段通常依赖规则、黑名单或人工配置策略，对已知攻击具有较强针对性，但面对攻击行为变体、异常连接模式和高维流量特征时，单纯依靠固定规则容易出现漏报或误报。入侵检测系统（Intrusion Detection System，IDS）通过采集主机、网络或应用层行为数据，对潜在攻击进行识别，是网络安全防护体系中的重要组成部分。

Denning 较早提出了入侵检测模型，认为系统行为可以通过审计记录、统计特征和异常模式进行刻画，并可据此发现偏离正常行为的入侵活动[1]。在网络入侵检测场景中，检测对象通常表现为网络连接或流量记录，特征包括协议类型、连接状态、字节数、包间隔、连接计数等。由于这些特征具有较强的结构化表格数据特征，机器学习方法能够在人工规则之外学习正常流量与攻击流量之间的统计差异。

本文研究基于 UNSW-NB15 数据集展开。该数据集由 Moustafa 和 Slay 构建，用于网络入侵检测系统研究，包含正常流量以及 Fuzzers、Analysis、Backdoor、DoS、Exploits、Generic、Reconnaissance、Shellcode、Worms 等多类攻击[2]。UNSW 官方资料说明，该数据集使用 Argus、Bro-IDS 等工具及特征生成算法形成 49 个特征及标签，并提供训练集与测试集划分，其中训练集包含 175341 条记录，测试集包含 82332 条记录[2-3]。本项目使用的 `UNSW_NB15_training-set.csv` 和 `UNSW_NB15_testing-set.csv` 与该官方划分一致，删除 `id` 字段后，以其余流量特征作为模型输入。

从网络安全实际需求看，入侵检测模型通常需要在漏报和误报之间进行权衡。漏报意味着攻击流量未被发现，可能导致安全风险继续扩大；误报则意味着正常流量被判定为攻击，虽然不会直接造成攻击遗漏，但会增加告警处理成本。本文实验结果也体现了这一点：默认二分类实验中 XGBoost 的 Recall 达到 0.9846，说明其能够识别绝大多数攻击样本；但系统汇总结果显示，该模型仍将 9571 条正常样本误判为攻击，误报问题仍然存在。因此，本文不只关注 Accuracy，而是同时报告 Precision、Recall、F1 和 ROC-AUC，以更完整地评价模型在入侵检测任务中的实际表现。

此外，网络入侵检测数据具有明显的表格数据属性。与图像或文本任务不同，UNSW-NB15 中的样本不是像素矩阵或自然语言序列，而是由多列统计特征、协议特征和连接状态特征组成的结构化记录。这类数据常见的建模路线包括线性模型、树模型、集成学习和表格深度学习。本文将这些方法放在同一实验框架下比较，能够更清楚地说明：在当前数据划分和参数设置下，哪些方法只是作为基线，哪些方法更适合作为主模型，哪些方法虽然有理论潜力但在当前实验中未表现出明显优势。

### 1.2 研究目的与意义

本文目标是在现有项目系统基础上，构建并分析一个基于机器学习的离线网络入侵检测实验流程。系统围绕 UNSW-NB15 数据集完成二分类入侵检测、多分类攻击类型识别、MLP 神经网络补充实验、表格深度学习补充实验以及主流机器学习扩展实验。研究重点不在于宣称某一模型在所有场景中最优，而在于基于统一数据划分、统一预处理流程和可复现实验结果，比较不同模型在当前任务中的实际表现。

本文的意义主要体现在三个方面。第一，从应用角度看，二分类模型可用于判断网络连接是否属于攻击流量，多分类模型则进一步识别具体攻击类别，为后续告警分级和安全分析提供参考。第二，从方法角度看，本文比较线性模型、决策树、随机森林、梯度提升树、XGBoost、LightGBM、CatBoost、MLP 与表格 ResNet-like 模型，有助于观察不同模型族在结构化入侵检测数据上的差异。第三，从工程角度看，项目保存了训练脚本、评估指标、模型结果和特征重要性文件，能够为后续改进模型、补充交叉验证、开展在线检测流程提供基础。

需要强调的是，当前项目是离线机器学习实验系统，并未实现实时流量采集、在线特征提取和生产环境部署。因此，本文仅讨论基于离线数据集的模型训练与测试结果，不将其描述为已经上线运行的实时网络入侵检测系统。

围绕上述目标，本文需要回答以下几个具体问题。第一，在官方训练集和测试集划分下，传统机器学习模型能否对攻击流量取得较高识别能力。第二，二分类和多分类任务之间的性能差异有多大，这种差异是否与类别数量和类别不均衡有关。第三，普通 MLP 与面向表格数据的深度学习结构是否能够超过树模型。第四，当引入 LightGBM、CatBoost、HistGradientBoosting 等主流扩展模型后，是否能够明显改变默认实验中 XGBoost 作为主模型的结论。

本文的写作口径严格限定在项目已有实验结果和可核验文献范围内。对于项目没有保存的内容，例如真实在线部署吞吐量、实时采集模块性能、硬件环境详细配置、多分类逐类别混淆矩阵和大规模超参数搜索结果，本文均不作肯定性描述。对于无法由实验直接证明的内容，本文只在理论或文献综述部分引用已有研究，不将其作为本系统实验结论。

### 1.3 研究现状

入侵检测研究大体可分为基于特征规则的检测、基于异常行为的检测以及基于机器学习的检测。基于规则的方法依赖专家经验，适合识别已知攻击，但对未知攻击和变种攻击的适应性有限。基于异常的方法通过建立正常行为模型，识别偏离正常模式的样本，但在真实网络环境中容易受业务波动影响产生误报。机器学习方法则通过标注数据学习判别边界，能够在高维特征中发现复杂的非线性关系。

在表格型安全数据上，树模型和集成学习方法被广泛采用。随机森林通过构建多棵决策树并进行集成，降低单棵树的方差[4]；梯度提升机通过逐步拟合残差提升模型表达能力[5]；XGBoost 在梯度提升框架中引入正则化、列采样、并行优化等机制，提升了树提升模型的效率和泛化能力[6]；LightGBM 进一步通过基于直方图的算法和叶子优先生长策略提高训练效率[7]；CatBoost 则针对类别特征处理和有序提升提出改进[8]。

近年来，深度学习也被用于表格数据建模。Gorishniy 等对表格深度学习模型进行了系统比较，并指出 ResNet-like 与 FT-Transformer 等结构可以作为较强的深度表格模型基线，但深度模型与梯度提升树之间并不存在普遍意义上的绝对优势[9]。这与本文实验现象基本一致：在当前 UNSW-NB15 二分类任务中，普通 MLP 未超过 XGBoost；表格 ResNet-like 模型相较普通 MLP 有改进，但在同口径子集实验中仍略低于 XGBoost，且训练耗时更长。

从已有研究和本文实验设置可以看出，网络入侵检测模型比较应避免只报告单一指标。一方面，入侵检测数据往往存在类别不均衡，Accuracy 容易被多数类影响；另一方面，安全场景中误报和漏报的代价不同，Precision 与 Recall 需要同时观察。本文采用 F1 作为主要排序指标，是因为 F1 同时考虑 Precision 和 Recall，在二分类攻击识别任务中比单独 Accuracy 更能反映模型综合检测效果。对于多分类任务，本文使用 macro 指标，是因为 macro 平均对每个类别赋予相同权重，更适合观察模型对少数类攻击的平均识别能力。

综上，本文研究现状部分可以归纳出三个判断。第一，传统规则方法仍具有实用价值，但在本文数据驱动实验中不作为主要研究对象。第二，树模型和梯度提升模型是结构化入侵检测数据上的重要基线，应作为本文重点比较对象。第三，深度学习方法在表格数据上具有研究价值，但其效果需要通过同口径实验验证，不能仅凭模型复杂度推断其一定优于树模型。

## 二、相关理论与技术

### 2.1 入侵检测基本原理

入侵检测的基本思想是从系统行为或网络流量中提取可观测特征，再根据规则、统计模型或机器学习模型判断样本是否异常。按照检测数据来源，入侵检测可分为主机入侵检测和网络入侵检测。本文研究对象属于网络入侵检测，即以网络连接记录为样本，对连接行为进行分类。

从检测方式看，入侵检测可分为误用检测和异常检测。误用检测根据已知攻击特征进行匹配，通常对已知攻击准确性较高，但依赖规则库更新。异常检测通过建立正常行为模型识别偏离样本，理论上更适合发现未知攻击，但容易受到正常业务变化影响。本文采用监督学习方式，使用 UNSW-NB15 中的 `label` 和 `attack_cat` 字段作为标注信息，分别开展二分类与多分类实验。

在二分类任务中，样本标签 `label=0` 表示正常流量，`label=1` 表示攻击流量；在多分类任务中，标签 `attack_cat` 表示具体类别，包括 Normal 以及九类攻击。二分类更关注是否发生攻击，多分类更关注攻击类型识别，因此多分类任务通常更难，尤其在攻击类别样本分布不均衡时，少数类的识别难度会明显上升。

监督学习型入侵检测流程通常包括四个环节。第一是数据采集和标注，即获得包含正常行为与攻击行为的样本集合。第二是特征处理，即将原始网络连接字段转换为模型可接受的数值表示。第三是模型训练，即使用训练集学习输入特征与目标标签之间的映射关系。第四是模型评估，即在独立测试集上观察模型对未参与训练样本的识别能力。本文项目严格使用 UNSW-NB15 官方训练集训练、官方测试集测试，没有将测试集样本加入训练过程。

在入侵检测任务中，混淆矩阵具有直接解释意义。对于二分类任务，可将攻击流量视为正类、正常流量视为负类。真正例（TP）表示攻击样本被正确判定为攻击；假正例（FP）表示正常样本被误判为攻击；真负例（TN）表示正常样本被正确判定为正常；假负例（FN）表示攻击样本被漏检。由此可知，FP 对应误报，FN 对应漏报。本文在实验分析中使用这些概念解释 XGBoost 的主要错误模式，即误报数量较多而漏报数量较少。

二分类与多分类在应用目标上也存在差异。二分类模型适合作为告警触发的前置模块，其输出可以回答“是否存在攻击”的问题；多分类模型则更接近告警解释模块，其输出可以提供攻击类型线索。但多分类模型的可靠性不仅取决于总体 Accuracy，还取决于各类攻击是否都能被有效识别。由于当前项目未保存每类详细报告，本文在多分类分析中只讨论整体 macro 指标和类别不均衡事实，不进一步推断具体类别的识别强弱。

### 2.2 数据集概述

本文使用 UNSW-NB15 数据集。根据 UNSW 官方说明，该数据集由澳大利亚网络安全中心的 Cyber Range Lab 生成，包含正常流量和多种现代攻击类型，并提供训练集和测试集划分[2]。本文项目中的原始数据文件为：

| 文件 | 样本数 | 字段数 |
| --- | ---: | ---: |
| `UNSW_NB15_training-set.csv` | 175341 | 45 |
| `UNSW_NB15_testing-set.csv` | 82332 | 45 |

本项目读取数据后删除 `id` 字段，因为该字段仅表示样本编号，不作为分类特征。二分类任务删除 `label` 和 `attack_cat` 两个目标相关字段后，将 `label` 作为目标变量；多分类任务同样删除输入中的目标相关字段，将 `attack_cat` 作为目标变量，并对缺失类别填充为 `Normal`。

训练集二分类标签分布如下：

| 标签 | 含义 | 样本数 |
| ---: | --- | ---: |
| 0 | Normal | 56000 |
| 1 | Attack | 119341 |

测试集二分类标签分布如下：

| 标签 | 含义 | 样本数 |
| ---: | --- | ---: |
| 0 | Normal | 37000 |
| 1 | Attack | 45332 |

多分类训练集类别分布如下：

| 类别 | 样本数 |
| --- | ---: |
| Normal | 56000 |
| Generic | 40000 |
| Exploits | 33393 |
| Fuzzers | 18184 |
| DoS | 12264 |
| Reconnaissance | 10491 |
| Analysis | 2000 |
| Backdoor | 1746 |
| Shellcode | 1133 |
| Worms | 130 |

多分类测试集类别分布如下：

| 类别 | 样本数 |
| --- | ---: |
| Normal | 37000 |
| Generic | 18871 |
| Exploits | 11132 |
| Fuzzers | 6062 |
| DoS | 4089 |
| Reconnaissance | 3496 |
| Analysis | 677 |
| Backdoor | 583 |
| Shellcode | 378 |
| Worms | 44 |

上述分布说明，UNSW-NB15 在当前官方划分下存在明显类别不均衡。二分类中攻击样本多于正常样本；多分类中 Normal、Generic、Exploits 样本较多，而 Worms、Shellcode、Backdoor 等类别样本较少。这一数据事实会影响模型评价，因此本文在多分类任务中重点报告 macro precision、macro recall 和 macro F1。

从比例上看，训练集中攻击样本为 119341 条，占训练集约 68.06%；正常样本为 56000 条，占约 31.94%。测试集中攻击样本为 45332 条，占测试集约 55.06%；正常样本为 37000 条，占约 44.94%。这说明训练集与测试集中的正负类比例并不完全相同，训练集中攻击样本占比更高。因此，模型如果过度倾向于预测攻击类别，可能在训练分布下表现较好，但在测试集上产生较多误报。本文二分类 XGBoost 的误报现象与这一风险具有一定关联，但由于本文没有进行进一步分布迁移实验，因此这里只将其作为结果解释，不作因果断言。

从多分类比例看，训练集中 Normal、Generic、Exploits 三类样本合计 129393 条，占训练集约 73.80%；而 Worms 只有 130 条，占训练集约 0.07%。测试集中 Worms 也只有 44 条。如此极端的少数类分布会使模型在训练阶段很难学习到稳定模式，也会使测试阶段单个样本预测错误对该类指标产生较大影响。因此，多分类任务的 Macro F1 低于二分类 F1 是符合数据特征的实验现象。

项目中特征类型包括数值特征和类别特征。类别特征主要为 `proto`、`service` 和 `state`。训练集中 `proto` 有 133 个取值，`service` 有 13 个取值，`state` 有 9 个取值；测试集中 `proto` 有 131 个取值，`service` 有 13 个取值，`state` 有 7 个取值。由于训练集和测试集类别取值集合可能并不完全一致，预处理阶段使用 `OneHotEncoder(handle_unknown="ignore")` 是必要的。该设置可以在测试集中出现训练阶段未见类别时避免编码报错，并将未知类别对应的已知独热列置为 0。

需要注意的是，UNSW 官方资料中的完整特征说明与本项目 CSV 文件中的字段数量存在表述角度差异。官方资料提到数据集包含 49 个特征及标签[2]，而本项目使用的训练集和测试集 CSV 各有 45 个字段。本文在描述项目实验时以本地 CSV 实际字段数量为准；在介绍数据集来源时引用官方说明，并明确本项目读取的是 `UNSW_NB15_training-set.csv` 和 `UNSW_NB15_testing-set.csv` 两个具体文件。

### 2.3 常用分类模型理论

本文默认实验包含 Logistic Regression、Decision Tree、Random Forest 和 XGBoost，并在扩展实验中加入 Extra Trees、AdaBoost、Gradient Boosting、HistGradientBoosting、LightGBM、CatBoost、线性 SGD 分类器和 GaussianNB。

Logistic Regression 是线性分类模型，通过对特征线性组合进行概率映射完成分类。该模型结构简单、训练效率较高，但对复杂非线性关系表达能力有限。

Decision Tree 通过特征划分构建树形分类规则，具有较好的可解释性，但单棵树容易受数据扰动影响。Random Forest 在多棵决策树基础上进行集成，通过样本和特征随机性降低方差[4]。

Gradient Boosting 通过序列化方式逐步训练弱学习器，使后续模型拟合前一阶段残差或负梯度，从而提升整体预测能力[5]。XGBoost 属于高效梯度提升树实现，在目标函数中加入正则项并支持列采样等机制[6]。LightGBM 通过直方图算法、叶子优先生长和特征并行等策略提升训练效率[7]。CatBoost 则重点处理类别特征和目标泄漏问题，提出有序提升等机制[8]。

本文选择 Logistic Regression 作为线性基线，是因为它能够反映在仅使用线性决策边界时模型可以达到的基本水平。如果线性模型与复杂模型差距较大，通常说明特征和标签之间存在较强非线性关系，或者类别之间存在复杂交互。本文实验中 Logistic Regression 的二分类 F1 为 0.8492，明显低于 XGBoost 的 0.8968，说明在当前特征空间中，非线性模型具有更强适应性。

Decision Tree 的优点是路径规则直观，每次预测都可以追溯到一系列特征划分；缺点是单棵树容易过拟合训练数据。Random Forest 通过训练多棵树并集成预测结果缓解单棵树不稳定问题。本文二分类中 Random Forest 的 F1 为 0.8936，高于单棵 Decision Tree 的 0.8853，符合集成方法通常比单模型更稳定的特点。但 Random Forest 仍低于 XGBoost，说明在当前任务中，逐步提升式集成比简单 bagging 集成更适合。

XGBoost、LightGBM、CatBoost 和 HistGradientBoosting 均属于梯度提升思想下的重要模型或实现。它们的共同特点是通过多轮弱学习器叠加逐步改善预测结果；差异主要体现在工程优化、树生长策略、类别特征处理和正则化设计上。本文扩展实验中 LightGBM、XGBoost reference、HistGradientBoosting 和 CatBoost 排名前四，F1 均达到 0.8947 以上，说明梯度提升类模型整体适合当前二分类入侵检测任务。

GaussianNB 属于朴素贝叶斯模型，通常假设特征在给定类别下条件独立，并使用概率模型完成分类。该假设在高维网络流量统计特征上可能较强。本文实验中 GaussianNB 的 Precision 接近 1，但 Recall 仅为 0.1941，说明其只将少量样本判定为攻击，虽然这些预测攻击的样本大多正确，但漏掉了大量真实攻击样本。因此，GaussianNB 不适合作为本文系统主模型。

### 2.4 面向表格数据的深度学习技术

表格数据通常由数值特征、类别特征和统计特征组成，与图像、语音、文本等数据不同，其特征之间缺少天然的空间或序列结构。因此，在许多表格任务中，梯度提升树仍是强有力的基线。Gorishniy 等指出，表格深度学习领域存在模型比较协议不统一的问题，并提出 ResNet-like 和 FT-Transformer 等结构作为更可靠的深度表格模型基线[9]。

本文系统中包含两类深度学习补充实验。第一类是基于 scikit-learn `MLPClassifier` 的普通 MLP，隐藏层结构为 `(128, 64)`，激活函数为 ReLU，初始学习率为 0.001，最大迭代次数为 50，并启用 early stopping。由于 MLP 需要稠密输入，项目在 MLP 实验中将预处理器输出设置为 dense 格式。

第二类是基于 PyTorch 的 Tabular ResNet-like 二分类模型。该模型对类别特征使用 embedding，对数值特征进行标准化，并通过残差块建模特征交互关系。由于训练成本较高，该补充实验使用训练集前 160000 条样本形成子集，并划分约 144000 条训练样本和约 16000 条验证样本，测试集仍使用完整官方测试集。为了公平比较，项目在同一子集口径下重新运行 XGBoost 和 MLP。

普通 MLP 将所有输入特征经过编码和标准化后视为一个稠密向量，再通过全连接层学习特征组合。其优点是结构通用，能够拟合非线性关系；缺点是对表格数据中的类别特征、稀疏独热特征和特征尺度较敏感，且通常需要较多调参。本文 MLP 改进实验尝试了更长训练和更宽网络，但 F1 均低于基础 MLP，说明简单增加模型容量并不一定带来更好效果。

Tabular ResNet-like 模型与普通 MLP 的重要区别在于，它对类别特征使用 embedding 表示，而不是直接使用独热编码后的稀疏向量；同时通过残差块缓解深层网络训练难度。项目脚本中类别特征 embedding 维度由类别基数计算，并限制在合理范围内；数值特征使用 `StandardScaler` 标准化后输入模型；训练过程中以验证集 AUC 选择最佳 epoch，并在阈值调优版本中使用验证集搜索 F1 最优阈值。这些设计比普通 MLP 更贴近表格深度学习方法。

但是，深度学习模型的训练结果受数据规模、网络结构、优化器、学习率、正则化和阈值选择影响较大。本文只运行了两组 Tabular ResNet-like 配置，不能据此全面评价所有深度表格模型。本文能够得出的结论仅限于当前实验：Tabular ResNet default 在同口径子集上优于 MLP，但未超过 XGBoost。

### 2.5 模型评价指标

本文二分类实验采用 Accuracy、Precision、Recall、F1 和 ROC-AUC 作为评价指标。Accuracy 表示整体分类正确比例；Precision 表示被预测为攻击的样本中真实攻击的比例；Recall 表示真实攻击样本中被正确检测出的比例；F1 是 Precision 与 Recall 的调和平均，适合在类别不均衡任务中综合考察误报与漏报；ROC-AUC 衡量模型在不同阈值下区分正负样本的能力，常用于二分类模型排序能力评价[10]。

多分类任务中，本文使用 Accuracy、macro precision、macro recall 和 macro F1。macro 指标先分别计算每个类别的指标，再对类别取平均，因此不会让样本量大的类别完全主导结果。考虑到 UNSW-NB15 多分类类别分布不均衡，macro F1 比单纯 Accuracy 更能反映模型对少数类攻击的平均识别能力。

二分类指标可由混淆矩阵计算得到。设攻击类为正类，则：

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 × Precision × Recall / (Precision + Recall)

在入侵检测语境下，Precision 较低通常意味着误报较多，Recall 较低则意味着漏报较多。本文默认二分类 XGBoost 的 Precision 为 0.8234，Recall 为 0.9846，说明其更倾向于提高攻击检出率，同时付出一定误报代价。若实际系统更关注减少告警数量，可以在后续工作中通过阈值调整提高 Precision，但这可能降低 Recall。

ROC-AUC 与固定阈值下的 Precision、Recall、F1 不同。Precision、Recall 和 F1 依赖最终预测标签，而预测标签通常由概率分数和阈值得到；ROC-AUC 则评价模型对正负样本的整体排序能力。本文 XGBoost、LightGBM 等模型 ROC-AUC 均较高，说明模型分数具有较好区分能力；但高 ROC-AUC 并不自动保证某个固定阈值下误报最少，因此仍需结合 Precision、Recall 和 F1 进行判断。

多分类 macro 指标计算时，每个类别先独立计算 Precision、Recall 和 F1，再对类别求平均。如果某个模型只擅长识别 Normal、Generic 等多数类，即使 Accuracy 较高，macro F1 也可能不高。本文多分类 Random Forest 的 Accuracy 为 0.7521，但 Macro F1 为 0.4610，低于 Decision Tree 的 0.4964，说明 Accuracy 和 Macro F1 在类别不均衡场景下可能给出不同排序。

## 三、方法设计

### 3.1 整体设计思路

本文系统采用离线监督学习流程，整体思路为：读取 UNSW-NB15 官方训练集与测试集，完成字段清理和特征预处理，在统一数据划分下训练多种分类模型，并使用统一指标评估模型性能。

系统设计遵循两个原则。第一，二分类和多分类共享相同的数据读取与预处理逻辑，避免因不同模型输入处理不一致导致比较偏差。第二，所有模型均使用官方训练集训练、官方测试集评估，不在测试集上进行训练或重采样，从而保证测试结果能够反映模型在固定测试划分上的泛化表现。

从项目结构看，系统主要代码位于 `src/ids_ml/` 目录。`data.py` 负责读取训练集和测试集，并删除 `id` 字段；`preprocess.py` 负责划分特征和目标变量，并构建数值标准化与类别独热编码组成的预处理器；`train.py` 定义默认二分类、多分类和 MLP 模型；`evaluate.py` 定义 Accuracy、Precision、Recall、F1 和 ROC-AUC 等指标计算；`pipeline_binary.py` 和 `pipeline_multiclass.py` 分别执行二分类与多分类主实验；`additional_binary_experiments.py`、`mlp_binary_experiments.py`、`tabular_dl_binary_experiments.py` 和 `subset_reference_binary_experiments.py` 则负责补充实验。

本文实验比较采用“同任务同口径”的原则。默认二分类实验中，各模型使用相同训练集、测试集、目标标签和评价指标；默认多分类实验同理。扩展机器学习实验在同一二分类任务下重新运行 XGBoost reference，避免直接将不同脚本或不同运行条件下的结果混为一谈。表格深度学习实验由于训练成本较高，使用训练集前 160000 条样本构建子集，因此本文只将其与同口径重新运行的 `xgboost_subset_144k` 和 `mlp_subset_144k` 比较，不将其直接替代完整训练集上的默认二分类实验。

在可复现性方面，项目中可设置随机种子的模型统一使用 `random_state=42` 或 `random_seed=42`。这并不意味着所有运行环境下结果完全逐位一致，因为不同库版本、硬件和底层并行实现可能带来微小差异；但固定随机种子能够减少随机初始化、样本划分和模型训练过程中的不确定性，使实验结果更便于复核。

### 3.2 系统流程设计

系统流程包括数据加载、特征目标划分、预处理、模型训练、模型评估、结果保存和可解释性分析。

数据加载阶段，系统从 `data/raw/UNSW_NB15_training-set.csv` 和 `data/raw/UNSW_NB15_testing-set.csv` 读取数据，并删除 `id` 字段。特征目标划分阶段，二分类任务以 `label` 为目标变量，多分类任务以 `attack_cat` 为目标变量。预处理阶段，系统识别类别特征 `proto`、`service`、`state`，并对其使用 `OneHotEncoder(handle_unknown="ignore")` 进行独热编码；数值特征使用 `StandardScaler` 标准化。

模型训练阶段，系统将预处理器与分类器封装为 scikit-learn `Pipeline`，保证训练和测试阶段使用一致的数据变换。评估阶段，二分类模型计算 Accuracy、Precision、Recall、F1、ROC-AUC，并保存混淆矩阵图；多分类模型生成分类报告并保存整体 macro 指标。结果保存阶段，系统将二分类指标保存为 `results/binary_metrics.csv`，多分类指标保存为 `results/multiclass_metrics.csv`，扩展实验保存为 `results/additional_ml_binary_experiments.csv`。

系统流程可以进一步拆分为以下步骤。

第一，数据读取。程序调用 `load_unsw_nb15()` 分别读取训练集和测试集，并通过 `drop(columns=DROP_COLUMNS, errors="ignore")` 删除 `id`。`errors="ignore"` 的设置保证即使某些数据文件不存在该字段，程序也不会因此中断。

第二，任务划分。二分类任务调用 `split_binary_features_target()`，从输入特征中删除 `label` 和 `attack_cat`，并将 `label` 作为目标变量；多分类任务调用 `split_multiclass_features_target()`，同样删除目标相关字段，并将 `attack_cat` 作为目标变量。这样可以避免模型在输入中直接看到目标标签或目标相关字段，减少数据泄漏风险。

第三，预处理。数值特征使用 `StandardScaler`，类别特征使用 `OneHotEncoder(handle_unknown="ignore")`。对于普通树模型和线性模型，预处理器可以输出稀疏矩阵；对于 MLP、HistGradientBoosting、GradientBoosting、AdaBoost、SGD、GaussianNB 和 CatBoost 等需要或更适合稠密输入的模型，脚本设置 `dense_output=True`。

第四，模型训练与保存。默认二分类和多分类流程使用 `fit_model()` 将预处理器与模型封装为 `Pipeline` 并调用 `fit()` 训练。二分类模型保存为 `models/*_binary.joblib`，多分类模型保存为 `models/*_multiclass.joblib`。这些模型文件可用于后续离线加载和复查，但本文没有对其进行在线部署实验。

第五，结果输出与可视化。二分类实验输出 `binary_metrics.csv`，并为各模型保存混淆矩阵图；具有 `feature_importances_` 属性的树模型还会导出特征重要性 CSV。多分类实验输出 `multiclass_metrics.csv`。扩展实验、MLP 改进实验和表格深度学习实验分别输出独立 CSV，以便区分不同实验口径。

### 3.3 模型设计

#### 3.3.1 二分类

二分类任务目标是判断一条网络连接记录属于正常流量还是攻击流量。本文默认二分类实验比较 Logistic Regression、Decision Tree、Random Forest、XGBoost 和 MLP。其中 Logistic Regression 作为线性基线；Decision Tree 用于观察单棵树模型表现；Random Forest 用于评估 bagging 集成方法；XGBoost 用于评估梯度提升树方法；MLP 用于评估普通神经网络在当前表格数据上的效果。

默认二分类模型主要参数如下。Logistic Regression 设置 `max_iter=1000`；Decision Tree 使用 `random_state=42`；Random Forest 使用 `n_estimators=200`、`random_state=42`、`n_jobs=1`；XGBoost 使用 `n_estimators=200`、`max_depth=6`、`learning_rate=0.1`、`subsample=0.9`、`colsample_bytree=0.9`、`eval_metric="logloss"`、`random_state=42`、`n_jobs=1`；MLP 使用隐藏层 `(128, 64)`、ReLU、初始学习率 0.001、最大迭代 50 次和 early stopping。

二分类实验将攻击类作为正类进行指标计算。由于测试集包含 45332 条攻击样本和 37000 条正常样本，若只观察 Accuracy，可能无法充分区分误报和漏报。本文将 F1 作为二分类模型排序的主要指标，同时结合 ROC-AUC 判断模型概率分数的排序能力。对于入侵检测场景，较高 Recall 代表较低漏报风险，但如果 Precision 不足，误报会增加。因此，本文在结果分析中重点讨论 Precision 与 Recall 的组合关系。

二分类实验中的 XGBoost 被作为默认模型集合中的主模型，原因来自实验结果而不是预设偏好。具体来说，XGBoost 在默认二分类模型中取得最高 F1 和最高 ROC-AUC，并且在多分类任务中也取得最高 Macro F1。因此，本文将其作为主模型进行特征重要性和错误模式分析。扩展实验中 LightGBM 略高于 XGBoost reference，但差异很小，且 LightGBM 属于可选依赖，因此本文仍保持 XGBoost 为默认主模型。

#### 3.3.2 多分类

多分类任务目标是识别具体攻击类别。该任务包含 Normal、Generic、Exploits、Fuzzers、DoS、Reconnaissance、Analysis、Backdoor、Shellcode 和 Worms 共 10 个类别。与二分类相比，多分类不仅需要区分正常与攻击，还需要进一步区分攻击类型，因此受类别不均衡影响更明显。

多分类默认模型包括 Logistic Regression、Decision Tree、Random Forest、XGBoost 和 MLP。系统使用 `LabelEncoder` 将多分类标签编码为整数，以适配 XGBoost 的 `multi:softprob` 目标函数。多分类 Random Forest 设置 `n_estimators=50`；多分类 XGBoost 设置 `n_estimators=100`、`max_depth=6`、`learning_rate=0.1`、`subsample=0.9`、`colsample_bytree=0.9`、`objective="multi:softprob"` 和 `random_state=42`。预测完成后，系统将编码结果还原为原始类别名称，并计算 macro precision、macro recall 和 macro F1。

多分类任务保留 Normal 类，是因为 `attack_cat` 字段中正常样本对应 Normal。这样，多分类模型同时承担正常流量识别和攻击类型识别任务。与只在攻击样本内部识别攻击类别相比，这一设置更接近完整检测流程，但也使任务难度提高：模型不仅需要区分不同攻击类别，还需要区分正常流量与攻击流量。

当前项目没有对多分类类别进行重采样或类别权重调整，也没有保存每类详细分类报告。这样做的优点是实验流程相对直接，结果能够反映默认设置下模型表现；缺点是对少数类攻击识别不足的问题无法在当前结果文件中展开细粒度诊断。因此，本文在方法设计中明确将多分类实验定位为整体性能比较，而不是逐攻击类别的深入分析。

#### 3.3.4 MLP/深度学习补充实验

为了分析深度学习方法在当前表格入侵检测任务中的表现，本文设置了 MLP 调参与 Tabular ResNet-like 补充实验。

MLP 调参实验在基础 MLP 外增加两组配置。`mlp_tuned_longer` 将学习率降为 0.0005，最大迭代次数提高到 120，`n_iter_no_change` 设为 10，并使用 `alpha=0.0005`、`batch_size=512`；`mlp_tuned_wider` 使用更宽的隐藏层 `(256, 128)`，学习率 0.0005，最大迭代次数 120，`alpha=0.001`、`batch_size=512`。该实验用于判断基础 MLP 低于 XGBoost 是否主要由训练轮数不足或网络宽度不足造成。

Tabular ResNet-like 实验使用 PyTorch 实现，包含类别特征 embedding、数值特征标准化和残差块。`tabular_resnet_default` 使用宽度 256、残差块数 2、dropout 0.15、学习率 0.001、权重衰减 `1e-5`、batch size 2048、最大 epoch 20、patience 4，阈值固定为 0.5。`tabular_resnet_threshold_tuned` 使用宽度 256、残差块数 3、dropout 0.2、学习率 0.0008、权重衰减 `5e-5`、最大 epoch 25、patience 5，并在验证集上选择阈值。

MLP 改进实验的设计重点是排查基础 MLP 是否因为训练不足而低于 XGBoost。`mlp_tuned_longer` 降低学习率并提高最大迭代次数，意图让模型以更小步长训练更久；`mlp_tuned_wider` 在此基础上扩大隐藏层宽度，意图提高模型容量。两组实验都保留 early stopping，以避免训练过程在验证集表现不再提升后继续迭代。

Tabular ResNet-like 实验的设计重点是比较“普通神经网络”和“面向表格数据设计的神经网络”之间的差异。该模型没有使用独热编码作为类别特征最终表示，而是为每个类别特征构建 embedding，并将 embedding 与数值特征拼接后输入残差网络。残差块由 LayerNorm、Linear、ReLU、Dropout 和 Linear 组成，并将输入与变换结果相加后再经过 ReLU。该结构可以在一定程度上缓解深层全连接网络训练困难。

阈值调优实验用于观察提高 Recall 是否能够改善整体 F1。脚本在验证集上从 0.10 到 0.90 以 0.01 为步长搜索最佳 F1 阈值，再将该阈值用于测试集。实验结果显示，阈值调优版本的 Recall 提高，但 Precision 下降更多，最终 F1 反而降低。这说明阈值调优必须结合 Precision 与 Recall 权衡，不能只追求 Recall。

#### 3.3.5 新增主流机器学习扩展实验

为了进一步检验默认主模型 XGBoost 的合理性，本文在二分类任务中增加主流机器学习扩展实验。扩展实验保持与默认二分类实验相同的数据划分、预处理流程和评价指标，并重新运行 `xgboost_reference` 作为同口径基线。

扩展模型包括 LightGBM、CatBoost、HistGradientBoosting、GradientBoosting、ExtraTrees、带类别权重平衡的 ExtraTrees、AdaBoost tree、Linear SVM SGD 和 GaussianNB。其中 LightGBM 与 CatBoost 属于项目可选第三方依赖，并非默认依赖范围。扩展实验的目标不是替换默认主模型，而是观察在加入更多主流模型后，是否存在明显优于 XGBoost 的二分类模型。

扩展实验中的参数主要来自脚本固定配置。ExtraTrees 使用 300 棵树和 `max_features="sqrt"`；平衡版本额外设置 `class_weight="balanced"`；HistGradientBoosting 使用 `max_iter=200`、`learning_rate=0.05`、`max_leaf_nodes=31`；GradientBoosting 使用 150 个弱学习器、学习率 0.1、最大深度 3；AdaBoost 使用最大深度为 2 的决策树作为基学习器，迭代 200 次，学习率 0.5；SGDClassifier 使用 `modified_huber` 损失；LightGBM 使用 300 棵树、学习率 0.05、`num_leaves=31`；CatBoost 使用 300 次迭代、深度 6、学习率 0.05。

本文不将扩展实验称为“大规模模型搜索”，因为脚本中没有网格搜索、贝叶斯优化或多随机种子重复实验。扩展实验的性质是固定参数下的横向对照，用于观察主流模型在相同预处理和数据划分下的大致表现。基于这一实验性质，本文对 LightGBM 的表述保持谨慎：它在扩展实验中取得最高 F1，但只比 XGBoost reference 高约 0.00058。

## 四、实验结果

### 4.1 二分类实验结果

二分类实验结果如下：

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.8753 | 0.8234 | 0.9846 | 0.8968 | 0.9843 |
| MLP | 0.8737 | 0.8300 | 0.9691 | 0.8942 | 0.9749 |
| Random Forest | 0.8708 | 0.8172 | 0.9859 | 0.8936 | 0.9789 |
| Decision Tree | 0.8636 | 0.8243 | 0.9560 | 0.8853 | 0.8557 |
| Logistic Regression | 0.8099 | 0.7536 | 0.9726 | 0.8492 | 0.9561 |

结果显示，默认模型集合中 XGBoost 取得最高 F1，为 0.8968，同时 ROC-AUC 为 0.9843。MLP 和 Random Forest 的 F1 分别为 0.8942 和 0.8936，接近但未超过 XGBoost。Logistic Regression 的 Recall 达到 0.9726，但 Precision 只有 0.7536，说明线性模型在当前任务中更容易将正常流量误判为攻击流量。

根据系统保存的结果汇总，最佳二分类 XGBoost 的主要错误模式是误报：测试集中有 9571 条正常样本被误判为攻击。结合 Recall 0.9846 可知，该模型对攻击样本的检出能力较强，但仍存在较明显误报问题。对于入侵检测系统而言，高 Recall 有助于减少漏报，但过高误报会增加安全运维人员的告警处理成本，因此后续可围绕阈值调整、代价敏感学习和误报样本分析继续优化。

从模型族角度看，树模型整体优于线性模型。Decision Tree 的 F1 为 0.8853，高于 Logistic Regression 的 0.8492；Random Forest 的 F1 进一步提高到 0.8936，说明集成多棵树能够改善单棵树的稳定性和泛化效果；XGBoost 在此基础上取得 0.8968 的 F1，说明梯度提升树在当前二分类任务中具有更强拟合能力。

MLP 的结果值得单独分析。基础 MLP 的 Accuracy 为 0.8737，F1 为 0.8942，已经接近 Random Forest 和 XGBoost。这说明普通神经网络能够学习到部分有效非线性特征组合。但 MLP 的 ROC-AUC 为 0.9749，低于 XGBoost 的 0.9843 和 Random Forest 的 0.9789，说明其整体排序能力仍弱于树集成模型。结合后续 MLP 改进实验可以看到，普通 MLP 并未通过简单调参超过 XGBoost。

从 Precision 与 Recall 的平衡看，所有默认模型的 Recall 都相对较高，其中 Random Forest 达到 0.9859，XGBoost 达到 0.9846，Logistic Regression 也达到 0.9726。但 Precision 差异较明显：XGBoost 为 0.8234，Random Forest 为 0.8172，Logistic Regression 为 0.7536。也就是说，多个模型都能检出大部分攻击样本，但线性模型误报更多，导致 F1 下降。

特征重要性结果有助于理解树模型的判断依据。根据 `results/xgboost_feature_importance.csv`，XGBoost 前五个重要特征为 `sttl`、`proto_tcp`、`ct_srv_dst`、`proto_arp` 和 `ct_dst_sport_ltm`。其中 `sttl` 对应源到目的方向的 TTL 信息，`proto_tcp` 和 `proto_arp` 与协议类型有关，`ct_srv_dst` 和 `ct_dst_sport_ltm` 属于连接计数类特征。这说明 XGBoost 在当前任务中同时利用了网络包头统计、协议类别和连接行为统计信息。由于特征重要性只反映模型内部的相对贡献，并不等价于因果解释，因此本文仅将其作为模型可解释性辅助分析。

Random Forest 和 Decision Tree 的特征重要性也显示出类似趋势。Random Forest 前列特征包括 `sttl`、`ct_state_ttl`、`dload`、`rate` 和 `dttl`；Decision Tree 前列特征包括 `sttl`、`ct_srv_dst`、`sbytes`、`smean` 和 `ct_srv_src`。这些结果说明，树模型普遍关注 TTL、流量速率、字节数和连接计数等统计特征。不同模型的重要性排序并不完全一致，这是由于模型结构和训练机制不同造成的，不能简单理解为某一模型发现了唯一正确的特征集合。

### 4.2 多分类实验结果

多分类实验结果如下：

| 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.7707 | 0.5394 | 0.5443 | 0.5065 |
| Decision Tree | 0.7357 | 0.4934 | 0.5276 | 0.4964 |
| Random Forest | 0.7521 | 0.5179 | 0.4835 | 0.4610 |
| MLP | 0.7410 | 0.5129 | 0.4325 | 0.4147 |
| Logistic Regression | 0.6981 | 0.3993 | 0.3865 | 0.3400 |

结果显示，XGBoost 在多分类任务中仍取得最高 Accuracy 和 Macro F1，分别为 0.7707 和 0.5065。Decision Tree 的 Macro F1 为 0.4964，接近 XGBoost，但 Accuracy 低于 XGBoost。Random Forest 的 Accuracy 为 0.7521，但 Macro F1 为 0.4610，说明其整体正确率尚可，但对各类别的平均识别能力弱于 XGBoost 和 Decision Tree。

多分类 Macro F1 明显低于二分类 F1，主要原因来自两个方面。第一，多分类任务需要区分具体攻击类别，分类边界更复杂。第二，数据集中类别分布不均衡，Worms、Shellcode、Backdoor 等少数类训练样本较少，模型难以充分学习其稳定特征。当前项目仅保存了多分类整体 macro 指标，未保存每个攻击类别的详细分类报告，因此本文不能进一步断言某个具体攻击类别识别效果最好或最差。

从模型排序看，多分类中 XGBoost 仍保持领先，说明其在二分类和多分类两个任务上都具有较好稳定性。Decision Tree 的 Macro F1 为 0.4964，接近 XGBoost 的 0.5065，但 Accuracy 只有 0.7357，低于 XGBoost 的 0.7707。这说明 Decision Tree 可能在部分类别上取得较高平均表现，但整体预测正确率仍不如 XGBoost。Random Forest 的 Accuracy 为 0.7521，高于 Decision Tree，但 Macro F1 低于 Decision Tree，说明它可能更偏向多数类，少数类平均识别能力不足。

MLP 在多分类任务中的 Macro F1 为 0.4147，低于树模型；Logistic Regression 的 Macro F1 为 0.3400，是默认模型中最低的。该结果说明，随着类别数量增加，简单线性边界和普通 MLP 在当前设置下更难处理不同攻击类别之间的复杂差异。多分类任务不仅需要识别攻击和正常，还需要在攻击内部区分相近行为模式，因此比二分类任务对模型表达能力和数据平衡性要求更高。

本文多分类实验没有引入类别权重、过采样或欠采样策略，因此结果反映的是默认训练设置下各模型表现。考虑到 Worms、Shellcode、Backdoor 等类别样本极少，未来如果要提升多分类 Macro F1，可以重点考虑类别权重、少数类重采样、分层分类策略或专门针对少数类的阈值调整。但这些方法尚未在当前项目中实验，本文仅将其作为后续方向提出。

### 4.3 MLP 改进实验结果

MLP 改进实验结果如下：

| 实验 | Train(s) | n_iter | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP baseline | 326.44 | 37 | 0.8737 | 0.8300 | 0.9691 | 0.8942 | 0.9749 |
| MLP tuned longer | 179.43 | 55 | 0.8616 | 0.8129 | 0.9724 | 0.8855 | 0.9758 |
| MLP tuned wider | 222.73 | 46 | 0.8641 | 0.8126 | 0.9789 | 0.8881 | 0.9773 |

结果表明，延长训练轮数或扩大隐藏层宽度并未提升 F1。`mlp_tuned_longer` 和 `mlp_tuned_wider` 的 Recall 略高或接近基础 MLP，但 Precision 明显下降，导致 F1 低于基础 MLP。这说明在当前设置下，MLP 未超过 XGBoost 并不只是由于训练轮数不足或网络规模不足造成的，也可能与普通 MLP 对表格特征交互的建模能力有关。

从训练迭代情况看，基础 MLP 在 37 次迭代后停止，`mlp_tuned_longer` 在 55 次迭代后停止，`mlp_tuned_wider` 在 46 次迭代后停止。由于三组实验均启用 early stopping，实际迭代次数低于最大迭代次数。这说明模型训练并不是简单达到最大迭代上限后停止，而是在验证表现不再提升时提前结束。

`mlp_tuned_longer` 的 ROC-AUC 为 0.9758，高于基础 MLP 的 0.9749，但 F1 从 0.8942 降至 0.8855。`mlp_tuned_wider` 的 ROC-AUC 进一步提高到 0.9773，但 F1 仍只有 0.8881。这表明概率排序能力略有提高并不必然带来固定阈值下 F1 提升。如果阈值仍使用默认决策规则，模型可能在 Precision 与 Recall 的平衡上变差。

从 Precision 与 Recall 看，两组调参模型都提高了 Recall 或保持较高 Recall，但 Precision 降低到约 0.812。对于入侵检测任务，增加 Recall 有时是有价值的，但如果 Precision 下降过多，会导致误报数量增加。本文以 F1 作为综合指标，因此认为这两组 MLP 调参未改善基础 MLP。

该实验也说明，神经网络性能不应只从网络规模判断。更宽的网络具有更强表达能力，但也可能更容易学习到训练数据中的噪声或形成不利于测试集阈值分类的概率分布。对于表格入侵检测数据，模型结构、特征表示和阈值选择往往比单纯增加隐藏层宽度更重要。

### 4.4 表格深度学习补充实验结果

同口径子集实验结果如下：

| 实验 | Train(s) | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost subset 144k | 7.12 | 0.8795 | 0.8294 | 0.9833 | 0.8998 | 0.9845 |
| MLP subset 144k | 69.33 | 0.8688 | 0.8229 | 0.9707 | 0.8907 | 0.9758 |
| Tabular ResNet default | 361.61 | 0.8781 | 0.8355 | 0.9694 | 0.8975 | 0.9783 |
| Tabular ResNet threshold tuned | 403.63 | 0.8481 | 0.7894 | 0.9876 | 0.8775 | 0.9769 |

Tabular ResNet default 的 F1 为 0.8975，高于同口径 MLP 的 0.8907，说明面向表格数据设计的残差结构相较普通 MLP 有一定改进。但该模型仍略低于同口径 XGBoost 的 0.8998，并且训练耗时为 361.61 秒，高于 XGBoost 的 7.12 秒。阈值调优版本虽然 Recall 提高到 0.9876，但 Precision 降至 0.7894，最终 F1 降至 0.8775。

因此，本文不将深度学习补充实验解释为深度模型优于树模型，而是认为：在当前数据、特征和参数设置下，表格 ResNet-like 模型优于普通 MLP，但尚未超过 XGBoost；树模型在性能和训练效率上仍具有优势。

同口径子集实验是理解该结果的关键。因为 Tabular ResNet-like 只使用训练集前 160000 条样本，并从中划分训练与验证子集，如果直接将其与完整训练集上的 XGBoost 比较并不公平。因此，项目重新运行了 `xgboost_subset_144k` 和 `mlp_subset_144k`。在该口径下，XGBoost 的 F1 为 0.8998，MLP 的 F1 为 0.8907，Tabular ResNet default 的 F1 为 0.8975。由此可以得出两个受限但可靠的结论：第一，Tabular ResNet default 明显优于同口径 MLP；第二，它仍略低于同口径 XGBoost。

训练耗时方面，XGBoost subset 训练耗时 7.12 秒，MLP subset 为 69.33 秒，Tabular ResNet default 为 361.61 秒，阈值调优版本为 403.63 秒。虽然不同库和实现之间的耗时不能简单等同于算法本身效率，但在当前项目运行结果中，树模型训练成本明显低于深度模型。这对于需要频繁重训或快速迭代的实验系统具有实际意义。

阈值调优版本的结果进一步说明，追求更高 Recall 可能损害 F1。该模型 Recall 达到 0.9876，高于 Tabular ResNet default 的 0.9694，也高于同口径 XGBoost 的 0.9833；但 Precision 只有 0.7894，低于其他模型，最终 F1 只有 0.8775。若实际场景极度重视漏报，可以考虑这种高 Recall 设置；但在本文以 F1 为综合指标的评价口径下，该模型不是更优选择。

### 4.5 新增主流机器学习扩展实验结果

扩展二分类实验结果如下：

| 模型 | Train(s) | Accuracy | Precision | Recall | F1 | ROC-AUC |
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

扩展实验显示，LightGBM 取得最高 F1，为 0.8975，略高于 `xgboost_reference` 的 0.8969；LightGBM 的 ROC-AUC 为 0.9855，也略高于 `xgboost_reference` 的 0.9842。两者 F1 差值约为 0.00058，提升幅度很小。因此，本文仅将 LightGBM 表述为扩展实验中表现最好的补充对照模型，而不将其描述为显著优于 XGBoost。

HistGradientBoosting 和 CatBoost 也取得较高性能，F1 分别为 0.8956 和 0.8947，但均低于 XGBoost reference。ExtraTrees、GradientBoosting、AdaBoost tree、Linear SVM SGD 和 GaussianNB 表现逐步下降。其中 GaussianNB 虽然 Precision 接近 1，但 Recall 仅为 0.1941，说明其只识别出少量攻击样本，不适合作为当前系统主模型。

根据 `results/lightgbm_feature_importance.csv`，LightGBM 前五个重要特征为 `smean`、`sbytes`、`ct_srv_src`、`ct_srv_dst` 和 `ct_dst_src_ltm`。这些特征主要反映流量字节统计和连接计数信息，说明在当前二分类任务中，流量大小、连接关系和会话行为统计对攻击识别具有较高贡献。

从训练耗时看，LightGBM 训练耗时 4.60 秒，XGBoost reference 为 7.77 秒，HistGradientBoosting 为 14.48 秒，CatBoost 为 17.18 秒。GradientBoosting、AdaBoost 和 ExtraTrees 系列耗时明显更长，其中 ExtraTrees balanced 达到 1626.40 秒。需要说明的是，这些耗时来自当前项目运行记录，受库实现、硬件、线程数和参数设置影响，不能作为所有环境下的通用结论。但在本文项目中，LightGBM 同时取得最高 F1 和较短训练时间，说明它是有价值的扩展对照模型。

从模型效果看，排名靠前的 LightGBM、XGBoost reference、HistGradientBoosting 和 CatBoost 都属于梯度提升类模型或相近思想下的实现。这与第二章理论分析一致：梯度提升模型能够逐步修正前一轮模型错误，适合处理结构化特征中的非线性关系。相比之下，Linear SVM SGD 和 GaussianNB 的 F1 明显较低，说明简单线性或强独立性假设模型难以充分适应当前入侵检测数据。

ExtraTrees 与 ExtraTrees balanced 的结果也值得注意。加入 `class_weight="balanced"` 后，F1 从 0.8880 提升到 0.8888，提升幅度很小，且训练耗时略长。由于本文没有进一步调整树数量、深度或采样策略，不能断言类别权重对该任务无效；只能说明在当前 ExtraTrees 固定参数设置下，平衡类别权重没有带来明显提升。

LightGBM 特征重要性与 XGBoost 特征重要性存在差异。XGBoost 更突出 `sttl`，LightGBM 则将 `smean`、`sbytes`、`ct_srv_src` 等排在前列。这说明不同梯度提升实现虽然性能接近，但在特征利用偏好上可能不同。本文不据此判断哪个模型解释“更正确”，而是将其作为模型行为差异的辅助说明。

综合第四章实验结果，可以得到一个稳健结论：在当前 UNSW-NB15 官方划分、统一预处理和固定参数设置下，梯度提升类模型整体表现最好。XGBoost 是默认模型集合中的最佳模型，LightGBM 是扩展实验中的最佳补充模型；普通 MLP 和 Tabular ResNet-like 模型具有一定效果，但没有在当前实验中超过树提升模型。

## 五、总结与展望

本文基于 UNSW-NB15 数据集和现有项目系统，完成了网络入侵检测离线实验研究。系统实现了统一的数据读取、预处理、模型训练和评估流程，并围绕二分类、多分类、MLP 补充实验、表格深度学习补充实验和主流机器学习扩展实验进行了比较分析。

实验结果表明，在默认模型集合中，XGBoost 在二分类和多分类任务上均取得最优表现。二分类任务中，XGBoost 的 F1 为 0.8968，ROC-AUC 为 0.9843；多分类任务中，XGBoost 的 Accuracy 为 0.7707，Macro F1 为 0.5065。MLP 和 Random Forest 在二分类任务中表现接近 XGBoost，但仍略低；Logistic Regression 与 GaussianNB 等简单模型在当前任务中存在明显不足。

深度学习补充实验表明，单纯延长 MLP 训练或扩大网络宽度并未提升 F1。Tabular ResNet-like 模型相较普通 MLP 有改进，但在同口径子集实验中仍略低于 XGBoost，且训练耗时更长。扩展机器学习实验中，LightGBM 以 F1 0.8975 略高于 XGBoost reference 的 0.8969，但差异仅约 0.00058，因此本文仍将 XGBoost 作为默认模型集合中的主模型，将 LightGBM 作为扩展实验中表现最好的补充对照模型。

本文的主要工作可以概括为四点。第一，基于 UNSW-NB15 官方训练集和测试集构建了统一的离线入侵检测实验流程，明确区分二分类和多分类任务。第二，在相同预处理流程下比较了线性模型、单棵树、随机森林、梯度提升树、MLP 和多个扩展机器学习模型，形成了较完整的模型横向对照。第三，通过 MLP 改进实验和 Tabular ResNet-like 实验分析了深度学习方法在当前表格数据上的表现，避免直接假设深度模型一定优于传统机器学习模型。第四，结合特征重要性和混淆矩阵错误模式，对模型结果进行了可解释性和误报风险分析。

从系统角度看，当前项目已经具备离线实验闭环。输入端能够读取固定数据集，处理端能够完成标准化和独热编码，训练端能够运行多类模型，输出端能够保存指标、图像、模型文件和特征重要性文件。这为后续扩展提供了基础。例如，如果要继续优化二分类主模型，可以在现有 `pipeline_binary.py` 和 `additional_binary_experiments.py` 基础上加入超参数搜索；如果要深入分析多分类任务，可以在 `pipeline_multiclass.py` 中保存每类分类报告和混淆矩阵。

本文仍存在以下局限。第一，实验主要基于 UNSW-NB15 官方训练集和测试集划分，未进行 K 折交叉验证。第二，模型参数主要采用固定配置或少量人工调整，未进行大规模系统化超参数搜索。第三，多分类实验仅保存整体 macro 指标，未保存各攻击类别详细分类报告，因此无法对具体攻击类别识别难度进行更细粒度分析。第四，当前系统属于离线实验流程，尚未实现实时流量采集、在线特征提取和告警联动。

后续工作可从三个方向展开。第一，对 XGBoost 和 LightGBM 进行更系统的超参数搜索，并结合阈值调整分析误报与漏报之间的权衡。第二，补充多分类每类 precision、recall、F1 和混淆矩阵，重点分析 Worms、Shellcode、Backdoor 等少数类攻击的识别问题。第三，在保证特征可实时获取的前提下，探索在线流量特征提取、模型推理接口和告警输出流程，使离线实验系统逐步扩展为可验证的实时检测原型。

更具体地说，后续超参数优化可以优先围绕树提升模型展开。对于 XGBoost，可调整树数量、最大深度、学习率、子采样比例、列采样比例和正则化参数；对于 LightGBM，可调整 `num_leaves`、`learning_rate`、`n_estimators`、`subsample` 和 `colsample_bytree` 等参数。由于当前 LightGBM 与 XGBoost 的 F1 差异很小，后续应通过交叉验证或多次重复实验判断差异是否稳定，而不是仅依据一次固定划分结果断言某个模型显著更优。

针对误报问题，后续可在二分类模型上进行阈值分析。当前默认模型通常使用 0.5 或模型默认阈值产生类别标签，但在实际入侵检测中，可以根据业务目标选择不同阈值。如果希望减少漏报，可降低阈值以提高 Recall；如果希望减少误报，可提高阈值以提高 Precision。阈值调整需要结合混淆矩阵、Precision-Recall 曲线和安全运维成本共同判断。

针对多分类问题，后续应保存逐类别指标。只有获得每类 Precision、Recall、F1 和混淆矩阵，才能判断模型是否将 Backdoor 误判为 Exploits，是否将 Worms 误判为 Normal，或者是否只在多数类上表现较好。当前论文没有这些实验数据，因此没有展开具体类别错误分析。后续补充该部分后，论文可以进一步讨论少数类攻击识别困难的具体来源。

针对在线检测方向，后续需要解决两个前提问题。第一，UNSW-NB15 中的特征是否能够从实时网络流量中稳定提取；第二，离线训练得到的模型在真实网络分布下是否仍然有效。只有完成实时特征提取、模型推理延迟测试、数据漂移监测和告警验证后，才能将系统称为实时网络入侵检测系统。当前项目尚未完成这些步骤，因此本文保持离线实验系统的表述。

总体而言，本文基于真实项目实验结果得出的核心结论是：在当前 UNSW-NB15 离线实验设置下，梯度提升类树模型是最有效的模型族；XGBoost 是默认模型集合中的主模型，LightGBM 是扩展实验中表现最好的补充模型；深度学习方法具有研究价值，但在当前实验配置下没有超过树提升模型。该结论受数据集、预处理方式、参数设置和评价指标限制，后续仍需要通过更系统的实验进一步验证和完善。

## 参考文献

[1] DENNING D E. An intrusion-detection model[J]. IEEE Transactions on Software Engineering, 1987, SE-13(2): 222-232.

[2] MOUSTAFA N, SLAY J. UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)[C]//2015 Military Communications and Information Systems Conference. Canberra: IEEE, 2015.

[3] MOUSTAFA N, SLAY J. The evaluation of Network Anomaly Detection Systems: statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set[J]. Information Security Journal: A Global Perspective, 2016, 25(1-3): 18-31.

[4] BREIMAN L. Random forests[J]. Machine Learning, 2001, 45(1): 5-32.

[5] FRIEDMAN J H. Greedy function approximation: a gradient boosting machine[J]. The Annals of Statistics, 2001, 29(5): 1189-1232.

[6] CHEN T, GUESTRIN C. XGBoost: a scalable tree boosting system[C]//Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York: ACM, 2016: 785-794.

[7] KE G, MENG Q, FINLEY T, et al. LightGBM: a highly efficient gradient boosting decision tree[C]//Advances in Neural Information Processing Systems 30. Red Hook: Curran Associates, 2017: 3146-3154.

[8] PROKHORENKOVA L, GUSEV G, VOROBEV A, et al. CatBoost: unbiased boosting with categorical features[C]//Advances in Neural Information Processing Systems 31. Red Hook: Curran Associates, 2018: 6638-6648.

[9] GORISHNIY Y, RUBACHEV I, KHRULKOV V, et al. Revisiting deep learning models for tabular data[C]//Advances in Neural Information Processing Systems 34. Red Hook: Curran Associates, 2021: 18932-18943.

[10] FAWCETT T. An introduction to ROC analysis[J]. Pattern Recognition Letters, 2006, 27(8): 861-874.
