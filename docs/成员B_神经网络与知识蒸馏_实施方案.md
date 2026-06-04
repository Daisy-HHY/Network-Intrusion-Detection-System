# 成员B：神经网络教师模型 & 知识蒸馏 — 实施方案

## Context

本次课程作业要求小组成员分别实现 SVM 和神经网络（含蒸馏）对 CIFAR-10 进行分类。成员B负责：
1. 构建教师模型 M（VGG-like CNN）在 CIFAR-10 上训练
2. 构建学生模型（小型3层CNN）
3. 实现知识蒸馏（5组温度 T=1,2,4,7,10）
4. 对比"学生直接训练" vs "学生蒸馏训练"的精度差异

当前项目是 UNSW-NB15 网络入侵检测系统，无 CIFAR-10 相关代码。需要在 `src/cifar10/` 新建完整子模块。

---

## 文件结构

```
src/cifar10/
├── __init__.py              # 包初始化
├── config.py                # 超参数、路径、常量配置
├── data.py                  # CIFAR-10 数据加载与增强
├── models.py                # 教师模型(VGG-like) + 学生模型(SmallCNN)
├── distillation.py          # 蒸馏损失函数 + 蒸馏训练循环
├── train.py                 # 统一训练入口（教师/学生直接/学生蒸馏）
├── evaluate.py              # 评估指标、混淆矩阵、对比图表生成
└── run_all.py               # 一键运行全部实验并保存结果
```

---

## 实施步骤

### Step 1: `src/cifar10/__init__.py` — 包初始化
- 空文件，仅使 `src/cifar10` 成为可导入的 Python 包。

### Step 2: `src/cifar10/config.py` — 全局配置
参考现有 `src/ids_ml/config.py` 的模式，集中管理所有配置：

```
关键配置项：
- RANDOM_STATE = 42
- NUM_CLASSES = 10
- BATCH_SIZE = 128
- TEACHER_EPOCHS = 100（含 early stopping patience=10）
- STUDENT_EPOCHS = 80（含 early stopping patience=8）
- LEARNING_RATE_TEACHER = 0.001
- LEARNING_RATE_STUDENT = 0.005
- WEIGHT_DECAY = 5e-4
- TEMPERATURES = [1, 2, 4, 7, 10]  # 5组温度实验
- ALPHA = 0.3  # 蒸馏损失中 hard loss 的权重
- FIGURES_DIR / MODELS_DIR / RESULTS_DIR  路径定义
```

### Step 3: `src/cifar10/data.py` — 数据加载与增强

**功能**：
- 使用 `torchvision.datasets.CIFAR10` 自动下载并加载 CIFAR-10
- 定义训练集增强策略：`RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` + `ToTensor()` + `Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))`
- 定义测试集变换：`ToTensor()` + `Normalize`（均值/标准差同上）
- 返回 `DataLoader`（训练/测试），可选验证集（从训练集分出 5000 样本）

**关键函数**：
- `get_dataloaders(batch_size, val_ratio=0.1)` → 返回 `(train_loader, val_loader, test_loader)`

### Step 4: `src/cifar10/models.py` — 模型定义

#### 教师模型：`VGGLikeTeacher`

```
架构设计（适合 CIFAR-10 的 32×32 输入）：
├── Conv Block 1: Conv(3→64, 3×3) → BN → ReLU → Conv(64→64, 3×3) → BN → ReLU → MaxPool(2×2)
├── Conv Block 2: Conv(64→128, 3×3) → BN → ReLU → Conv(128→128, 3×3) → BN → ReLU → MaxPool(2×2)
├── Conv Block 3: Conv(128→256, 3×3) → BN → ReLU → Conv(256→256, 3×3) → BN → ReLU → MaxPool(2×2)
├── Conv Block 4: Conv(256→512, 3×3) → BN → ReLU → Conv(512→512, 3×3) → BN → ReLU → MaxPool(2×2)
├── AdaptiveAvgPool(1×1)  (处理 2×2→1×1)
├── FC: 512 → 512 → Dropout(0.5) → 10

参数量：约 5-7M
预期准确率：92-94%
输出：返回 logits（用于蒸馏）+ 可选返回 softmax 后的概率
```

#### 学生模型：`SmallCNNStudent`

```
架构设计（轻量级 3 层卷积）：
├── Conv Block 1: Conv(3→32, 3×3, padding=1) → BN → ReLU → MaxPool(2×2)
├── Conv Block 2: Conv(32→64, 3×3, padding=1) → BN → ReLU → MaxPool(2×2)
├── Conv Block 3: Conv(64→128, 3×3, padding=1) → BN → ReLU → MaxPool(2×2)
├── Flatten: 128 × 4 × 4 = 2048
├── FC: 2048 → 256 → ReLU → Dropout(0.3) → 10

参数量：约 0.5M（~500K）
预期直接训练准确率：80-85%
预期蒸馏后准确率：83-87%
```

**关键设计**：
- 两个模型都有 `forward(x)` 返回 logits
- 学生模型额外提供 `forward_with_features(x)` 返回中间特征（用于可选的特征蒸馏扩展）
- 模型可打印架构摘要（参数量统计）

### Step 5: `src/cifar10/distillation.py` — 蒸馏核心

**核心公式**：
```
Distillation Loss = α × CrossEntropy(student_logits, hard_labels)
                  + (1-α) × T² × KLDiv(student_soft, teacher_soft)

其中：
  student_soft = softmax(student_logits / T)
  teacher_soft = softmax(teacher_logits / T)
  T = 温度参数
  α = hard loss 权重（默认 0.3）
```

**关键函数**：
- `distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)` → 计算蒸馏损失
- `train_one_epoch_distill(student, teacher, train_loader, optimizer, temperature, alpha, device)` → 一个 epoch 的蒸馏训练
- `train_one_epoch_direct(model, train_loader, optimizer, device)` → 一个 epoch 的直接训练（用于对比基线）
- `evaluate(model, val_loader, device)` → 返回 (loss, accuracy, top5_accuracy)

### Step 6: `src/cifar10/train.py` — 统一训练管线

**训练流程**：

#### 阶段 A：训练教师模型
```
1. 加载数据
2. 创建 VGGLikeTeacher
3. SGD(lr=0.1, momentum=0.9, weight_decay=5e-4) + CosineAnnealingLR
4. 训练最多 100 epochs，early stopping (patience=10)
5. 保存最优模型到 models/teacher_vgg_cifar10.pth
6. 记录训练曲线（loss/acc per epoch）
7. 在测试集上评估 → 保存结果
```

#### 阶段 B：学生模型直接训练（baseline）
```
1. 创建 SmallCNNStudent
2. SGD(lr=0.005, momentum=0.9, weight_decay=5e-4) + CosineAnnealingLR
3. 训练最多 80 epochs，early stopping (patience=8)
4. 保存最优模型到 models/student_direct_cifar10.pth
5. 在测试集上评估 → 保存结果
```

#### 阶段 C：学生模型蒸馏训练（5组温度实验）
```
对每个 T in [1, 2, 4, 7, 10]:
  1. 加载已训练好的教师模型（冻结参数）
  2. 创建新的 SmallCNNStudent（从零初始化）
  3. Adam(lr=0.001) + CosineAnnealingLR
  4. 使用 distillation_loss 训练最多 80 epochs
  5. 保存最优模型到 models/student_distill_T{t}_cifar10.pth
  6. 在测试集上评估 → 保存结果
```

**关键函数**：
- `train_teacher(device, save_dir)` → 训练教师并返回评估结果
- `train_student_direct(device, save_dir)` → 直接训练学生
- `train_student_distilled(device, teacher_path, temperatures, alpha, save_dir)` → 蒸馏训练
- 所有训练函数返回 `dict` 包含：accuracy, top5_acc, train_time, best_epoch, history

### Step 7: `src/cifar10/evaluate.py` — 评估与可视化

**生成以下图表**（用于 Word 报告）：

1. **教师模型训练曲线**：`teacher_training_curves.png`
   - 左子图：训练/验证 Loss vs Epoch
   - 右子图：训练/验证 Accuracy vs Epoch

2. **学生直接训练曲线**：`student_direct_training_curves.png`

3. **蒸馏温度对比柱状图**：`distillation_temperature_comparison.png`
   - X 轴：T=1,2,4,7,10 + 学生直接训练 + 教师模型
   - Y 轴：测试准确率
   - 用不同颜色区分

4. **混淆矩阵**：
   - `teacher_confusion_matrix.png`
   - `student_direct_confusion_matrix.png`
   - `student_distill_best_confusion_matrix.png`（最佳温度对应的学生）

5. **模型参数量对比饼图**：`model_size_comparison.png`

6. **蒸馏过程 Loss 曲线**（各温度）：`distillation_loss_curves.png`

**保存以下 CSV 结果**：
- `results/cifar10_teacher_results.csv`
- `results/cifar10_student_direct_results.csv`
- `results/cifar10_distillation_comparison.csv`（核心表格：温度/准确率/训练时间）

### Step 8: `src/cifar10/run_all.py` — 一键运行

主入口脚本，串联全部实验：
```
1. 检测并设置设备（CUDA / CPU）
2. 调用 train_teacher() → 保存教师模型
3. 调用 train_student_direct() → 保存学生基线
4. 循环 5 组温度调用 train_student_distilled()
5. 调用 evaluate 生成所有图表和 CSV
6. 打印汇总表格到终端
```

运行命令：`python -m src.cifar10.run_all`

---

## 技术要点

### 蒸馏损失实现细节
```python
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    # Soft targets: 教师的软标签
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)

    # Hard targets: 真实标签
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * hard_loss + (1 - alpha) * soft_loss
```

### 训练技巧
- 教师模型使用 **SGD + Momentum + Weight Decay + CosineAnnealing**（经典 VGG 训练方案）
- 学生直接训练使用 **SGD**（保持一致性）
- 学生蒸馏训练使用 **Adam**（蒸馏场景下收敛更快）
- 所有实验使用 **early stopping** 防止过拟合
- 教师推理时使用 `torch.no_grad()` + `model.eval()`

### 设备兼容
- 自动检测 CUDA 可用性，无 GPU 时回退到 CPU
- 注意：CPU 训练教师模型约需 1-2 小时（100 epochs），建议有 GPU

---

## 预期实验结果格式

### 核心对比表格（distillation_comparison.csv）

| 实验 | 模型 | 温度T | 测试准确率 | Top-5准确率 | 训练时间(s) | 参数量 |
|------|------|-------|-----------|------------|------------|--------|
| 教师模型 | VGG-like | - | ~93% | ~99.5% | ~1800 | ~7M |
| 学生直接 | SmallCNN | - | ~83% | ~98% | ~600 | ~0.5M |
| 蒸馏T=1 | SmallCNN | 1 | ~84% | ~98% | ~800 | ~0.5M |
| 蒸馏T=2 | SmallCNN | 2 | ~85% | ~98.5% | ~800 | ~0.5M |
| 蒸馏T=4 | SmallCNN | 4 | ~86% | ~98.5% | ~800 | ~0.5M |
| 蒸馏T=7 | SmallCNN | 7 | ~85.5% | ~98.5% | ~800 | ~0.5M |
| 蒸馏T=10 | SmallCNN | 10 | ~84% | ~98% | ~800 | ~0.5M |

---

## 依赖更新

需要在 `requirements.txt` 中追加：
```
torch>=2.0
torchvision>=0.15
```

---

## 验证方案

1. **单元验证**：导入模块，确认模型可实例化且前向传播正确
   ```bash
   python -c "from src.cifar10.models import VGGLikeTeacher, SmallCNNStudent; import torch; x=torch.randn(2,3,32,32); print(VGGLikeTeacher()(x).shape, SmallCNNStudent()(x).shape)"
   ```
   预期输出：`torch.Size([2, 10]) torch.Size([2, 10])`

2. **快速冒烟测试**（减少 epoch 确认流程跑通）：
   ```bash
   python -m src.cifar10.run_all --smoke
   ```
   用 2-3 个 epoch 快速验证全部流程

3. **完整实验**：
   ```bash
   python -m src.cifar10.run_all
   ```
   检查 `results/` 目录下生成所有 CSV 和 PNG

4. **结果检查**：
   - 教师模型准确率 > 90%
   - 学生蒸馏最佳准确率 > 学生直接训练准确率
   - 所有 6 张图表正确生成
   - CSV 表格数据完整无空值
