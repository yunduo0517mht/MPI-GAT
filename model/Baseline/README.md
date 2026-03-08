# Baseline Models for Ablation Study

本目录包含消融实验的基线模型实现。

## 📁 文件结构

```
model/Baseline/
├── __init__.py           # 模块初始化
├── mlp_models.py         # MLP模型定义
└── README.md             # 本文件
```

## 🎯 模型说明

### Stage 1: Baseline_MLP

**用途**: 只使用新闻文本，建立弱基线

**输入**: 新闻 BERT embedding (1024维)

**输出**: 6类分类预测

**架构**:
```
Input (1024) → Linear(1024, 512) → ReLU → Dropout(0.5)
             → Linear(512, 256) → ReLU → Dropout(0.5)
             → Linear(256, 6) → Output
```

**参数量**: ~657K

### Stage 2: Comment_MLP (TODO)

**用途**: 使用新闻文本 + 评论聚合特征

**输入**:
- 新闻 BERT embedding (1024维)
- 评论聚合特征 (1024维)

**输出**: 6类分类预测

**状态**: 等待评论网络数据准备完成

## 🚀 使用方法

### 1. 准备数据

确保你已经生成了 `data/liar_clean_embeddings.pt` 文件，格式如下：

```python
{
    "news_id_1": {
        "embedding": torch.Tensor([1024]),  # BERT embedding
        "label": 0  # 0-5, 六类标签
    },
    "news_id_2": {
        "embedding": torch.Tensor([1024]),
        "label": 2
    },
    ...
}
```

### 2. 训练 Baseline_MLP

```bash
python train_baseline_mlp.py
```

**配置参数** (在 `train_baseline_mlp.py` 中修改):
- `data_path`: 数据文件路径 (默认: `data/liar_clean_embeddings.pt`)
- `output_dir`: 输出目录 (默认: `./checkpoints/baseline_mlp`)
- `embedding_dim`: embedding维度 (默认: 1024)
- `hidden_dims`: 隐藏层维度 (默认: [512, 256])
- `batch_size`: 批次大小 (默认: 32)
- `num_epochs`: 训练轮数 (默认: 50)
- `learning_rate`: 学习率 (默认: 1e-4)

### 3. 输出结果

训练完成后，会在 `checkpoints/baseline_mlp/` 目录下生成：

- `best_model.pth`: 最佳模型checkpoint
- `training_history.pth`: 训练历史记录

## 📊 预期结果

根据论文消融实验设计，预期结果：

| Stage | Model | Input | Expected Acc |
|-------|-------|-------|--------------|
| 1 | Baseline_MLP | News text only | 25-30% |
| 2 | Comment_MLP | News + Comments | 32-36% |
| 3 | MPI-GAT (w/o metapath) | Heterograph | 38-42% |
| 4 | MPI-GAT (Full) | Heterograph + Metapaths | 43-47% |

## 🔍 代码测试

测试模型是否正常工作：

```bash
cd model/Baseline
python mlp_models.py
```

应该看到：

```
============================================================
Testing Baseline Models
============================================================

1. Testing Baseline_MLP
------------------------------------------------------------
Model created!
  Total parameters: 657,926
...
[PASS] All models tested successfully!
============================================================
```

## 📝 论文写作

**Table X: Ablation Study**

| # | Model Configuration | Acc | F1 | Δ |
|---|---------------------|-----|----|----|
| 1 | Text-only (Baseline_MLP) | XX.X | XX.X | - |
| 2 | + Comment features | XX.X | XX.X | +X.X |
| 3 | + GNN structure | XX.X | XX.X | +X.X |
| 4 | + Metapath features | XX.X | XX.X | +X.X |

## 🛠️ TODO

- [x] 实现 Baseline_MLP
- [x] 实现训练脚本
- [ ] 准备评论聚合特征数据
- [ ] 实现 Comment_MLP 训练脚本
- [ ] 运行所有baseline实验
- [ ] 整理实验结果到论文表格
