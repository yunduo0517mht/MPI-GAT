# Baseline Models for Ablation Study

本目录包含消融实验的基线模型实现。

## 文件结构

```
model/Baseline/
├── __init__.py           # 模块初始化
├── mlp_models.py         # MLP模型定义
└── README.md             # 本文件
```

## 模型说明

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

### Stage 2: Feature_MLP

**用途**: 新闻 + 额外特征 (消融实验)

**支持的配置**:

| 配置 | 额外特征 | 输入维度 | 说明 |
|------|----------|----------|------|
| news_author | 作者 embedding | 2048 | 证明作者特征有帮助 |
| news_comment | 评论平均 (平均池化) | 2048 | 证明评论特征有帮助 |
| news_author_comment | 作者 + 评论 | 3072 | 证明两者互补 |

**架构**:
```
Input (2048/3072) → Linear(→512) → ReLU → Dropout(0.5)
                  → Linear(512, 256) → ReLU → Dropout(0.5)
                  → Linear(256, 6) → Output
```

## 使用方法

### 1. 训练 Stage 1 (只用新闻)

```bash
python train_baseline_mlp.py
```

### 2. 训练 Stage 2 (额外特征消融)

```bash
# 新闻 + 作者
python train_mlp_features.py --features author

# 新闻 + 评论
python train_mlp_features.py --features comment

# 新闻 + 作者 + 评论
python train_mlp_features.py --features author,comment

# 多次运行取平均
python train_mlp_features.py --features author --runs 5
```

### 3. 输出目录

```
checkpoints/
├── baseline_mlp/           # Stage 1
│   └── <timestamp>/
│
└── mlp_ablation/           # Stage 2
    ├── news_author/
    │   └── <timestamp>/
    ├── news_comment/
    │   └── <timestamp>/
    └── news_author_comment/
        └── <timestamp>/
```

### 4. 数据格式

Stage 2 数据格式 (每条新闻一个JSON文件):
```json
{
    "news_id": "news_1",
    "news_label": "false",
    "news_embedding": [1024 floats],
    "author_embedding": [1024 floats],
    "comments": [
        {"embedding": [1024 floats], ...},
        ...
    ]
}
```

## 预期结果

| Stage | Model | Input | Expected Acc |
|-------|-------|-------|--------------|
| 1 | Baseline_MLP | News only | 25-30% |
| 2a | Feature_MLP (author) | News + Author | ~32% |
| 2b | Feature_MLP (comment) | News + Comment | ~32% |
| 2c | Feature_MLP (both) | News + Author + Comment | ~35% |
| 3 | MPI-GAT | Heterograph | 38-42% |
| 4 | MPI-GAT (Full) | Heterograph + Metapaths | 43-47% |

## 论文写作

**Table: Ablation Study on Feature Contributions**

| # | Features | Acc | F1 | Δ |
|---|----------|-----|----|----|
| 1 | News only | XX.X | XX.X | - |
| 2 | + Author | XX.X | XX.X | +X.X |
| 3 | + Comment | XX.X | XX.X | +X.X |
| 4 | + Author + Comment | XX.X | XX.X | +X.X |

## TODO

- [x] 实现 Baseline_MLP
- [x] 实现 Feature_MLP
- [x] 实现训练脚本
- [ ] 运行 Stage 2 消融实验
- [ ] 整理实验结果到论文表格
