# PyTorch Norm 算子测试集

24 个 norm 类算子，全部来源于 PyTorch (`torch.nn` / `torch.nn.functional` / `torch.linalg` / `torch.nn.utils`)，
分为 **3 组 × 8 个**，格式遵循 KernelBench level1 标准。

## 分组总览

### Group 1 — 统计归一化 + 向量范数

| # | 文件 | 算子 | 难度 | PyTorch API | 计算模式 |
|---|------|------|------|-------------|---------|
| 1 | `01_LayerNorm.py` | LayerNorm | 中等 | `nn.LayerNorm` | reduce(dim=-1) + affine |
| 2 | `02_RMSNorm.py` | RMS Normalization | 中等 | 手动 (LLaMA) | square → mean → rsqrt → mul |
| 3 | `03_L2Normalize.py` | L2 归一化 | 中等 | `F.normalize(p=2)` | reduce → broadcast div |
| 4 | `04_FrobeniusNorm.py` | Frobenius 范数归一化 | 中等 | `torch.norm(p='fro')` | 全局 reduce → div |
| 5 | `05_VectorNorm.py` | 向量 L2 范数 | 中等 | `linalg.vector_norm(ord=2)` | square → sum → sqrt |
| 6 | `06_BatchNorm2d.py` | Batch Normalization (2D) | **困难** | `nn.BatchNorm2d` | batch reduce + running stats |
| 7 | `07_NuclearNorm.py` | 核范数 | **困难** | `linalg.matrix_norm('nuc')` | SVD → sum(σ) |
| 8 | `08_SpectralNorm.py` | 谱归一化 | **困难** | `nn.utils.spectral_norm` | power iteration + matmul |

### Group 2 — 分组/实例归一化 + 权重归一化

| # | 文件 | 算子 | 难度 | PyTorch API | 计算模式 |
|---|------|------|------|-------------|---------|
| 1 | `01_GroupNorm.py` | Group Normalization | 中等 | `nn.GroupNorm` | 分组 reduce + affine |
| 2 | `02_InstanceNorm2d.py` | Instance Norm (2D) | 中等 | `nn.InstanceNorm2d` | per-instance reduce |
| 3 | `03_L1Normalize.py` | L1 归一化 | 中等 | `F.normalize(p=1)` | abs → sum → div |
| 4 | `04_LocalResponseNorm.py` | 局部响应归一化 | 中等 | `nn.LocalResponseNorm` | 通道滑动窗口 |
| 5 | `05_VectorNormL1.py` | 向量 L1 范数 | 中等 | `linalg.vector_norm(ord=1)` | abs → sum |
| 6 | `06_BatchNorm1d.py` | Batch Normalization (1D) | **困难** | `nn.BatchNorm1d` | batch reduce + running stats |
| 7 | `07_InstanceNorm3d.py` | Instance Norm (3D) | **困难** | `nn.InstanceNorm3d` | 5D per-instance reduce |
| 8 | `08_WeightNorm.py` | 权重归一化 | **困难** | `nn.utils.weight_norm` | g * v / ‖v‖ + matmul |

### Group 3 — 相似度/距离 + 矩阵范数

| # | 文件 | 算子 | 难度 | PyTorch API | 计算模式 |
|---|------|------|------|-------------|---------|
| 1 | `01_InstanceNorm1d.py` | Instance Norm (1D) | 中等 | `nn.InstanceNorm1d` | 3D per-instance reduce |
| 2 | `02_CosineSimilarity.py` | 余弦相似度 | 中等 | `nn.CosineSimilarity` | dot + L2 + div |
| 3 | `03_PairwiseDistance.py` | 成对 Lp 距离 | 中等 | `nn.PairwiseDistance` | diff → pow → reduce → root |
| 4 | `04_VectorNormInf.py` | 无穷范数 | 中等 | `linalg.vector_norm(ord=inf)` | abs → max |
| 5 | `05_MatrixInfNorm.py` | 矩阵无穷范数 | 中等 | `linalg.matrix_norm(ord=inf)` | abs → rowsum → max |
| 6 | `06_BatchNorm3d.py` | Batch Normalization (3D) | **困难** | `nn.BatchNorm3d` | 5D batch reduce + running stats |
| 7 | `07_Renorm.py` | 切片重归一化 | **困难** | `torch.renorm` | 条件 per-slice clamp |
| 8 | `08_LayerNormMultiDim.py` | 多维 LayerNorm | **困难** | `nn.LayerNorm` (3D shape) | 超大域 reduce + affine |

## 难度统计

- **中等难度**：15 个 (每组 5 个)
- **困难难度**：9 个 (每组 3 个)

## 分组设计原则

1. **难度均衡**：每组 5 中 + 3 难，避免某组偏易或偏难
2. **模式多样**：每组覆盖 reduction、normalization、distance/similarity 等不同计算模式
3. **张量维度覆盖**：每组包含从 2D 到 5D 不同维度的算子
4. **独立可测**：每组可作为独立测试批次运行

## Shape 设计

| 类型 | 典型 shape | 说明 |
|------|-----------|------|
| Transformer | `(32, 512, 768)` | 标准 LLM/BERT 中间层 |
| 4D CNN | `(32, 128, 64, 64)` | CNN 中间特征图 |
| 4D CNN (压力) | `(16, 64, 256, 256)` | BatchNorm2d 保留大 shape 做压力测试 |
| 5D Video | `(4, 64, 8, 32, 32)` | 视频/体积数据处理 |
| Matrix | `(32, 256, 256)` / `(16, 512, 512)` | 矩阵范数计算 |
| Vector | `(128, 16384)` / `(16, 16384)` | 向量范数 / 归一化 |

## 使用方式

```python
import importlib.util

spec = importlib.util.spec_from_file_location("mod", "group_1/01_LayerNorm.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.Model(*mod.get_init_inputs())
output = model(*mod.get_inputs())
```
