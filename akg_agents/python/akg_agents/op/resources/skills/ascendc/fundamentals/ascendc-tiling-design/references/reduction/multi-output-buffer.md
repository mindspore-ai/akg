# 多输出归约的 Buffer 规划

> 许多归约算子需要同一遍扫描输出多个结果，比单输出多占 UB。本文给出通用 Buffer 方程。

---

## 常见多输出场景

| 算子 | 输出数 | 输出内容 | 累加器 dtype |
|------|--------|---------|-------------|
| bn_training_reduce | 2 | sum[C] + squareSum[C] | FP32 |
| reduce_var / reduce_std | 2 | mean[A] + var[A] | FP32 |
| reduce_std_with_mean | 2 | std[A] + mean[A] | FP32 |
| arg_max / arg_min | 2 | value[A] + index[A] | FP32 + INT32/FP32 |
| arg_max_with_value | 2 | value[A] + index[A] | 同上 |

## 通用 UB 方程

```
输入:
  T_in  = 输入元素大小 (FP16=2, FP32=4)
  T_acc = 累加器元素大小 (通常 FP32=4)
  K     = 输出个数（双输出 K=2，三输出 K=3）
  A_aligned = 保留轴对齐后大小

Buffer 清单:
  inBuf × 2              = tileSize × T_in × 2          ← 输入双缓冲
  castBuf (仅低精度)      = tileSize × T_acc             ← FP16/BF16 → FP32
  accumBuf × K           = A_aligned × T_acc × K         ← K 个累加器
  tmpBuf                 = A_aligned × T_acc             ← 中间计算（如 x²）
  outBuf × 2             = A_aligned × T_acc × 2         ← 输出双缓冲（可选）

UB 方程:
  tileSize × (T_in × 2 + T_acc)              ← 输入 + Cast
  + A_aligned × T_acc × (K + 1 + 2)          ← K 累加器 + tmp + out双缓冲
  ≤ UB_SIZE (DAV_2201: 192KB)

求解 tileSize:
  fixedBuf = A_aligned × T_acc × (K + 3)
  perTileBuf = T_in × 2 + T_acc
  tileSize = (UB_SIZE - fixedBuf) / perTileBuf
```

## 双输出示例：bn_training_reduce (K=2, FP16 输入)

```
A_aligned = CeilAlign(C, 8)        ← FP32 按 32B 对齐: 32/4=8
T_in = 2 (FP16), T_acc = 4 (FP32)

fixedBuf = A_aligned × 4 × (2 + 3) = A_aligned × 20
  分解: sumBuf(A×4) + sqSumBuf(A×4) + tmpBuf(A×4) + outBuf×2(A×4×2)

perTileBuf = 2 × 2 + 4 = 8   (FP16 双缓冲 + FP32 Cast)

tileSize = (192KB - A_aligned × 20) / 8

示例: C=64 → A_aligned=64
  fixedBuf = 64 × 20 = 1280B
  tileSize = (196608 - 1280) / 8 = 24416 元素
  实际取 tileRows = tileSize / A_aligned = 381 行
```

## 双输出示例：ArgMax (K=2, value + index)

```
A_aligned = CeilAlign(A, 8)

累加器:
  maxValBuf: A_aligned × 4 (FP32)      ← 当前最大值
  maxIdxBuf: A_aligned × 4 (FP32)      ← 当前最大值的下标（存为 float）

额外:
  cmpBuf: A_aligned / 8 (uint8_t)      ← Compare mask
  注意: DAV_2201 上 Select 不支持 int32 dst → 下标存为 float，最后 Cast 为 int32

fixedBuf = A_aligned × 4 × 2 + A_aligned × 4 + max(A_aligned/8, 32) + outBuf
```

## 跨核合并时 Workspace 方程

多输出时 workspace 也要乘 K：

```
workspace = coreNum × CeilAlign(A_aligned × T_acc × K, cacheLineSize)

示例: bn_training_reduce, C=256, 20核
  workspace = 20 × CeilAlign(256 × 4 × 2, 256) = 20 × 2048 = 40KB
```

---

## NCHW 布局的数据搬运注意事项

"保留 C 归约 NHW" 场景（bn_training_reduce）中，NCHW 布局的内存不是按通道连续的：

```
NCHW 内存布局: x[n,c,h,w] 地址 = n×C×H×W + c×H×W + h×W + w
  同一 (n,h,w) 的 C 个通道值步长 = H×W（不连续！）
  同一 (n,c) 的 H×W 个空间位置连续

两种处理方式:
  方式A（推荐，DAV_2201/DAV_3510 通用）: 按通道遍历，外层 (n, c)，内层连续搬 H×W
    → 每次搬运连续内存（高效），需 C 个独立累加器
    → 适合大多数场景

  方式B（仅 DAV_3510）: 用 NDDMA 多维搬运，一次配置自动处理 stride 跳跃
    → DAV_2201 不支持 NDDMA

  方式C（DAV_2201）: DataCopyPad 配置 stride 参数
    → blockCount=R 行数, blockLen=连续片段字节数, srcStride=跳跃步长
    → 适合有规律的 stride 模式

实际 batch_norm_v3 采用方式A: 按通道遍历，每通道累加 N×H×W 个连续值
  参见 /ascendc-api-best-practices 获取 DataCopyPad stride 配置详情
```
