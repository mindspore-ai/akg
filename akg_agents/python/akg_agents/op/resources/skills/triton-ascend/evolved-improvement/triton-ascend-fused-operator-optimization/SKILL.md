---
name: triton-ascend-fused-operator-optimization
description: Ascend NPU 上融合算子的深度优化方法论。覆盖性能天花板分析框架、多 Pass 合并策略、数据访问模式重构、Normalization 两阶段决策、NPU 原生算子评估方法论。适用于 elementwise 融合、归一化融合、softmax+topk 融合、matmul+activation 融合等场景。
category: improvement
version: "1.0.0"
metadata:
  case_type: improvement
  backend: ascend
  dsl: triton_ascend
---

# 融合算子深度优化方法论

## 优化前：性能天花板分析框架

优化前先分析算子的**瓶颈类型**，选择正确的优化方向，避免在物理极限附近浪费时间：

| 瓶颈类型 | 判断方法 | 优化方向 | 典型天花板 |
|---------|---------|---------|----------|
| 内存带宽受限 | 计算量少、数据搬运多 | 减少 HBM 读写次数 | ~1.5-2x |
| 多次遍历 | 同一数据被读取 3+ 次 | 多 pass 合并为单 pass | ~3-4x |
| 数据访问模式 | 非连续/strided 访问 | 重构为连续访问 | ~5-20x |
| 计算主导 | matmul weight 矩阵大 | 融合几乎无效 | ~1.0x |

### 天花板计算方法

**理论加速比 = baseline 总 HBM 访问量 / 优化后总 HBM 访问量**

示例：对于 `y = f(x) * z` 类融合：
- Baseline（2 个 PyTorch op）：读 x → 写 f(x) → 读 f(x) + z → 写 y = **4 次**
- Triton 融合：读 x + z → 写 y = **2 次**
- 理论上限 = 4/2 = 2x

**实际天花板更低**的原因：baseline 中间 tensor 常命中 L2 cache，等效减少了 HBM 访问次数。

### Matmul 主导型融合的判断

## 优化方法 1：多 Pass 合并

### 适用条件
算子对同一数据进行多次独立遍历（如 softmax 的 max→exp_sum→normalize，或 topk 的多次扫描）。

### 方法
将所有 pass 合并为单次遍历，在寄存器内完成全部计算：

```python
# 次优：多次遍历
max_val = pass_find_max(data)          # 遍历 1
exp_sum = pass_compute_exp(data)       # 遍历 2
topk = pass_find_topk(data)            # 遍历 3+

# 推荐：单次遍历
data = tl.load(...)
max_val = tl.max(data, axis=0)
exp_vals = tl.math.exp(data - max_val)
exp_sum = tl.sum(exp_vals, axis=0)
probs = exp_vals / exp_sum
# topk 直接在同一 block 内完成
first_val = tl.max(probs, axis=0)
```

### 关键约束
当归约维度能放入单个 BLOCK 时效果最佳；维度过大则需分块归约，收益递减。

## 优化方法 2：数据访问模式重构

### 适用条件
算子涉及 strided / 非连续访问模式（如需要访问相邻元素的配对计算）。

### 方法
从按元素展平处理，改为按语义分组处理，使相关元素落在同一 block 内：

```python
# 次优：展平后按 flat_idx 处理，需跨 stride 访问配对元素
for block_id in range(pid, total_elements // BLOCK_SIZE, CORE_NUM):
    flat_idx = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    d = flat_idx % D
    d_pair = d ^ 1  # 跨 stride 随机访问

# 推荐：按语义维度分组，内层连续加载
for group_idx in range(pid, total_groups, CORE_NUM):
    # 计算分组坐标
    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        # 配对元素天然在同一 block 内
        vals = tl.load(ptr + base + d_offsets)
```

### 为什么有效
Ascend 硬件对非连续访问有显著性能惩罚。重构后连续加载减少 gather 操作，性能差距可达数倍至数十倍。

## 优化方法 3：Normalization 两阶段决策

### 适用条件
LayerNorm / RMSNorm / GroupNorm 等需要先统计再归一化的算子。

### 结论：两阶段（2-pass）优于单 Pass

| 方案 | 优点 | 缺点 |
|------|------|------|
| 2-pass（统计→归一化） | 编译器流水效率高，UB 压力可控 | 数据遍历两次 |
| 单 pass（在线统计） | 数据仅遍历一次 | 循环体活跃 tensor 多，UB 压力大，编译器流水优化受限 |

### 原因分析
单 pass 虽然减少一次遍历，但循环体内同时保持 mean/var 累加器和原始数据指针，导致：
- UB 中活跃 tensor 增多，可用空间减少
- 编译器对复杂循环体的多级流水优化效率降低
- 实际带宽利用率反而下降

### 推荐写法

```python
# Pass 1: 统计量
mean_acc = 0.0
var_acc = 0.0
for n_start in range(0, N, BLOCK_SIZE_N):
    data = tl.load(...)
    mean_acc += tl.sum(data, axis=0)
    var_acc += tl.sum(data * data, axis=0)
mean_val = mean_acc / N
std_val = tl.sqrt(var_acc / N - mean_val * mean_val + eps)

# Pass 2: 归一化
for n_start in range(0, N, BLOCK_SIZE_N):
    data = tl.load(...)  # 重新加载
    normalized = (data - mean_val) / std_val
    tl.store(out_ptr + ..., normalized * weight + bias)
```

## 优化方法 4：BLOCK_SIZE 调优策略

### Elementwise 类

| 场景 | 推荐 BLOCK_SIZE | 原因 |
|------|----------------|------|
| 数据量大（>1M 元素） | 4096 | 充分利用 UB，减少循环开销 |
| 数据量小（<100K 元素） | 1024 | 减少 UB 压力 |
| 中间变量多（>4 个活跃 tensor） | 2048 | 防止 UB overflow |

### 使用 Autotune 自动选优

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['n_elements'],
)
```

## NPU 原生算子（torch_npu）评估方法论

在手写 Triton kernel 前，应评估 NPU 原生融合算子，但需注意常见陷阱：

### 评估 Checklist

- [ ] **输入格式要求**：是否要求特定的连续 tensor 布局？数据拼接/重排的开销是否抵消收益？
- [ ] **精度一致性**：内部累积顺序、舍入模式是否与 baseline 一致？需实测 diff 值
- [ ] **编译开销**：首次调用是否有较大的编译延迟？

### 常见问题模式

| 问题 | 表现 | 解决方案 |
|------|------|---------|
| 输入布局不匹配 | 需要 concat/reshape 预处理，copy 开销 > kernel 收益 | 仅当输入天然满足格式要求时使用 |
| 精度不一致 | 验证 diff 超限（常见于 fp16） | 改用分步 PyTorch 操作 |
| approximate 模式差异 | 激活函数精度偏差大 | 手动实现精确公式 |

### torch.compile / torch.jit.script

在 Ascend NPU 上，对简单 elementwise 融合**通常无显著收益**：
- 编译开销大，稳态性能与手动 fusion 接近
- 图优化受 NPU 后端支持程度限制

## 不要过度融合

将所有操作塞入一个 kernel 可能导致：
- UB 溢出 → 被迫缩小 BLOCK_SIZE → 整体变慢
- 编译器流水线优化失效 → 实际带宽利用率下降
- 反而比拆分为 2-3 个简单 kernel 更慢

**判断标准**：当 kernel 内活跃 tensor 数 × BLOCK_SIZE × sizeof(dtype) × multi_buffer 系数（2~3）接近 UB 容量时，应考虑拆分。
