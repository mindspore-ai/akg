---
name: triton-ascend-case-elemwise-broadcast-2d
description: "2D广播除法优化：小维度完整处理不切分（循环外加载复用），通过固定NUM_BLOCKS实现核间并行（40核），核内SUB_M控制粒度平衡UB利用率，适用于broadcast轴大但非broadcast轴小的2D场景"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  hardware: "Atlas A2, Atlas A3"
---

# 2D Broadcast Division 优化案例

## 任务特征（两种配置）

### 配置1：(131072, 16) / (1, 16)
- broadcast第一根轴
- broadcast轴shape大(131072)，非broadcast轴shape小(16)

### 配置2：(2048, 131072) / (2048, 1)
- broadcast第二根轴
- broadcast轴中等(2048)，非broadcast轴shape大(131072)

## 优化 1：小维度完整处理不切分

```python
# N维度较小(N=16)，完整处理不切分
offs_n = tl.arange(0, N)  # N=16

# divisor所有行共享，在循环外加载一次
divisor = tl.load(divisor_ptr + offs_n)  # shape: (N,)

# 内层循环：每次处理SUB_M行
for sub_start in range(row_start, row_end, SUB_M):
    offs_m = sub_start + tl.arange(0, SUB_M)
    dividend = tl.load(dividend_ptr + dividend_offs, mask=mask_2d, other=0.0)
    output = dividend / divisor  # divisor广播: (N,) -> (SUB_M, N)
```

### 优化内容
- 由于N维度较小(N=16)，选择完整处理不切分，最大化UB利用率
- divisor所有行共享，在循环外加载一次，循环内自动广播复用
- 维度大小决定是否切分，而非广播方向

## 优化 2：Grid切分配置

```python
# NUM_BLOCKS控制核数，SUB_M控制内部每次处理行数
triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 512}), # 8.55us，最优，核数=40，用满物理核
triton.Config({'NUM_BLOCKS': 64, 'SUB_M': 512}), # 9.83us，核数>40，调度开销大
triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 256}), # 9.78us，ub未用满
triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 1024}), # 超ub
grid = lambda meta: (meta['NUM_BLOCKS'],)
```

### 优化内容
- 通过grid切分M维度，控制核数≤40
- SUB_M=512时在UB利用率和寄存器压力之间达到平衡

## 优化 3：通用2D调度方法

对于一般的2D shape，通用调度方法：
1. **核间并行（NUM_BLOCKS）**：沿M维度切分，分配到不同计算核
2. **核内行切分（SUB_M）**：控制每次处理的行数，平衡UB利用率
3. **列向量化（BLOCK_N）**：沿N维度分块加载，实现连续访问和向量化

### 总结
1. 对于较小的维度，应完整处理不切分以最大化UB利用率
2. 通过固定NUM_BLOCKS实现核间并行，核内参数切分控制数据粒度
3. 可通过autotune参数进行调优
