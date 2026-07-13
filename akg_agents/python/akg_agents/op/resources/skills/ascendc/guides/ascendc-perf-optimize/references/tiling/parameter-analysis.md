# 参数空间与算法使能分析

> 候选枚举的前置步骤。不完成本分析则候选解集无效。

---

## 执行顺序

四个阶段串行执行，不可跳过或合并：

| 阶段 | 目标 | 门禁 |
|------|------|------|
| §1 | 确定 kernel 实际接收的 tiling 参数全集 | 清单非空 |
| §2 | 参数类型分类，约束来源追溯 | 每参数有分类 |
| §3 | 判定当前 shape/dtype 下可用的算法 | ≥1 个算法命中 |
| §4 | 构建各命中算法的候选解空间 | 每算法有生成规则 |

产出写入 `{工作目录}/参数空间分析.md`。

---

## §1 Kernel 参数全集

1. 通读 demo 代码，找到 host 向 kernel 传递 tiling 参数的数据结构，列出所有字段
2. 区分：哪些字段传递给了 kernel，哪些只在 host 侧使用。后者不是调优维度，排除
3. 分析 kernel 内部如何使用这些字段，提取隐含约束（分支条件、范围检查等）

输出：参数清单 + 每参数在 kernel 中的用途和约束。

---

## §2 参数分类与约束追溯

对 §1 中每个参数做两件事：

### 类型判定

| 类型 | 判定 | 搜索行为 |
|------|------|---------|
| 固定输入 | 直接来自用户指定的问题规模 | 不搜索 |
| 独立可调 | 存在搜索/选择逻辑，有多合法值 | 搜索维度 |
| 派生 | 由其他参数公式计算 | 随独立变量自动计算 |

### 约束来源追溯

对每个参数的每个取值范围限制，追溯定义来源：

- 来自芯片规格配置（buffer 容量、核数、对齐粒度）→ **硬件约束**，不可放松
- 来自硬编码字面量 → **软件经验值**，可纳入搜索

判定方法：沿赋值链追溯到原始定义——是平台配置字段还是常数。

输出：分类表 + 约束来源表 + 被软件截断的参数列表（标注截断值和硬件上限）。

---

## §3 算法使能判定

加载该算子族的 tiling-flow.md 中的算法决策树，逐算法检查门禁条件。对每个命中的算法，提取其与默认基线的参数差异（额外约束、固定常量、可调参数增减）。

输出：`算法 → 命中/未命中（原因） + 可调参数 + 特殊约束`。

### 样本分桶

多 shape / 多 dtype 算子不能只构造一个全局搜索空间。先把样本按语义分桶，再为每个桶分析算法使能：

| 分桶维度 | 例子 | 影响的候选 |
|---|---|---|
| dtype | fp32、fp16、bf16、int8、int64 | tile 字节数、是否需要 Cast scratch、是否能 native 计算 |
| rank/layout | rank2 contiguous、rank5 contiguous、非 contiguous | 是否可 flatten、是否需要 index mapping |
| broadcast | same-shape、scalar、last-dim、general | 是否走 broadcast 快路径 |
| reduction axis | last-dim、小 D、非连续轴、大 D | 标量规约、单 tile 规约、二阶段规约 |
| index pattern | 连续段、dim0/dim1、随机 index | DataCopy 块化或 scalar fallback |
| 特殊语义 | all-zero、all-NaN、single segment、identity reduce | fill/copy/sum 快路径 |

输出需要包含每个桶的样本数量、总耗时占比和最慢样本。候选空间优先覆盖“总耗时占比高”的桶，而不是只覆盖样本数量最多的桶。

示例：

```text
bucket large_same_shape_int8:
  samples: 1
  total time share: 42%
  enabled algorithms:
    - native integer vector path
    - framework bypass upper-bound
  tunables:
    - tileLength: 4096, 8192, 12288
    - bufferNum: 1, 2

bucket last_dim_broadcast_half:
  samples: 3
  total time share: 12%
  enabled algorithms:
    - row-wise broadcast reuse
    - generic index mapping
  tunables:
    - rowsPerTile
    - broadcast cache length
```

---

## §4 候选解空间

对每个命中算法：

1. 独立可调参数取其**硬件约束下的完整范围**，步长为硬件对齐粒度，不受软件截断值限制
2. 派生参数按 §2 公式计算。若存在软件截断，同时保留截断值和硬件上限值作为候选维度
3. 所有候选用硬件约束做最终校验，不满足则剔除

输出：每算法的独立变量范围、硬件约束、派生公式、扩展维度、预计候选数。

候选空间不要只包含数字参数，也应包含结构候选：

| 结构候选 | 何时纳入 |
|---|---|
| queue depth 1/2/3 | MTE 与 VEC 未重叠，且 UB 账本允许 |
| scalar small-D path | 小规约同步成本明显 |
| bulk CopyOut | 每行/每元素小写回较多 |
| native dtype path | Cast 占比高且精度允许 |
| direct copy/fill path | special value 或 identity 语义明确 |
| host dispatch bypass | 通用 kernel 重写风险高，某语义桶明显退化 |

每个结构候选都应写清楚启用条件和回退路径，不能只给一个开关名。
