# UnifiedSketch 设计

## 1. 目标与原则

用最小 DSL 表达算子设计意图，便于 LLM 理解和 Coder 实现。

## 2. 核心语法元素

### 2.0 策略锚点 (Strategy Anchors) - 必须包含

```python
# Applied Meta-Strategies: [strat_id1, strat_id2, ...]
```

### 2.1 结构声明

```python
sketch <op_name> {
  symbols: M, N, K;                    # 符号变量声明
  tensors: A[M, K]: f16; B[K, N]: f16; C[M, N]: f32;  # 张量声明
  constexpr: m0, k0, n0
  ...
}
```

### 2.2 内存管理 IR：alloc 操作

```python
tile = alloc([shape], llm_hint=["存储要求", "用途说明", "性能要求"])
```

**llm_hint**
- **存储要求**: `"fastest"` (Register), `"fast"` (Shared/L1), `"medium"` (L2), `"slow"` (Global)
- **用途说明**: `"accumulator"`, `"input_cache"`, `"output_buffer"`, `"temp_workspace"`
- **初始化**: `"init_zero"`, `"init_neg_inf"`, `"no_init"`

### 2.3 数据流 IR：Load/Store 操作

```python
load(tensor[slice] -> tile, mask=mask_tile, llm_hint=["..."])
store(tile -> tensor[slice], mask=mask_tile, llm_hint=["..."])
```

**llm_hint**:
- **Parallel**: `"cooperative_load"`, `"cooperative_store"`
- **Vectorized**: `"vectorized"` (连续), `"strided"`, `"broadcast"`
- **Boundary**: `"mask_boundary"`, `"assume_aligned"`
- **Access**: `"block_ptr"` (强制使用块指针优化，需满足线性索引条件), `"atomic_add"` (原子加)

### 2.4 计算 IR

- **基础**: `add`, `sub`, `mul`, `div`, `exp`, `log`, `sqrt`, `max`, `min`, `clamp`, `where`, `abs`
- **线性代数**: `gemm(a,b,dst)`, `dot`, `outer_product`
- **归约**: `reduce_sum(src, axis)`, `reduce_max`, `reduce_min`, `reduce_argmax`
- **扫描**: `local_cumsum`, `local_cumprod`
- **复合**: `softmax`, `relu`, `gelu`, `sigmoid`

### 2.5 @llm_hint & @metaprompt 装饰器

```python
@llm_hint("parallel", "grididx.x/y/z")  # 映射到 Grid 维度
@llm_hint("pipeline")                   # 启用软件流水线
@llm_hint("vectorize")                  # 强制向量化
@llm_hint("unroll")                     # 循环展开
@metaprompt("strat_id")                 # 标注此处代码实现了哪个元提示策略 (如 strat_tiling_2d_block_ptr)
```


## 3. For 循环 IR 示例

```python
# GPU 风格：Grid 并行
@llm_hint("parallel", "grididx.x")
for i in range(0, M, BM):
    @llm_hint("parallel", "grididx.y") 
    for j in range(0, N, BN):               
        a_tile = alloc([BM, BK], llm_hint=["fast", "input_cache"])
        b_tile = alloc([BK, BN], llm_hint=["fast", "input_cache"])
        acc = alloc([BM, BN], llm_hint=["fast", "accumulator", "init_zero"])
        @llm_hint("pipeline")
        for k in range(0, K, BK):
            # 根据tensor_a, tensor_b 维度确定，以2维为例
            a/b/c_i/j/k_idx 是根据 i j k 索引计算得到的
            mask_a/b/ctile 是根据 a/b/c_i/j/k_idx 的实际范围得到的
            load(tensor_a[a_i_idx, a_k_idx] -> a_tile, mask=mask_atile, llm_hint=["..."])
            load(tensor_b[b_k_idx, b_j_idx] -> b_tile, mask=mask_btile, llm_hint=["..."])
            gemm(a_tile, b_tile, acc)

        mask_store 是根据 c_i_idx, c_j_idx 实际范围得到的
        store(acc -> tensor_c[c_i_idx, c_j_idx], mask=mask_store, llm_hint=["..."])
```

## 4. UnifiedSketch 规范

1. **判断逻辑**：在IR中尽量不要使用判断逻辑，需要判断的地方尽量使用mask实现
2. **坐标逻辑**：索引不要使用...缩写，索引展开为一个个 idx 构成的 slice。同时每个 idx 的边界条件通过 & 的方式添加到 mask 中
3. **mask**：利用 mask 取代 if 判断放在 load/store 函数中

## 基础算子设计提示

1.  循环不变量（如 Norm 均值、Conv 坐标判定）移出最内层循环。
2.  **Conv3d 输出大小（其他维度同理）**：
    - $D_{out} = \lfloor \frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\rfloor$
    - $H_{out} = \lfloor \frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\rfloor$
    - $W_{out} = \lfloor \frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\rfloor$
3. **Conv3d Implicit Gemm 维度（考虑group G）**：
    - M 维度：$N\times D\times  P \times Q = \text{batch size} \times D_{out} \times H_{out}\times W_{out}$
    - N 维度：$C_{out}/G$
    - K 维度：$R\times S \times T \times (C_{in}/G) = \text{kernel\_size[0]} \times \text{kernel\_size[1]} \times \text{kernel\_size[2]} \times (C_{in}/G) $

4. **Conv3d 中根据其他坐标计算输入坐标**：
    - $d_{in} = d_{out} \times \text{stride}_d + r \times  \text{dilation}_d - \text{padding}_d$
    - $h_{in} = h_{out} \times \text{stride}_h + s \times  \text{dilation}_h - \text{padding}_h$
    - $w_{in} = w_{out} \times \text{stride}_w + t \times  \text{dilation}_w - \text{padding}_w$

5. **ConvTrans3d 输出大小（其他维度同理）**：
    - $D_{out} = (D_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding[0]} + \text{dilation[0]} \times (\text{kernel\_size[0]} - 1) + \text{output\_padding[0]} + 1$
    - $H_{out} = (H_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding[1]} + \text{dilation[1]} \times (\text{kernel\_size[1]} - 1) + \text{output\_padding[1]} + 1$
    - $W_{out} = (W_{in} - 1) \times \text{stride}[2] - 2 \times \text{padding[2]} + \text{dilation[2]} \times (\text{kernel\_size[2]} - 1) + \text{output\_padding[2]} + 1$

6. **ConvTrans3d Implicit Gemm 维度（group单独考虑）**：
    - M 维度：$N\times D_{out} \times H_{out} \times W_{out}$
    - N 维度：$C_{out}/G$
    - K 维度：$R\times S \times T \times (C_{in} / G)$

7. **ConvTrans3d 根据其他坐标计算输入坐标(r,s,t是卷积核维度三维大小)**：
    - $d_{in} = \frac{d_{out} + \text{pad}_d - r \times \text{dilation}_d}{\text{stride}_d}$
    - $h_{in} = \frac{h_{out} + \text{pad}_h - s \times \text{dilation}_h}{\text{stride}_h}$
    - $w_{in} = \frac{w_{out} + \text{pad}_w - t \times \text{dilation}_w}{\text{stride}_w}$

## 5. 硬件 specific 优化

- **GPU (NVIDIA/AMD)**:
  - `llm_hint="fast"` -> Shared Memory
  - `llm_hint="fastest"` -> Register File
  - Grid Mapping: 映射 `grididx.x/y/z` 到 Grid Block。
- **NPU (Ascend)**:
  - `llm_hint="fast"` -> L1 Buffer / Unified Buffer
  - `llm_hint="coreidx"` -> AI Core 并行

## 7. 性能禁忌

1.  **内循环标量化**：禁止单标量 Load，必须 `arange` 向量化。
4.  **Alloc 滥用**：严禁分配大面积工作空间用于 Input Cache，导致 Register Spilling。
6.  **标量线程映射**：严禁 One-thread-per-pixel 映射，必须 Block-based。
7.  **手书归约逻辑**：严禁手写 Tree-reduction，使用 `reduce_sum`。
8.  **多步扫描同步**：严禁使用全局多步同步扫描。
9.  **Tile 内标量循环**：严禁 `for i in range(BLOCK_SIZE)`。
10. **二阶段串行归约**：严禁 "Partial Sum -> Global Sum" 的二阶段写法，使用 Atomic。
11. **手动线程 ID**：严禁使用 `thread_idx` 或 Grid-Stride Loop。
12. **手动 Shared Memory**：严禁手动管理 SMEM 进行归约。
13. **过度分配中间缓冲**：严禁为每个算术步骤 alloc tile，使用 Math Fusion (直接表达式)。
14. **损失函数多 Pass**：严禁将 Loss 拆分为“写回 + 求和”，必须 Atomic。
15. **过度 Tiling**：在 Grid 并行度充足时 (如 B*C 很大)，严禁在 Block 内部对 Batch/Instance 维度再次 Tiling (如 `for bc in range(0, B*C, BC)`)，应直接映射 `range(0, B*C, 1)` 到 Grid (Step=1)。

