# Triton 常见错误补充

## tl.load/tl.store
### 描述
`tl.load(pointer, ...)/tl.store(pointer, ...)` 等指针相关的函数，需要传入指针类型。
`x_ptr = x.data_ptr()`得到的是一个整数类型，如果直接传给`tl.load/tl.store`，将会编译错误，例如:
```python
Unsupported ptrtype <['512'],int32> in tl.load
```

此外，**严禁使用 `tl.where` 代替 `tl.load` 的 `mask` 参数**。
例如：`tl.where(mask, tl.load(ptr), 0.0)` 是极度危险的。因为 `tl.load` 会在 `tl.where` 执行前对所有线程评估指针，如果指针指向非法地址（常见于卷积 Padding 处的负偏移），会直接导致 `illegal memory access` 报错，且该报错往往会被异步报告在后续的 Torch 代码中（导致误以为是 Torch 的 bug）。

### 修复方法
1. 不要使用 `data_ptr()` 获取指针，直接将 tensor 传入 kernel。
```
kernel[grid](x,...)
```
2. 始终使用 `tl.load` 的 `mask` 参数进行边界检查。
```python
# 错误：会触发 illegal memory access
val = tl.where(mask, tl.load(ptr), 0.0)

# 正确：安全加载
val = tl.load(ptr, mask=mask, other=0.0)
```

## 非法内存访问 (Illegal Memory Access) 深度排查

### 异步报错陷阱
由于 CUDA 异步执行，Triton 内核的越界错误常被延迟报告在后续的 Torch 代码中。
**修复建议**：
1. 检查 `tl.load/tl.store` 是否存在没有 `mask` 的裸访问。

### 隐式 GEMM 的二维掩码对齐 (The 2D Masking Rule)
在卷积类算子的隐式 GEMM 实现中，输入索引通常是 $M$ 轴（空间）和 $K$ 轴（权重）的函数：$InputIndex = f(M, K)$。
**修复建议**：
1. **坐标 2D 化**：必须使用 `M_idx[:, None]` 和 `K_idx[None, :]` 构建 2D 坐标矩阵。
2. **掩码完整性**：掩码必须覆盖 $BM \times BK$ 的每一个点。严禁为了“简化”而使用 1D 向量掩码广播。
3. **关键场景**：转置卷积的整除判断 `% stride == 0` 和 Padding 处的负索引检查，必须全部通过 2D Mask 屏蔽。

* **错误**：`d_in = (d_out + pad - r) // stride` (得到 1D 向量) -> `tl.load(ptr, mask=d_in_mask[:, None])` (1D 广播导致 $K$ 轴检查缺失)。
* **正确**：`d_in = (d_out[:, None] + pad - r[None, :]) // stride` (得到 2D 矩阵) -> `tl.load(ptr, mask=d_in_mask_2d)`。


## 报错在torch代码中
### 描述
由于 CUDA 误差报告是异步的，这个错误往往会在下一个 CUDA API（即 Torch 标准代码）执行时被报告，导致误以为是 Torch 的问题
### 修复方法
不要认为是Torch的问题，问题就是生成的代码的问题。

## 索引访问张量
### 描述
triton中不能通过`[index]`访问tensor元素。`tl.tensor`不是数组，而是 SIMD 向量 / 矢量寄存器的抽象。
### 修复方法
将`for`循环+逐元素访问，改为使用`mask`的向量化操作。

## 精度问题/输出不一致
### 描述
出现不是编译错误，而是输出不一致导致的问题。
### 修复方法
优先分析以下几个方面：
1. weight等参数tensor是否是先在host端创建，后传递到device端。
2. 参数shape是否与算子描述的相同。
3. tensor的dtype（精度）是否与算子描述的相同。
4. 关键累加步骤使用 `float32`；必要时显式转换类型。
5. tensor的索引计算是否超出了维度范围。
6. mask的shape是否正确，是否屏蔽了正确的维度。
7. **Reshape/View 布局陷阱 (The Reshape/View Layout Trap)**：
    *   **核心准则**: 在对 Tensor 执行 `view` 或 `reshape` 操作前，必须确保 **“逻辑上的平坦化顺序”** 与 **“物理存储顺序”** 完全一致。
    *   **现象**: 如果忽略此点，相对误差可能高达 $10^7$ 或出现规律性的数值偏移。
    *   **修正**: 绝大多数情况下（如 1x1 卷积转 GEMM 或转置卷积权重转换），必须先通过 `permute` 调整维度顺序，再执行 `reshape`。例如：`x.permute(0, 2, 3, 1).reshape(-1, C_in)`。直接 `view` 会因 NCHW 的内存不连续性导致通道数据被错误混合。
8. **标量思维导致的“逻辑缺失” (Scalar Thinking Trap)**：
    *   **现象**: 相对误差接近 1.0 且大量输出为 0。
    *   **原因**: Grid 规模是以 `BLOCK_SIZE` 为单位，内核却只用 `pid` 处理单一标量，导致 99% 的 Block 位点未被覆盖。
    *   **修正**: 始终使用 `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` 向量化偏移。

## 张量维度对齐与掩码逻辑 (Tensor Dimension Alignment)
### 描述
在 Triton 中，最隐蔽且致命的错误是**逻辑轴（Logical Axis）与物理轴（Physical Axis）的错位**。这不仅会导致索引越界，更会导致掩码（Mask）通过广播机制作用在错误的维度上，从而产生大规模且具有规律性的数值错误（典型的“数值幻觉”）。

### 修复方法
**索引计算和掩码形状必须严格遵循目标 Tensor 的物理维度范围（Physical Shape）。**

1.  **分量合法性**: 无论全局逻辑偏移（Global Offset）如何计算，最终作用于特定物理张量的每一个索引分量，必须落入该张量 `shape` 定义的范围内。
2.  **掩码维度匹配 (Mask Dimension Matching)**: 
    *   **核心逻辑**: 掩码必须在它试图“屏蔽”的那一维上具有有效长度（即非 $1$ 长度）。
    *   **广播陷阱**: 如果掩码形状选错（例如本该是列向量却用了行向量），Triton 会通过广播机制将掩码在错误的轴上复制。
    *   **后果**: 掩码失效或屏蔽了错误的区域。例如，想要屏蔽无效的行（$K$ 轴），却误用了行向量掩码，导致结果变成“只有部分列有效”，而无效的行却被非法保留并参与了累加。

3.  **局部 vs 全局映射**: 
    在分组计算或隐式 GEMM 塌陷维度（如 $M = N \times H \times W$）中，必须根据目标张量的物理存储方式，决定使用**相对索引**（局部偏移）还是**全局映射索引**。

## 指针与偏移类型不匹配 (Pointer & Offset Type Mismatch)
### 描述
Triton 的指针加法要求偏移量必须为整数。如果使用浮点数除法计算索引（即使最后结果是整数值），也会触发 `IncompatibleTypeErrorImpl`。

### 修复方法
**避免在指针偏移计算中使用浮点逻辑。**
1.  使用 **整除符号** `//` 代替 `/`。
2.  避免使用 `tl.math.floor(x / y)`，应直接使用 `x // y`。
3.  必要时使用 `tl.cast(offsets, tl.int32)` 进行显式类型转换。


## tl.make_block_ptr
### 描述
`tl.make_block_ptr(base: tensor, shape, strides, offsets, block_shape, order)`，没有`mask`，`other`参数。
### 修复方法
传入正确的参数。

## tl.arange
### 描述
`tl.arange(start, end)`，传入的参数需要为`constexpr`。
### 修复方法
传入参数为常量/常量表达式。

## tl.reshape
### 描述
`tl.reshape(input, *shape, can_reorder=False)`，input为输入的tensor，shape是改变后的形状，需要保证改变前后tensor的元素总数保持一致。
### 修复方法
确保正确的shape，保证reshape前后tensor元素总数保持一致。

## tl.program_id
### 描述
`tl.program_id(axis)`，triton的axis只能为0，1，2。即triton是3D启动网格。
### 修复方法
确保正确的`axis`值。


## 其他方面
- **constexpr**: 仅限 Kernel 签名中的编译时常量，Host 端不可调用。
- **Ascend 特化**: 避免在 `tl.load/store` 中使用 `tl.where` 进行动态偏移计算，改用 Host 端静态逻辑。
- **内存**: 是否都有 `mask` 或 `boundary_check`？Stride 是否匹配张量布局？
- **控制流**: 是否误用了禁止的 Python 语法？
- **初始化**: Host 端初始化是否在 CPU 完成且精度与输入一致？
- **原子操作**: 涉及并发写入时是否使用了 `tl.atomic_add/max`？
- Triton 里不存在 **tl.static_local_array**/**tl.static_array** 这些 API
- Triton中不允许使用`R = S = T = kernel_size`之类的连等赋值。
- 过多的autotune配置，容易导致超时（Verification timed out）。

