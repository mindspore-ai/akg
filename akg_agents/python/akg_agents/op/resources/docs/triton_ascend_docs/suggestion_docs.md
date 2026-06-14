# Triton 专家技巧与优化建议

本文档提供 Triton 开发的技巧、性能优化和问题排查指南。

## 1. 特定算子优化

### matmul 算子

**注意**：合理的切分是提升matmul算子性能的关键。

#### Ascend后端切分优化：充分发挥带宽，算子行宽为512B的整数倍，且单次行数尽量大，以fp16和bf16为例：
- **A、B都不转置**：分块行宽分别为K0和N0，则M0=128，K0=256，N0=256
- **A不转置，B转置**：分块行宽都是K0，则K0=256，M0和N0影响较小
- **A、B都转置**：分块行宽分别为M0和K0，则M0=256，K0=256，N0=128
- **A转置，B不转置**：分块行宽分别为M0和N0，则左右矩阵均无法同时满足512B的整数倍，需根据实际情况调整

### Attention 算子

#### 标准Attention计算流程：
1. **QK^T计算**：`scores = Q @ K^T / sqrt(d_k)`，计算注意力分数
2. **Softmax归一化**：`attn_weights = softmax(scores)`，确保权重和为1
3. **加权求和**：`output = attn_weights @ V`，得到最终输出

#### Flash Attention优化策略：
- **分块计算**：将大矩阵分块处理，减少内存占用
- **在线Softmax**：使用增量式softmax算法，分块计算，维护全局最大值和归一化因子，避免存储完整注意力矩阵，具体逻辑如下：
```python
# 初始化全局统计量
m_i = -float("inf")  # 全局最大值
l_i = 0.0           # 全局exp和
acc = 0.0           # 输出累加器

# 分块处理
for start_n in range(0, seq_len, BLOCK_SIZE):
    # 1. 加载当前块的分数
    scores = tl.load(scores_ptr + start_n, mask=load_mask, other=-float("inf"))
    
    # 2. 更新全局最大值
    m_ij = tl.maximum(m_i, tl.max(scores, 0))
    
    # 3. 计算当前块的exp值（数值稳定化）
    scores = scores - m_ij
    p = tl.math.exp2(scores)
    
    # 4. 更新全局exp和
    l_ij = tl.sum(p, 0)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    
    # 5. 更新输出累加器
    acc = acc * alpha + p
    
    # 6. 更新全局最大值
    m_i = m_ij

# 最终归一化
acc = acc / l_i
```

## 2. 性能优化

### 块大小选择策略
- **调优**: 平衡并行度与资源占用，避免过大或过小

### 内存访问优化
- **2D数据**: 优先使用 `tl.make_block_ptr` 配合 `boundary_check`，自动优化内存合并
- **步幅设计**: 仔细设计stride参数，错误设置会严重影响性能
- **数据布局**: 保持内存访问的连续性和局部性

#### 连续内存的一维访问优化
张量在内存中连续存储时，可用一维指针遍历，避免多维索引开销：

```python
# 方案1：转连续后用一维访问（推荐）
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
    # 非连续张量转为连续（一次性开销）
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    
    output_tensor = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    elementwise_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE)
    return output_tensor

@triton.jit
def elementwise_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = tl.load(input_ptr + offsets, mask=mask)
    result = compute(data)
    tl.store(output_ptr + offsets, result, mask=mask)

# 方案2：使用stride访问（不推荐）
# 每次load/store都需计算stride偏移，性能较差
@triton.jit
def elementwise_kernel_stride(input_ptr, output_ptr, M, N, stride_m, stride_n, 
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 每个元素访问都有额外开销
    ...
```

**性能分析**：
- **`.contiguous()`转换**：一次性内存拷贝开销，后续访问高效（连续内存，缓存友好）
- **stride访问**：每次load/store都需计算偏移，累积开销大，内存访问不连续
- **建议**：非连续张量先调用`.contiguous()`转换，再用一维访问，整体性能更优

**要点**：
- 优先使用`.contiguous()`转换 + 一维访问
- 连续内存访问效率远高于stride计算开销
- `torch.empty_like()`创建的输出默认连续
- 输入输出同形状的element-wise算子无需reshape

### 算子拆分策略
- **复杂算子**: 拆分为多个简单kernel，避免单个kernel过于复杂

## 3. 数值稳定性技巧

### 防溢出处理
```python
# 归一化前先减去最大值
max_val = tl.max(data, axis=0)
stable_data = data - max_val
exp_data = tl.exp(stable_data)
```

### 防负值开方处理
- 在任何平方根操作前，确保被开方数是非负的（例如，使用 max(input, 0.)或 max(input, eps)）

### 精度提升
- **中间计算**: 关键步骤转为float32提升精度
- **累加操作**: 使用高精度累加器防止精度丢失

## 4. API使用限制与替代方案

### 禁止使用的语法
- 禁止 `return`, `break`, `continue` → 使用mask控制
- 禁止 lambda表达式 → 使用内联函数或tl.where
- 禁止 链式布尔运算 → 分步计算mask
- 禁止 张量直接索引 → 使用tl.load/tl.store
- 禁止 if-else分支中产生负偏移 → 使用mask分离加载，用`tl.maximum(offset, 0)`确保偏移非负
**Ascend后端**
- 复杂场景 `tl.where` → 使用if-else
- 禁止 `while` 循环 → 使用 for 替代（见下文）

### while 循环替代方案（Ascend后端）

Ascend后端不支持`while`循环，需根据循环上限是否为编译时常量选择替代方案。

**情况1：循环上限是静态值（编译时常量）**

直接用`for range`替代，无需额外处理：

```python
# ❌ 错误：while 循环
i = 0
while i < N_ITERS:  # N_ITERS 是编译时常量
    # 处理逻辑
    i += 1

# ✅ 正确：直接用 for range
for i in range(N_ITERS):  # N_ITERS: tl.constexpr
    # 处理逻辑
```

**情况2：循环上限是动态值（运行时参数）**

设置足够大的编译时常量作为循环上界，用`if`判断控制实际执行：

```python
# ❌ 错误：while 循环（n_iters 是运行时动态值）
@triton.jit
def kernel_while(ptr, n_iters, TILE: tl.constexpr):
    i = 0
    while i < n_iters:
        offset = i * TILE + tl.arange(0, TILE)
        data = tl.load(ptr + offset)
        tl.store(ptr + offset, data * 2)
        i += 1

# ✅ 正确：for + if 替代方案
@triton.jit
def kernel_for_if(
    ptr,
    n_iters,              # 运行时动态值
    TILE: tl.constexpr,
    MAX_ITERS: tl.constexpr,  # 编译时常量上界（需足够大）
):
    for i in range(MAX_ITERS):
        if i < n_iters:
            offset = i * TILE + tl.arange(0, TILE)
            data = tl.load(ptr + offset)
            tl.store(ptr + offset, data * 2)
```

**注意事项**：
- `MAX_ITERS` 需设置得足够大，覆盖所有可能的运行时值
- 当实际迭代次数远小于上界时，会有空循环迭代开销
- 上界设置过大会增加编译时间

### 切片操作规范
Triton不支持Python风格的直接切片语法（如`b[0]`或`b[i:j]`），需使用专用API：
- **单元素提取**：`tl.get_element(tensor, (index,))` 或单次load
- **切片提取**：`tl.extract_slice(tensor, offsets, sizes, strides)`
- **切片插入**：`tl.insert_slice(ful, sub, offsets, sizes, strides)` - 将sub张量插入到ful张量的指定位置

```python
# 一维切片插入
output_sub = x_sub + y_sub
output = tl.insert_slice(output, output_sub, [offset], [size], [1])

# 二维切片插入（逐行构建）
tmp_buf = tl.zeros((rows, cols), dtype)
val = tl.load(in_ptr + offset, mask)
tmp_buf = tl.insert_slice(tmp_buf, val[None,:], offsets=(i, 0), sizes=(1, cols), strides=(1, 1))
```

**重要限制**：禁止对`tl.arange`生成的张量使用`get_element()`
- `tl.arange`是编译时索引表达式，非实际张量，需直接计算而非提取
```python
# 错误：offsets = base + tl.arange(0, BLOCK_SIZE); value = tl.get_element(offsets, [i])
# 正确：value = base + i

# 正确用法
element = tl.get_element(tensor, (i, j))  # 实际张量
sub_tensor = tl.extract_slice(tensor, [0], [32], [1])  # 提取切片
```

### tl.constexpr 正确用法
- **仅在内核参数中使用**: `BLOCK_SIZE: tl.constexpr`
- **不可在host侧使用**: 启动函数中不可用tl.constexpr

### 输出张量创建规范
- host 侧使用 `torch.empty` 或 `torch.empty_like` 创建输出张量
- 不要使用 `torch.zeros` 或 `torch.ones`，避免不必要的初始化开销

### Ascend 后端避免使用 tl.where 计算内存偏移
Ascend 后端对`tl.where`生成的复杂指针运算支持不完全。复杂条件判断可以采用if-else静态分支处理，而非在内存访问时动态计算。

### 标量类型转换
- **仅支持to(type)**: 如`scalar.to(tl.float16)`，禁止使用`tl.float16(scalar)`
- **tl.constexpr类型转换**: 将常量赋值给临时变量再转换，如`scalar = CONST_A`

### 切分设置
**Ascend后端**
- BLOCK_SIZE/`tl.arange`对应的是单program内的tile大小，仍需受UB/L0和编译器限制约束；不要把它和grid总数混淆
- 若shape过大，单次切分后超过硬件缓存，并且BLOCK_SIZE超过限制，可以对循环进行多次切分

### Grid设置规范
- **维度限制**：grid必须是tuple类型，最多3维，如`(x,)`、`(x, y)`或`(x, y, z)`
- **大grid策略**：当前910B运行脚本默认设置`TRITON_ALL_BLOCKS_PARALLEL=1`，可以使用超过65535的**一维逻辑grid**。若逻辑grid来自2D/3D tile，请优先展平成1D，例如`grid=(num_m_tiles * num_n_tiles,)`，在kernel内用`pid // num_n_tiles`、`pid % num_n_tiles`反解；不要直接依赖超大的2D/3D grid。

### Grid与UB联合约束
- Ascend后端的主要资源风险是单program UB/L0容量。不要为了减少grid数量而盲目增大`BLOCK_SIZE`、`BLOCK_M`、`BLOCK_N`或`BLOCK_K`。
- 当逻辑tile很多时，优先保持小tile并使用1D flattened grid；若仍需限制物理并发，则使用固定核心数grid-stride循环或连续分块处理。每次实际处理的子块仍需满足UB/L0限制。
- 估算UB占用时，除输入/输出tile外，还要包含accumulator、mask、反解索引、cast/reshape/broadcast临时量，以及`tl.static_range`展开后同时存活的临时结果。
- 若编译或验证日志出现`ub overflow`、`requires ... bits while ... bits available`，说明单program live data过大；应减小tile、拆分sub-block、分片accumulator，或改成更小的K-block/reduction循环，而不是继续增大block来降低grid数量。

### Ascend调度选择优先级

Ascend Triton后端上，生成成功率优先级高于套用CUDA式implicit-GEMM。按下面顺序选择实现：

1. 纯elementwise或简单逐点计算：用连续tile或固定核心数grid-stride，保持每个program处理多个连续元素。
2. 规则Matmul、Linear、1x1 Conv：优先`tl.dot`，从小tile开始，确认UB/L0和grid约束后再扩大。
3. 普通Conv2d/Conv3d：只有反解input patch简单、`K_total`可分块、`BLOCK_M/BLOCK_N/BLOCK_K`不造成UB压力时使用`tl.dot`；否则用output-centric小tile加outer-product循环。
4. ConvTranspose：不要默认`tl.dot`。先判断shape和per-program work，优先选择能让单program live data稳定落入UB/L0的schedule。
5. 只有中小shape且性能不敏感时，才使用一输出点或一输出行的标量grid-stride fallback。大shape不要退化成每个输出元素长串行循环。

### ConvTranspose大shape策略

- 2D大shape首选有界output-centric tile：grid显式覆盖batch、H tile、W tile，例如`grid=(N, ceil(H_out/BLOCK_OH), ceil(W_out/BLOCK_OW))`。program内对`C_out`、`C_in`、`K_h`、`K_w`做小循环，`acc`只保留`(BLOCK_OH, BLOCK_OW)`，不要在单program内循环所有`W_out` block。
- 不要用`grid=(N,H_out)`后在单个program内循环所有`W_out` block；这会把过多空间位置、mask和临时反解索引同时放进UB。
- 2D权重布局是`(C_in, C_out, K_h, K_w)`。若使用output-centric tile，可在`__init__`中一次性沿空间维翻转并`contiguous()`，kernel中按翻转后的权重读取，减少每次反向索引。
- 2D若`C_out`很大、单program循环输出通道造成超时，改为把`C_out` block也纳入逻辑grid。
- 3D大shape或空间tile accumulator覆盖多个空间轴导致UB/编译超时，优先input-centric scatter-accum：先用独立zero kernel清零输出，再按每个kernel offset顺序launch scatter kernel。
- input-centric scatter-accum中，单个kernel offset内input到output是唯一映射，可避免`tl.atomic_add`；不同offset之间用host侧顺序launch做累加。不要在同一kernel中并发写同一输出位置。
- ConvTranspose反推masked load触发root-alloc/UB问题时，不要继续调大/调小`tl.dot` tile。2D先切到有界H/W output tile；3D或仍超资源时切到input-centric scatter-accum。

ConvTranspose output-centric模板要点：
- 2D stride=1/padding=0时，`ih = oh - kh`、`iw = ow - kw`；通用情况先算`num_h = oh + padding_h - kh * dilation_h`，满足`num_h % stride_h == 0`后`ih = num_h // stride_h`，W方向同理。
- input load mask同时包含stride整除、输入bounds、M/W tail；无效贡献用`other=0.0`。
- store地址必须包含block起点，不能只写相对offset：

```python
y_ptr = (
    y_base
    + (co_start + co_offsets)[:, None] * stride_y_c
    + (w_start + w_offsets)[None, :] * stride_y_w
)
store_mask = co_mask[:, None] & w_mask[None, :]
tl.store(y_ptr, acc, mask=store_mask)
```

### Ascend Triton索引限制
- 不要在`@triton.jit` kernel内写`tensor[0, :]`、`tensor[:, 0]`或`acc[i, :]`这类scalar+slice tensor indexing。Ascend Triton可能报`unsupported tensor index`；应直接构造一维tensor、保留广播维，或使用`tl.reshape`。

#### 大shape算子的Grid处理策略

对于输入shape较大的算子，直接按照`BLOCK_SIZE`切分得到的逻辑grid总数可能超过65535。当前910B运行脚本已开启`TRITON_ALL_BLOCKS_PARALLEL=1`，优先保留小tile并用一维逻辑grid或grid-stride覆盖所有逻辑块；不要通过放大tile来压低grid数量。有两种推荐方案：

**方案0：一维flattened逻辑grid（推荐，适用于原本想使用2D/3D tile grid的场景）**

将多维逻辑tile展平成一维grid，kernel内反解逻辑坐标。这样可以保留小tile，降低UB压力：

```python
num_m_tiles = triton.cdiv(M, BLOCK_M)
num_n_tiles = triton.cdiv(N, BLOCK_N)
grid = (num_m_tiles * num_n_tiles,)

@triton.jit
def kernel(..., NUM_N_TILES: tl.constexpr, ...):
    pid = tl.program_id(0)
    pid_m = pid // NUM_N_TILES
    pid_n = pid - pid_m * NUM_N_TILES
    # 继续按(pid_m, pid_n)处理一个小tile
```

不要直接把超大的`grid=(num_m_tiles, num_n_tiles)`交给`TRITON_ALL_BLOCKS_PARALLEL`，当前后端只对超大一维逻辑grid更可靠。

**方案1：交错循环处理（强烈推荐，适用于按行/按块独立处理的场景）**

将grid固定为核心数，每个核心以步长方式交错处理数据。这种方案代码最简洁，负载均衡最好：

```python
import torch_npu

# 示例：处理shape为(M, N)的张量，M可能非常大（如327680）
@triton.jit
def row_processing_kernel(
    input_ptr, output_ptr, 
    M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # 交错处理：每个核心处理 pid, pid+CORE_NUM, pid+2*CORE_NUM, ... 行
    # pid=0 处理第 0, CORE_NUM, 2*CORE_NUM, ... 行
    # pid=1 处理第 1, CORE_NUM+1, 2*CORE_NUM+1, ... 行
    # 这样所有行都会被恰好处理一次，负载均衡
    for row_idx in range(pid, M, CORE_NUM):
        # 计算当前行的指针偏移
        row_ptr = input_ptr + row_idx * stride_m
        out_row_ptr = output_ptr + row_idx * stride_m
        
        # 处理当前行的数据（可根据需要进一步分块）
        for col_start in range(0, N, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            mask = col_offsets < N
            
            data = tl.load(row_ptr + col_offsets * stride_n, mask=mask)
            result = compute_function(data)
            tl.store(out_row_ptr + col_offsets * stride_n, result, mask=mask)

# 启动方式
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 在初始化阶段获取核心数，向量计算类算子使用VEC核心数
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 48)
        except:
            self.VEC_CORE_NUM = 48  # Ascend 910B2 默认: 24个AI Core × 2个VEC/Core = 48

    def forward(self, input_tensor):
        M, N = input_tensor.shape
        output_tensor = torch.empty_like(input_tensor)
        
        grid = (self.VEC_CORE_NUM,)
        
        row_processing_kernel[grid](
            input_tensor, output_tensor,
            M, N,
            input_tensor.stride(0), input_tensor.stride(1),
            BLOCK_N=256,
            CORE_NUM=self.VEC_CORE_NUM,
        )
        return output_tensor
```

**动态获取核心数**：

根据算子类型选择对应的核心数，**必须在`__init__`中获取**（避免forward中重复调用导致同步开销）：
- **向量计算类算子**（element-wise、softmax、归一化等）：使用VEC核心数
- **矩阵计算类算子**（matmul、attention等）：使用CUBE核心数

```python
import torch_npu

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 在__init__中获取核心数，只执行一次，避免forward中的同步开销
        try:
            # 向量计算类算子使用VEC核心数
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
            # 矩阵计算类算子使用CUBE核心数
            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
        except:
            self.VEC_CORE_NUM = 40   # Ascend 910B4 默认: 20个AI Core × 2个VEC/Core = 40
            self.CUBE_CORE_NUM = 20  # Ascend 910B4 默认: 20个AI Core × 1个CUBE/Core = 20
```

**注意**：`torch_npu`的import和`get_device_limit`调用会触发设备同步，因此**禁止在forward中调用**，必须放在`__init__`初始化阶段。

**核心模式**：`for i in range(pid, total_items, core_num)` 是经典的并行工作分配模式。

**优点**：
- 代码极其简洁，一行for循环解决问题
- 负载天然均衡（每个核心处理的任务数差最多为1）
- 无需计算ELEMENTS_PER_GRID等复杂参数
- 适用于任意大小的输入shape

**适用场景及核心选择**：
- **VEC核心**：按行独立处理的算子（逐行softmax、逐行归一化、逐行reduce、element-wise等）
- **CUBE核心**：矩阵乘法类算子（matmul、attention等）
- 总任务数（行数/块数）远大于核心数的场景

**方案2：连续分块处理（适用于需要连续内存访问优化的场景）**

将grid设置得更小，让每个grid处理连续的一段数据。适用于对内存连续性有要求的场景：

```python
import torch_npu

# 示例：处理大shape的向量操作
@triton.jit
def large_vector_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_GRID: tl.constexpr,  # 每个grid负责的元素总数
):
    pid = tl.program_id(0)
    
    # 计算当前grid负责的数据范围
    grid_start = pid * ELEMENTS_PER_GRID
    grid_end = min(grid_start + ELEMENTS_PER_GRID, n_elements)
    
    # 分块处理当前grid负责的数据
    for block_start in range(grid_start, grid_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < grid_end
        
        # 加载、计算、存储
        data = tl.load(input_ptr + offsets, mask=mask)
        result = compute_function(data)
        tl.store(output_ptr + offsets, result, mask=mask)

# 启动方式
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 在__init__中获取核心数
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except:
            self.VEC_CORE_NUM = 40

    def forward(self, input_tensor):
        n_elements = input_tensor.numel()
        BLOCK_SIZE = *  # 设置为尽可能大的合适的每次处理的值，充分利用ub
        MAX_GRID_SIZE = self.VEC_CORE_NUM  # 使用初始化时获取的核心数
        
        # 计算每个grid需要处理的元素数
        ELEMENTS_PER_GRID = triton.cdiv(n_elements, MAX_GRID_SIZE)
        # 向上取整到BLOCK_SIZE的倍数，确保循环能完整处理
        ELEMENTS_PER_GRID = triton.cdiv(ELEMENTS_PER_GRID, BLOCK_SIZE) * BLOCK_SIZE
        
        # 计算实际需要的grid数量
        grid_size = triton.cdiv(n_elements, ELEMENTS_PER_GRID)
        
        output_tensor = torch.empty_like(input_tensor)
        large_vector_kernel[grid_size,](
            input_tensor, output_tensor, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            ELEMENTS_PER_GRID=ELEMENTS_PER_GRID,
        )
        return output_tensor
```

## 5. 调试与排查清单

### 内存访问问题
- [ ] 所有load/store是否都有mask或boundary_check？
- [ ] stride参数设置是否正确？
- [ ] 数组索引是否越界？

### 控制流检查
- [ ] 是否误用了return/break/continue？
- [ ] 复杂条件是否用mask组合实现？
- [ ] tl.constexpr是否只在内核参数中使用？

### Grid与Block配置检查
- [ ] 大shape是否使用1D flattened grid或grid-stride，而不是为了降低grid数量增大tile？
- [ ] 是否检查了单program的UB/L0占用，避免大`BLOCK_SIZE`/`BLOCK_M`/`BLOCK_N`/`BLOCK_K`导致`ub overflow`？
- [ ] 对于按行/按块独立的大shape算子，是否采用了交错循环`for i in range(pid, total, core_num)`处理？
- [ ] Grid维度是否为tuple类型且不超过3维？

### 并发与原子操作检查
- [ ] 并发写入是否使用了原子操作？
- [ ] 原子操作的数据类型是否匹配？

### 切片与索引检查
- [ ] 是否避免了Python风格的直接切片（如`b[0]`、`b[i:j]`）？
- [ ] 是否对`tl.arange`生成的张量误用了`tl.get_element`？
- [ ] 切片操作是否使用了正确的API（`tl.get_element`、`tl.extract_slice`等）？

### 性能优化检查
- [ ] 内存访问是否连续（避免跨步访问）？
- [ ] 是否充分利用了块内并行？
- [ ] 复杂算子是否考虑拆分为多个简单kernel？

## 6. 常见错误速查

| 错误类型 | 典型症状 | 常见原因 | 解决方案 |
|---------|---------|---------|---------|
| 内存越界访问 | 运行时错误、结果异常、随机崩溃 | load/store缺少mask或boundary_check | 添加正确的mask或boundary_check保护 |
| Grid超限 | 编译失败或运行时错误，日志包含`grid should be less than 65536` | 未开启`TRITON_ALL_BLOCKS_PARALLEL=1`或直接使用超大2D/3D grid | 确认运行环境已设置该变量；将多维逻辑grid展平成1D，或使用交错循环`for i in range(pid, total, core_num)` |
| 控制流错误 | 编译失败、语法错误 | 使用了return/break/continue | 移除禁用语句，使用mask控制流程 |
| while循环错误 | 编译失败（Ascend后端） | 使用了while循环 | 改用for + if替代：`for i in range(MAX): if i < n:` |
| 切片语法错误 | 编译失败 | 使用了`b[0]`或`b[i:j]`直接切片 | 使用`tl.get_element`或`tl.extract_slice` |
| tl.arange索引错误 | 编译失败 | 对`tl.arange`结果使用`get_element` | 直接计算索引值而非提取 |
| 类型转换错误 | 编译警告或错误 | 使用`tl.float16(scalar)`转换 | 改用`scalar.to(tl.float16)` |
| constexpr误用 | 编译失败 | 在host侧使用tl.constexpr | 仅在kernel参数中使用tl.constexpr |
| Stride设置错误 | 计算结果错误、数据错位 | stride参数计算或传递错误 | 验证stride设置，检查tensor.stride() |
| 数值不稳定 | 结果为NaN或Inf | softmax/sqrt等操作溢出 | 减去最大值、检查非负、使用float32 |
| 数据竞争 | 结果不确定、每次运行不同 | 多program并发写入同一位置 | 使用tl.atomic_add等原子操作 |
| BLOCK_SIZE过大 | 编译失败或运行时错误 | 单program内BLOCK_SIZE/tile超过UB/L0或编译器限制 | 减小BLOCK/SUB_BLOCK，使用循环处理 |
| UB溢出 | 编译或验证失败，日志包含`ub overflow`或`requires ... bits while ... bits available` | 单program内tile、accumulator、mask、broadcast临时量过大 | 减小BLOCK/SUB_BLOCK，拆分accumulator或K-block，避免大`static_range`展开 |
| root alloc分析失败 | 编译失败，日志包含`Unsupported op for finding the root alloc` | 复杂指针表达式、`tl.where`参与offset、复杂broadcast临时量 | load前计算合法mask，避免在内存偏移中使用`tl.where`，简化指针表达式 |
| tl.where偏移计算 | 编译失败（Ascend后端） | 在内存偏移中使用tl.where | 改用if-else静态分支处理 |
| 性能低下 | 运行缓慢 | 内存访问不连续、切分不合理 | 优化内存布局、调整BLOCK_SIZE、使用block_ptr |
