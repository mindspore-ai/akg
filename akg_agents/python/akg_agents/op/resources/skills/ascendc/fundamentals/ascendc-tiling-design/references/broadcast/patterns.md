# Broadcast 类算子场景路由

> 本文档用于**场景判定**和**策略选择**。确定场景后，按链接进入对应详细文档。

---

## 合轴（DimensionCollapse）

合轴是所有 Broadcast 分支的公共前置步骤，目的是减少维度以简化 Kernel 循环。

### 规则

1. **补维**：输入 shape 维度不足输出时，左侧补 1
2. **标记广播轴**：为每个轴计算 flag 位图，第 j 个输入在该轴 dim=1 → flag 第 j bit 置 1
3. **合并相邻同 flag 轴**：相邻两轴如果所有输入的 flag 相同 → 合并（维度相乘）
4. **计算 stride**：从右到左累乘；广播轴（dim=1 而输出 dim>1）stride 置 0

### 示例

**Add(x=[4,3,8], y=[1,3,8])**：
```
补维后：x=[4,3,8], y=[1,3,8], out=[4,3,8]

flag 计算（2 个输入，bit0=y, bit1=x）：
  轴 0: x=4≠1, y=1 → flag=01 (y 需要广播)
  轴 1: x=3≠1, y=3≠1 → flag=00
  轴 2: x=8≠1, y=8≠1 → flag=00

合轴：轴 1 和轴 2 flag 都是 00 → 合并
  x: [4, 24]    strides: [24, 1]
  y: [1, 24]    strides: [0,  1]   ← 轴 0 stride=0，需要广播
  out: [4, 24]  strides: [24, 1]
```

**Mul(x=[2,1,4], y=[2,3,1])**：
```
flag 计算：
  轴 0: flag=00
  轴 1: x=1 → flag=10 (x 需要广播)
  轴 2: y=1 → flag=01 (y 需要广播)

合轴：轴 0 的 flag=00，轴 1 的 flag=10 → 不同，不合并
  x: [2, 1, 4]    strides: [4, 0, 1]
  y: [2, 3, 1]    strides: [3, 1, 0]
  out: [2, 3, 4]  strides: [12, 4, 1]
```

---

## 场景判定流程

```
给定: N 个输入 shape + 1 个输出 shape（所有输入连续）

Step 0: 补维 + 合轴（见上节）
  得到 dims[N+1][≤8] + strides[N+1][≤8]

Step 1: 分支判定
  ├─ 合轴后只剩 1 维 → OneDim（纯 Elementwise，可能有标量输入） → [onedim.md]
  │
  └─ 合轴后 > 1 维 → 选择广播方式：
      ├─ DAV_2201 → UB Broadcast（优先静态接口，不满足对齐约束时用搬运指令方案） → [ub-broadcast.md]
      │
      └─ DAV_3510 → 广播发生在哪个阶段？
          ├─ GM→UB 搬入阶段 → 见下方决策链
          │
          └─ UB 内部广播（中间计算结果需要广播）：
              → UB Broadcast 动态接口（rank 1~9） → [dynamic-ub-broadcast.md]
```

### 需要广播的输入的广播方式决策链（DAV_3510）

| 优先级 | 条件 | 选择 |
|--------|------|------|
| 1 | 用户强制指定 NDDMA 或 UB BRC | 遵从 |
| 2 | NLast 场景，尾轴 >= dcache/2 | UB BRC → [dynamic-ub-broadcast.md] |
| 3 | dtype 为 INT8/FP16/BF16 且尾轴 32B 对齐 | UB BRC → [dynamic-ub-broadcast.md] |
| 4 | 其他 | NDDMA → [nddma-broadcast.md] |

**NLast** = 尾轴不需要广播（stride≠0），但非尾轴需要广播（stride=0）。尾轴数据量大时 NDDMA 反复读会刷 dcache，不如 UB 内 Broadcast API。

---

## 通用规则

以下规则适用于所有 Broadcast 分支。

**变量说明**：

| 变量 | 含义 |
|------|------|
| `dims[i]` | 合轴后输出 shape 的第 i 维大小 |
| `shapeLen` | 合轴后的总维度数 |
| `ubSize` | UB 总容量（字节） |
| `extraSize` | 额外预留空间（字节），如 tmpBuffer 等 |
| `bufferNum` | 计算图中存活 buffer 数量（输入 + 输出 + 中间） |
| `maxDtypeBits` | 计算图中最大数据类型的位宽 |
| `minDtypeBits` | 计算图中最小数据类型的位宽 |

**UB 切分**：从最内轴向外累乘，找到第一个放不下的轴作为 ubSplitAxis：
```
curProduct = 1
ubSplitAxis = 0
allFit = true
for i = shapeLen-1 downto 0:
    curProduct *= dims[i]
    if curProduct > maxElemNum:
        ubSplitAxis = i
        curProduct /= dims[i]
        allFit = false
        break

if allFit:                              # 所有维度都放得进 UB，ubSplitAxis 保持初始值 0
    curProduct /= dims[0]               # 在最外维上切分

if shapeLen == 1:                       # 单维场景（通常已被 OneDim 拦截）
    ubFormer = maxElemNum
else:
    ubFormer = maxElemNum / curProduct

ubOuter = ceil(dims[ubSplitAxis] / ubFormer)
ubTail  = dims[ubSplitAxis] - (ubOuter-1) * ubFormer
```

其中 maxElemNum 计算：
```
maxElemNum = (ubSize - extraSize) * 8 / (bufferNum * maxDtypeBits)
maxElemNum = floor_align(maxElemNum, 256 * 8 / minDtypeBits)
```

**多核切分**：把 ubSplitAxis 及其外层轴展平，均分给多核：
```
fusedProduct = ubOuter × (ubSplitAxis 之前所有轴乘积)
blockFormer  = ceil(fusedProduct / coreNum)
blockNum     = ceil(fusedProduct / blockFormer)
blockTail    = fusedProduct - (blockNum - 1) * blockFormer
```

核利用率不足时（`blockNum < coreNum`），循环缩小 maxElemNum（每次减 CACHE_LINE），重算 ubFormer/ubOuter/fusedProduct，直到能喂满更多核。

**对齐**：
- OneDim: 128B (CACHE_LINE) 对齐
- 多维: 256B (REPEAT) 对齐

**NDDMA 超过 5 轴时的处理（DAV_3510）**：
```
axesAfterSplit = shapeLen - ubSplitAxis

≤ 5 → WITHOUT_LOOP：一次 NDDMA 调用完成（API 为 DataCopy<T, 5, config>）
> 5 → WITH_LOOP：最内 5 轴交给 NDDMA，外层轴 Kernel for-loop 遍历
```
WITH_LOOP 模式下 NDDMA config 不变，只有 GM/UB 偏移随外层循环变化。详见 [nddma-broadcast.md](nddma-broadcast.md)。

---

## 跨场景参考

| 主题 | 文档 |
|------|------|
| OneDim 分支（合轴后单维） | [onedim.md](onedim.md) |
| UB Broadcast（DAV_2201，静态接口 + 搬运指令 fallback） | [ub-broadcast.md](ub-broadcast.md) |
| UB Broadcast 动态接口（DAV_3510，rank 1~9） | [dynamic-ub-broadcast.md](dynamic-ub-broadcast.md) |
| NDDMA Broadcast（DAV_3510，GM→UB 硬件广播） | [nddma-broadcast.md](nddma-broadcast.md) |
