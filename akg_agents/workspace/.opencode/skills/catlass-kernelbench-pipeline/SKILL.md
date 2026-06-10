---
name: "catlass-kernelbench-pipeline"
description: >
  KernelBench 端到端流水线：按当前 NPU 架构（CATLASS_ARCH 2201/3510）选取并改写 catlass example，调用 catlass-example-to-torch-intf 生成 torch.ops.catlass 与 catlass_op，交付 kernel.py/reference.py，并以 ref 做精度对比与 NPU device 侧性能测试。
argument-hint: >
  需要提供：1) KernelBench ref 文件路径；2) catlass 仓库本地路径；3) 可选：指定的 catlass example 目录名。
  示例：/catlass-kernelbench-pipeline ref=level1/63_conv_standard_2D__square_input__square_kernel.py catlass_root=/home/user/catlass
---

# Catlass KernelBench Pipeline Skill

将 KernelBench 任务（torch ref 代码）通过 catlass 高性能 NPU 算子库实现，端到端跑通：

```
Ref → 找相似 catlass example → 修改适配 → skill 转换(自动编译验证) → KernelBench 格式交付 + device侧性能测试
```

## 用户必须提供的参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `catlass_root` | catlass 仓库本地路径 | `D:\cwy\catlass` |
| `ref_file` | KernelBench ref 文件路径 | `level1/63_conv_standard_2D__square_input__square_kernel.py` |
| `catlass_example` | 最相似的 example 目录名（可选，skill 可自动推荐） | `33_basic_conv2d` |

## 架构与 example 选型（`CATLASS_ARCH`）

以**当前机器 NPU 代际**为准（`npu-smi info`），全程 `CATLASS_ARCH`、所选 example、`ArchTag`、编译选项须一致；**禁止** 2201 样例上 950 机或反之。

| 硬件 | `CATLASS_ARCH` | example 特征 |
|------|----------------|----------------|
| Atlas A2 / A3 | `2201` | 如 `00_`、`33_`（无 `ascend950` 前缀） |
| Ascend 950PR / 950DT | `3510` | 如 `43_ascend950_*`、`56_ascend950_*` |

编译：`-DCATLASS_ARCH=2201` 或 `3510`（`scripts/build.sh` 与 `catlass_op` 相同）。改 example 时 `ArchTag`/DispatchPolicy 须与上表一致，勿只改 MNK。

## 流程步骤

### Step 0: 环境准备

1. **复制 catlass 转换 skill 到 `.claude/skills/`**：

   catlass 转换 skill 在 `.agents/skills/` 下，Claude Code 只能自动发现 `.claude/skills/`。执行：

   ```bash
   cp -r <catlass_root>/.agents/skills/catlass-example-to-torch-intf <catlass_root>/.claude/skills/catlass-example-to-torch-intf
   ```

   复制后验证 `<catlass_root>/.claude/skills/catlass-example-to-torch-intf/SKILL.md` 存在。

2. **确认工作目录**：所有产物放在 `akg_agents/workspace/output/<task_name>/` 下。

3. **确认 NPU 环境**：`ASCEND_HOME_PATH` 已设置，`torch_npu` 可用；**`CATLASS_ARCH` 与 example 选型**见上一节。

### Step 1: 分析 Ref + 找最相似 catlass Example

1. **Read ref 文件**，提取：
   - 算子类型（matmul / conv / gemv 等）
   - 输入 shape 和 dtype
   - 特殊属性（bias / stride / padding / dilation / groups / activation 等）

2. **Read catlass example 目录列表**（`ls <catlass_root>/examples/`），在上一节 **架构 + 算子大类** 限定池内，按 ref **数据特征** 找最相似、后续改得最少的 example：

   - **shape**：从 ref 读出 M/N/K（或 conv 的 H/W/C 等）及量级、是否 batch/group、是否需 padding 等；对照候选目录的 `README.md`，选**面向同类 shape 场景**的模板，避免用「通用大 GEMM」硬套小 K、分组等场景
   - **dtype**：默认与 ref 一致；若 ref 非 fp16 且当前为 A2 (2201)，优先选 README/`.cpp` 里已有同 dtype 的候选。**例外**：A2 上 ref 为 fp32 而用户坚持用 fp32 容差验收的，不支持转换——A2 Cube 无原生 fp32 乘法器，输出精度只能到 fp16 级别，见「dtype 精度兼容矩阵」
   - **融合**：ref 含 bias、激活、标量组合、反量化等 → 优先选已带对应 Epilogue（或 quant 链路）的候选，勿从「纯 matmul/conv」从零拼融合
   - **conv 参数**：stride、padding、kernel、dilation、groups 等与 ref 最接近；参数形态差太多的（如 ref 非对称 kernel、候选仅 square）降低优先级

   对少数最接近的目录各扫一眼 `README.md` 再定稿；KernelBench **文件名仅作线索**，以 ref 代码里的 shape/算子为准。

3. **Read 选定 example 的 `README.md` + `.cpp`**，理解其：
   - 类型别名区域（ElementA/B/C, LayoutA/B, ArchTag, DispatchPolicy, TileShape, Epilogue）
   - Arguments 结构（`<Kernel>::Arguments` 的字段）
   - 与 ref 的差异点

4. **输出修改方案**：列出需要修改的类型别名和参数，向用户确认。

### Step 2: 修改 Example + 调用转换 Skill

1. **在 catlass 仓库内修改 example**：

   - 复制 example 目录到新名字（如 `33_basic_conv2d` → `33_conv2d_fp32_bias`）
   - 修改 `.cpp` 文件中的类型别名、shape 参数、Epilogue 等来适配 ref
   - 修改量取决于 ref 和 example 的差异：
     * 简单：改 dtype（half → float / bfloat16_t）
     * 中等：改 Epilogue（void → bias / relu / silu）
     * 较难：改 DispatchPolicy / TileShape / 增加 workspace

   **修改禁忌**（违反会导致转换 skill 无法工作）：
   - ❌ 不要加 FFTS 代码（`rtGetC2cCtrlAddr`、`SetSyncBaseAddr`、`fftsAddr`）— 转换 skill Rule 4 禁止
   - ❌ 不要把 example 改成手动 `<<<>>>` / 裸指针 launch 模式 — 源 example 宜保持「单 Kernel + DeviceGemm/RunAdapter」形态，便于转换 skill 抽取 `Kernel::Params`
   - ❌ 不要把单阶段 kernel 改成多阶段串联（如加中间 softmax/reduce）— 不再是上述可转换形态
   - ❌ 不要改 ACL 初始化/golden/清理代码 — 转换 skill 只提取 Part 1，其他会被丢弃

   **可以改的**（这些是类型别名组装区域，转换 skill 正好提取这部分）：
   - ✅ ElementA/B/C dtype（half / float / bfloat16_t）
   - ✅ LayoutA/B 格式（RowMajor / ColumnMajor 等）
   - ✅ ArchTag（AtlasA2 / Ascend950）
   - ✅ DispatchPolicy 类型（Pingpong / Preload / PreloadAsync 等）
   - ✅ TileShape 参数（L1/L0 block 大小）
   - ✅ Epilogue 类型（void / bias / relu / silu / gelu / dequant）
   - ✅ shape 参数（problem shape 默认值等）
   - ✅ BlockScheduler 的 swizzle offset/direction

   如果用户需求无法通过上述可改项实现（如需要多阶段串联、FFTS、手动 launch），必须向用户说明原因，并建议替代方案。

2. **在终端执行 `claude -p` 调用 catlass 转换 skill，禁止自行手动转换**：

   ```bash
   cd <catlass_root> && claude -p "使用 /catlass-example-to-torch-intf skill 将 examples/<修改后的example名> 转换为 PyTorch 接口"
   ```
   timeout 建议设 30 分钟（1800000ms）

3. **转换 skill 自动编译+验证**。产出在 `<catlass_root>/catlass_op/` 下。检查转换结果的 ref 是否与我们提供的公式和 shape 一致，如果不一致则需要重新修改并转换

4. **如果转换有问题**：分析编译/运行错误，回到修改步骤重试

### Step 3: KernelBench 格式交付 + 精度验证 + device 侧性能测试

将 catlass_op 产物搬到 workspace 并包装成 KernelBench 格式。**不要手写测试逻辑**——复制本 skill 固定模板即可。

1. **创建交付目录**：`mkdir -p <workspace>/output/<task_name>/`

2. **复制 catlass_op 到交付目录**：`cp -r <catlass_root>/catlass_op <workspace>/output/<task_name>/catlass_op`

3. **拷贝 KernelBench ref**：`cp <ref_file> <workspace>/output/<task_name>/reference.py`

4. **编写 `kernel.py`**：参考 `reference/kernelbench_template.py`，`forward()` 签名必须与 ref 一致

5. **复制验收脚本**：

   ```bash
   cp <this_skill>/reference/npu_perf_test_template.py \
      <workspace>/output/<task_name>/test_<task_name>.py
   ```

   - 脚本放在 **任务根目录**（与 `reference.py`、`kernel.py` 同级）。
   - 运行前确保 `catlass_op/build/libcatlass.so` 已存在（转换 skill 编译过即可）。

   ```bash
   cd <workspace>/output/<task_name>
   python test_<task_name>.py --dtype auto          # 默认：精度 + NPU profiler
   python test_<task_name>.py --dtype float16 --skip-perf
   ```

   | CLI | 说明 |
   |-----|------|
   | `--dtype` | `auto`（用 ref 输出 dtype 选容差表）或 `float16` / `float32` / `bfloat16` |
   | `--warmup` / `--active` | NPU profiler 预热与采样步数（默认 25 / 50） |
   | `--skip-perf` | 只做精度，不写性能段 |
   | `--result` | 输出 JSON 路径（默认 `bench_result.json`） |

   **分工**：
   - `catlass_op/test_<func_name>.py`（转换 skill 产物）→ 仅编译/单算子冒烟
   - `test_<task_name>.py`（本模板复制）→ **唯一** KernelBench 交付验收（ref vs ModelNew + 可选 perf）

6. **交付产物结构**：

   ```
   output/<task_name>/
   ├── kernel.py                    # KernelBench ModelNew 格式
   ├── reference.py                 # 原 KernelBench ref（拷贝过来）
   ├── test_<task_name>.py          # 从 reference/npu_perf_test_template.py 复制
   ├── catlass_op/                  # 编译后的 catlass 项目
   │   ├── build/libcatlass.so
   │   ├── kernel/catlass_kernel.asc
   │   ├── src/catlass_torch.cpp
   │   ├── include/
   │   ├── CMakeLists.txt
   │   └── test_<func_name>.py      # 转换 skill 的原始验证脚本
   └── bench_result.json            # 性能结果
   ```

## 重要规则

### 1. conv 算子的 Layout 转换（仅 conv 任务）

conv 算子使用特殊 5D Layout（`NC1HWC0` / `CI1KHKWCOCI0`），torch 的 NCHW 格式需要转换。卷积类任务须在 `ModelNew.__init__` 中通过固定随机种子 `torch_npu.npu.manual_seed(0)`，构建 `weight`/`bias`，确保与 reference `Model` 权重一致。

在 `catlass_torch.cpp` 中：
- 输入 fmap：`torch_npu` 自动做 NCHW → NC1HWC0 的 5D format 转换
- 输出：同上，torch_npu 自动转换回 NCHW
- weight：需要用 `torch_npu` 的 `npu_format_cast` 或在 torch binding 里显式转换

### 2. Device 侧性能测试

统一使用 `reference/npu_perf_test_template.py`（复制后运行，勿重写）：
- **推荐**：`torch_npu.profiler.profile` + `op_statistic.csv` 解析 → 纯 NPU 侧时间
- **禁止**：纯 host 侧计时（无 synchronize）；禁止在交付目录手写另一套 `allclose` 阈值

### 3. 修改只改 Part 1

修改 example 时只改类型别名区域（Part 1），不改：
- ACL 初始化代码
- H2D 数据拷贝代码
- Golden 计算代码
- 资源清理代码

因为转换 skill 只提取 Part 1，其他部分会被丢弃。

## catlass Example 类型速查

### 可转换（单 Kernel + DeviceGemm/RunAdapter）

| 类型 | Examples | 核心类型别名 |
|------|---------|-------------|
| 纯 matmul | 00,01,02,05,08,09,21,25,31,34,37 | `BlockMmad + BasicMatmulKernel + DeviceGemm` |
| matmul+Epilogue | 03(add),20(bias),26(relu),27(gelu),28(silu) | `BlockMmad + Epilogue + MmadAtlasA2*Bias` |
| quant matmul | 07,10,11,12,29,30,32,38 | `PrologueB TileCast* + Epilogue*Dequant` |
| GEMM | 15,16 | `EpilogueAtlasA2Gemm + alpha/beta` |
| conv | 24(conv3d+bias),33(conv2d) | `BlockConv/BlockConv2d + ConvBias/BasicConv2dKernel + DeviceConv` |
| gemv | 17(aiv),18(aic) | `BlockGemv + DeviceGemv` |
| TLA matmul | 13,14,39,41,42 | `TLA 版 DeviceGemm` |
| dynamic | 102,103 | `动态 dispatch` |

### 不可转换（手动 launch / FFTS）

| Example | 原因 |
|---------|------|
| 19(MLA) | 多阶段 BlockMmad 串联 + 自定义 softmax |
| 23,40(FAI) | 多阶段 + FFTS + 13 参数 + 自定义 tiling |
| 49(950 FAI) | 同 FAI，950 版 |

## dtype 精度兼容矩阵

catlass 所有 matmul 底层走 `AscendC::Mmad` → Cube 单元。Cube 在不同代际上支持的乘法精度不同，导致最终输出精度存在硬件上限。

| Arch | dtype | Cube 乘法精度 | 累加精度 | 实际输出精度 | 能否过 fp32 容差 | 建议 |
|------|-------|-------------|---------|------------|-----------------|------|
| **2201 (A2)** | fp32 | **fp16** (硬件截断) | fp32 | fp16 级别 | ❌ hard_fail~50 | `--dtype float16` |
| **2201 (A2)** | fp16 | fp16 | fp32 | fp16 级别 | ✅ | `--dtype float16` |
| **2201 (A2)** | bf16 | bf16 | fp32 | bf16 级别 | ✅ | `--dtype bfloat16` |
| **3510 (950)** | fp32 | **fp32** (原生) | fp32 | fp32 级别 | ✅ | `--dtype float32` |
| **3510 (950)** | fp16 | fp16 | fp32 | fp16 级别 | ✅ | `--dtype float16` |


## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 编译找不到 catlass include | CMakeLists.txt 中 `../include` 路径不对 | 确保 catlass_op 在 catlass 仓库内，或修改 `CATLASS_INCLUDE_DIR` |
| `TORCH_LIBRARY` 注册失败 | `.so` 未正确编译 | 检查 `libcatlass.so` 是否生成，`torch.ops.load_library` 路径是否正确 |
| conv 输出 shape 不对 | Layout format 转换问题 | 检查 `Conv2dParams` 参数是否与 ref 一致 |
| 精度验证 max_diff 过大 | dtype/TileShape/Epilogue 不匹配 | 回到 Step 1 检查修改方案 |
| 编译报 arch / dav- 相关错误 | `CATLASS_ARCH` 与 example 代际不一致 | 按「架构与 example 选型」重选 |
| `claude -p` 找不到 skill | `.claude/skills/` 下没有转换 skill | Step 0 的复制操作未完成 |