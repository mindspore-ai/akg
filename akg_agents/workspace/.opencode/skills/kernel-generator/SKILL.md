---
name: kernel-generator
description: >
  算子内核代码生成 Skill — 负责算子实现的全部智力工作：方案讨论、代码生成、基于反馈修改。
  支持多种 DSL：triton_cuda、triton_ascend、cpp、cuda_c、tilelang_cuda、pypto。
argument-hint: >
  必需：op_name、task_desc、framework、backend、arch、dsl。
  可选：previous_code、verifier_error、conductor_suggestion、user_requirements。
---

# 算子代码生成 Skill

<role>
你是算子内核代码生成专家。你负责算子实现的全部智力工作：方案讨论、代码生成、基于反馈修改。所有这些都需要 DSL 参考知识支撑，因此都在本 skill 中完成。
</role>

## 工作模式

本 skill 根据输入自动判断工作模式：

| 模式 | 触发条件 | 行为 |
|------|---------|------|
| 方案讨论 | `user_requirements` 是讨论性质（如"分析一下用什么策略"） | 加载参考文档 → 分析算子特征 → 输出方案建议（不生成代码文件） |
| 首次生成 | 无 `previous_code` | 加载参考文档 → 分析任务 → 生成代码 |
| 修复生成 | 有 `previous_code` + 有 `verifier_error` | 加载参考文档 → 分析错误 → 修复代码 |
| 纯修改 | 有 `previous_code` + 有 `user_requirements` + 无 `verifier_error` | 跳过参考文档 → 按用户需求修改代码 |

无论哪种模式，所有代码变更都必须通过本 skill 的代码生成步骤产出，禁止在 skill 外直接编辑代码文件。

## 生成流程

### Step 1: 加载参考文档

**纯修改模式跳过此步骤。** 其余模式根据参数，用 `read` 工具读取参考文档。

> 本 skill 加载后，`<base_url>` 标签提供 skill 目录路径（记为 **`$SD`**）。所有参考文档路径基于 `$SD/references/`。
>
> ⚠️ 参考文档路径下的文件虽然名为 `SKILL.md`，但在此上下文中它们是**参考内容文件**。请使用 `read` 工具按文件路径读取，**不要**使用 `skill` 工具加载。

#### 1.1 代码生成格式规范（必加载）

**始终加载**，按 `dsl` + `framework` 选择对应文件（路径：`$SD/references/format/`）：

| dsl | framework | 文件名 |
|-----|-----------|--------|
| `triton_cuda` / `triton_ascend` | `torch` | `triton-torch.md` |
| `triton_cuda` / `triton_ascend` | `mindspore` | `triton-mindspore.md` |
| `cpp` | `torch` | `cpp-torch.md` |
| `cuda_c` | `torch` | `cuda-c-torch.md` |
| `torch`（原生转换） | `torch` | `torch-native.md` |

#### 1.2 硬件规格（按 arch 加载）

路径：`$SD/references/hardware/{文件名}`

| arch 前缀 | 文件名 |
|-----------|--------|
| `a100` | `CUDA_A100.md` |
| `h20` | `CUDA_H20.md` |
| `l20` | `CUDA_L20.md` |
| `rtx3090` | `CUDA_RTX3090.md` |
| `ascend910b1` | `Ascend910B1.md` |
| `ascend910b2` | `Ascend910B2.md` |
| `ascend910b2c` | `Ascend910B2C.md` |
| `ascend910b3` | `Ascend910B3.md` |
| `ascend910b4` | `Ascend910B4.md` |
| `ascend310p3` | `Ascend310P3.md` |
| `ascend910_9362` | `Ascend910_9362.md` |
| `ascend910_9372` | `Ascend910_9372.md` |
| `ascend910_9381` | `Ascend910_9381.md` |
| `ascend910_9382` | `Ascend910_9382.md` |
| `ascend910_9391` | `Ascend910_9391.md` |
| `ascend910_9392` | `Ascend910_9392.md` |
| `ascend950dt_95a` | `Ascend950DT_95A.md` |
| `ascend950pr_950z` | `Ascend950PR_950z.md` |
| `ascend950pr_9572` | `Ascend950PR_9572.md` |
| `ascend950pr_9574` | `Ascend950PR_9574.md` |
| `ascend950pr_9575` | `Ascend950PR_9575.md` |
| `ascend950pr_9576` | `Ascend950PR_9576.md` |
| `ascend950pr_9577` | `Ascend950PR_9577.md` |
| `ascend950pr_9578` | `Ascend950PR_9578.md` |
| `ascend950pr_9579` | `Ascend950PR_9579.md` |
| `ascend950pr_957b` | `Ascend950PR_957b.md` |
| `ascend950pr_957d` | `Ascend950PR_957d.md` |
| `ascend950pr_9581` | `Ascend950PR_9581.md` |
| `ascend950pr_9582` | `Ascend950PR_9582.md` |
| `ascend950pr_9584` | `Ascend950PR_9584.md` |
| `ascend950pr_9587` | `Ascend950PR_9587.md` |
| `ascend950pr_9588` | `Ascend950PR_9588.md` |
| `ascend950pr_9589` | `Ascend950PR_9589.md` |
| `ascend950pr_958a` | `Ascend950PR_958a.md` |
| `ascend950pr_958b` | `Ascend950PR_958b.md` |
| `ascend950pr_9591` | `Ascend950PR_9591.md` |
| `ascend950pr_9592` | `Ascend950PR_9592.md` |
| `ascend950pr_9599` | `Ascend950PR_9599.md` |

#### 1.3 Triton Ascend API 文档（条件加载）

仅当 `dsl` 为 `triton_ascend` 时加载。

Triton Ascend API 文档是一份 markdown 格式的参考手册，包含当前环境中可用的全部 Triton 语言 API（`tl.load`、`tl.store`、`tl.dot`、`tl.sum`、`tl.arange` 等）的**函数签名、参数说明和使用示例**，按内核装饰器、程序 ID、内存操作、数学运算、规约操作等分类组织。生成 triton_ascend 代码时**必须查阅此文档**以确保 API 调用正确。

**获取方式**（按优先级）：

1. **运行时获取**（推荐）：通过命令模板执行以下命令，将当前 SDK 环境的 API 文档导出到临时文件：
   ```bash
   <命令模板> python -c "
   from akg_agents.op.utils.triton_ascend_api_docs import load_triton_ascend_api_docs
   import os, tempfile
   docs = load_triton_ascend_api_docs()
   path = os.path.join(tempfile.gettempdir(), 'triton_ascend_api_docs.md')
   with open(path, 'w') as f:
       f.write(docs)
   print(path)
   "
   ```
   命令输出一行文件路径，用 `read` 工具读取该文件即可获得完整 API 文档。此方式根据当前 SDK 版本过滤不存在的 API，返回与环境匹配的文档。

2. **离线快照**：若上述命令失败，读取离线文件：`$SD/references/triton-ascend-api/api.md`

如需查阅单个 API 详细文档，同目录下有按 API 名命名的独立文件（如 `triton.language.load.md`）。

#### 1.4 DSL 编程参考（分层选择）

参考文档按 DSL 组织在 `$SD/references/dsl-guides/` 下，每个 DSL 对应一个目录，内部按 category 分子目录。

**dsl → 目录映射**（⚠️ 目录名用连字符）：

| dsl 参数 | dsl-dir |
|----------|---------|
| `triton_ascend` | `triton-ascend` |
| `triton_cuda` | `triton-cuda` |
| `tilelang_cuda` | `tilelang-cuda` |
| `cuda_c` | `cuda-c` |
| `cpp` | `cpp` |
| `pypto` | `pypto` |

每个 DSL 目录下的子目录结构与 category 对应：

| 子目录 | category | Layer | 说明 |
|--------|----------|-------|------|
| `fundamentals/` | fundamental | 0 | 基础知识、API、内存模型 |
| `guides/` | guide | 1 | 按算子类型分的优化指南 |
| `examples/` | example | 1 | 代码示例（按 framework 区分） |
| `cases/` | case | 2 | 修复/优化案例 |
| `evolved-fix/` | case (fix) | 2 | 自动演化的错误修复案例 |

> 并非所有 DSL 都有全部子目录。如某子目录不存在则跳过。

**读取方式**：每个子目录下的参考以 `{name}/SKILL.md` 形式存放。例如：
```
$SD/references/dsl-guides/triton-ascend/fundamentals/triton-ascend-basics/SKILL.md
$SD/references/dsl-guides/triton-ascend/guides/triton-ascend-matmul/SKILL.md
```

##### 加载规则

| 状态 | 判断依据 | 加载范围 |
|------|---------|---------|
| **首次生成** | 无 `verifier_error`，无 `previous_code` | Layer 0 + Layer 1 |
| **修复生成** | 有 `verifier_error` | Layer 0 + Layer 1 + Layer 2 |

**Layer 0（始终加载）**：读取当前 dsl 目录下 `fundamentals/` 中的所有参考。

**Layer 1（选择性加载）**：
- **指南**：浏览 `guides/` 目录下各子目录名，选出与当前算子最相关的 1-2 个指南读取
- **示例**：浏览 `examples/` 目录下各子目录名，加载与选中 guide 算子类型匹配且 framework 相符的示例

**Layer 2（仅修复生成时）**：有 `verifier_error` 时，浏览 `cases/` 和 `evolved-fix/` 目录，选择与当前错误相关的案例读取。

### Step 2: 分析任务

1. 解析 `task_desc` 中的 `Model` 类，理解算子的计算逻辑
2. 分析 `get_inputs()` 和 `get_init_inputs()` 确定输入输出规格
3. 识别算子类型（elementwise、reduce、matmul、attention 等）
4. 结合已加载的参考文档，确定优化策略（分块、并行、内存布局等）
5. 如有 `user_requirements`，基于参考文档评估用户建议的可行性并纳入方案

**方案讨论模式**：完成分析后输出方案建议，不生成代码文件，等待用户确认或进一步讨论。
**其他模式**：继续进入 Step 3。

### Step 3: 生成代码

按已加载的代码生成格式规范（Step 1.1）生成包含 `ModelNew` 类的完整 Python 文件。

**首次生成**：基于任务描述和参考文档
**修复生成**：综合 `previous_code` + `verifier_error` + `conductor_suggestion` 修复问题
**纯修改**：基于 `previous_code` + `user_requirements` 修改

无论哪种模式，都必须输出**完整的代码文件**（不是 diff / 补丁）。调用方会将完整代码保存到新的 iteration 目录。

## 约束

| 约束 | 说明 |
|------|------|
| 类名 | 必须为 `ModelNew` |
| 接口一致 | `__init__` 和 `forward` 签名必须与 `Model` 兼容 |
| 返回一致 | 输出的类型、形状、数量必须与 `Model` 一致 |
| 自包含 | 所有 kernel 函数和辅助函数在同一文件 |
| 正确性优先 | 优先保证正确性，其次追求性能 |
| 完整输出 | 每次都输出完整代码文件，禁止输出 diff / 补丁，禁止用 `edit` 修改已有 iteration 目录中的文件 |
